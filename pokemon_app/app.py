import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re # Para expresiones regulares (limpieza de nombres)
import logging

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Pokémon TCG Explorer")
logging.basicConfig(level=logging.INFO)

# --- Constantes y Configuración de GCP ---
try:
    GCP_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
except KeyError:
    st.error("Error: 'project_id' no encontrado en los secrets de Streamlit.")
    st.stop()
except Exception as e:
    st.error(f"Error inesperado al leer secrets: {e}")
    st.stop()

BIGQUERY_DATASET = "pokemon_dataset"
CARD_METADATA_TABLE = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.card_metadata"

# --- Conexión Segura a BigQuery ---
@st.cache_resource
def connect_to_bigquery():
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Error: Sección [gcp_service_account] no encontrada en los secrets.")
            return None
        creds_json = dict(st.secrets["gcp_service_account"])
        required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"]
        missing_keys = [key for key in required_keys if key not in creds_json or not creds_json[key]]
        if missing_keys:
             st.error(f"Error: Faltan claves en [gcp_service_account]: {', '.join(missing_keys)}")
             return None
        creds = service_account.Credentials.from_service_account_info(creds_json)
        client = bigquery.Client(credentials=creds, project=GCP_PROJECT_ID)
        logging.info("Conexión a BigQuery establecida.")
        return client
    except Exception as e:
        st.error(f"Error al conectar con BigQuery: {e}.")
        logging.error(f"Error al conectar con BigQuery: {e}", exc_info=True)
        return None

bq_client = connect_to_bigquery()
if bq_client is None:
    st.stop()

# --- Funciones Auxiliares ---
@st.cache_data(ttl=3600)
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    query = f"SELECT table_id FROM `{_client.project}.{BIGQUERY_DATASET}`.__TABLES__ WHERE STARTS_WITH(table_id, 'monthly_') ORDER BY table_id DESC LIMIT 1"
    try:
        results = _client.query(query).result()
        if results.total_rows > 0:
            latest_table_id = list(results)[0].table_id
            return f"{_client.project}.{BIGQUERY_DATASET}.{latest_table_id}"
        st.warning(f"No se encontraron tablas 'monthly_...' en '{BIGQUERY_DATASET}'.")
        return None
    except Exception as e:
        st.error(f"Error buscando tabla snapshot: {e}.")
        return None

# Lista (incompleta) de sufijos comunes a eliminar para obtener el nombre base.
POKEMON_SUFFIXES_TO_REMOVE = [
    ' VMAX', ' VSTAR', ' V-UNION', ' V', ' GX', ' EX', ' BREAK', ' Prism Star', ' Star',
    ' Radiant', ' δ', # Delta Species
    ' Tag Team', ' & ', # Para Tag Teams
    ' Light', ' Dark', # Variantes Light/Dark
    # Símbolos especiales (requieren cuidado con el encoding si se leen de archivos)
    ' ◇', ' ☆',
    # Nombres de formas que a veces se pueden tratar como sufijos
    # ' Alolan', ' Galarian', ' Hisuian', # Podría ser mejor mantenerlos si se quiere distinguir
    # Formas específicas si se quiere agrupar más (ej. 'Unown A', 'Unown B' -> 'Unown')
    # ' A', ' B', ' C', ... (para Unown, por ejemplo)
]
# Casos especiales con múltiples palabras que son parte del nombre base
MULTI_WORD_BASE_NAMES = ["Mr. Mime", "Mime Jr.", "Farfetch'd", "Sirfetch'd", "Ho-Oh", "Porygon-Z", "Type: Null", "Tapu Koko", "Tapu Lele", "Tapu Bulu", "Tapu Fini", "Mr. Rime", "Indeedee M", "Indeedee F"]

def get_true_base_name(name_str, supertype, suffixes, multi_word_bases):
    if not isinstance(name_str, str) or supertype != 'Pokémon':
        return name_str

    # Primero, verificar si el nombre completo es un nombre base multi-palabra conocido
    for mw_base in multi_word_bases:
        if name_str.startswith(mw_base): # Usar startswith para capturar "Mr. Mime V" como "Mr. Mime"
            # Ahora, si es un multi-palabra, hay que ver si tiene sufijos después
            potential_base = mw_base
            remaining_part = name_str[len(mw_base):].strip()
            
            # Intentar quitar sufijos de la parte restante
            # Esto es para casos como "Mr. Mime GX" -> base "Mr. Mime"
            temp_remaining_name = remaining_part
            suffix_removed_from_remaining = False
            for suffix in suffixes:
                if temp_remaining_name.startswith(suffix.strip()): # Comparamos el inicio de la parte restante
                    temp_remaining_name = temp_remaining_name[len(suffix.strip()):].strip()
                    suffix_removed_from_remaining = True
                    break # Asumimos que solo hay un sufijo principal después del multi-word base
            
            # Si no se quitó nada de la parte restante Y la parte restante no está vacía,
            # podría ser algo como "Zacian V-UNION Left", donde "Left" no es un sufijo general.
            # En este caso, el "base" es el multi-palabra que encontramos.
            return potential_base

    # Si no es un multi-palabra conocido, proceder con la limpieza general
    cleaned_name = name_str
    for suffix in suffixes:
        if cleaned_name.endswith(suffix): # Usar endswith para la lógica general de sufijos
            cleaned_name = cleaned_name[:-len(suffix)].strip()
            # Considerar no hacer break si un nombre puede tener múltiples sufijos a quitar
            # o si el orden de `suffixes` está bien definido (más largos primero)

    # Último recurso: si aún hay espacios, tomar la primera palabra.
    # Pero solo si no es un multi-word base ya identificado.
    # Esto es para casos muy genéricos.
    # if ' ' in cleaned_name and cleaned_name not in multi_word_bases:
    # cleaned_name = cleaned_name.split(' ')[0]

    return cleaned_name if cleaned_name else name_str


@st.cache_data(ttl=3600)
def get_card_metadata_with_base_names(_client: bigquery.Client) -> pd.DataFrame:
    """Obtiene todos los metadatos y añade una columna 'base_pokemon_name'."""
    # Seleccionar solo las columnas necesarias para minimizar transferencia de datos
    query = f"""
    SELECT id, name, supertype, subtypes, rarity, set_id, set_name, artist, images_large 
    FROM `{CARD_METADATA_TABLE}`
    """
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            st.warning("No se pudo cargar metadatos de cartas.")
            return pd.DataFrame()

        df['base_pokemon_name'] = df.apply(
            lambda row: get_true_base_name(row['name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES),
            axis=1
        )
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower():
             st.error("Error: Falta 'db-dtypes'. Añádelo a requirements.txt.")
        else:
            st.error(f"Error al obtener metadatos de cartas: {e}.")
        return pd.DataFrame()

# --- Carga de Datos Inicial ---
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
all_card_metadata_df = get_card_metadata_with_base_names(bq_client)

if not LATEST_SNAPSHOT_TABLE or all_card_metadata_df.empty:
    st.error("No se pueden cargar datos esenciales. La aplicación no puede continuar.")
    st.stop()

# --- Lógica Principal de la Aplicación Streamlit ---
st.title("Explorador de Cartas Pokémon TCG")

# --- Barra Lateral: Filtros y Controles ---
st.sidebar.header("Filtros y Opciones")

# Dataframe que se va filtrando para poblar las opciones de los selectores
options_df_for_filters = all_card_metadata_df.copy()

# 1. Filtro por Supertype
supertype_options_list = sorted(options_df_for_filters['supertype'].dropna().unique().tolist())
select_supertype_options = ["Todos"] + supertype_options_list if supertype_options_list else ["Todos"]
selected_supertype = st.sidebar.selectbox("Categoría:", select_supertype_options, index=0)

if selected_supertype != "Todos":
    options_df_for_filters = options_df_for_filters[options_df_for_filters['supertype'] == selected_supertype]

# 2. Filtro por Set
set_options_list = sorted(options_df_for_filters['set_name'].dropna().unique().tolist())
selected_sets = st.sidebar.multiselect("Set(s):", set_options_list)

if selected_sets:
    options_df_for_filters = options_df_for_filters[options_df_for_filters['set_name'].isin(selected_sets)]

# 3. Filtro por Nombre (Base o Completo)
name_label = "Nombre de Carta:"
name_col_for_options = 'name' # Por defecto, usar nombre completo

if selected_supertype == 'Pokémon':
    name_col_for_options = 'base_pokemon_name'
    name_label = "Pokémon (Nombre Base):"
elif selected_supertype != "Todos": # Entrenador, Energía
    name_label = f"Nombre ({selected_supertype}):"
# Si es "Todos", se queda con 'name' y "Nombre de Carta:"

name_options_list = sorted(options_df_for_filters[name_col_for_options].dropna().unique().tolist())
selected_names_to_filter = st.sidebar.multiselect(name_label, name_options_list)

# Aplicar filtro de nombres a options_df_for_filters (para el filtro de rareza)
if selected_names_to_filter:
    options_df_for_filters = options_df_for_filters[options_df_for_filters[name_col_for_options].isin(selected_names_to_filter)]


# 4. Filtro por Rareza
rarity_options_list = sorted(options_df_for_filters['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Rareza(s):", rarity_options_list)

# La función fetch_card_data ahora usará las selecciones directamente.
# El options_df_for_filters solo sirvió para llenar los multiselect.

# 5. Ordenamiento
sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1)
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"


# --- Construcción y Ejecución de la Consulta Principal ---
@st.cache_data(ttl=600)
def fetch_card_data(
    _client: bigquery.Client,
    latest_table_path: str,
    # Filtros recibidos de la UI
    supertype_ui_filter: str | None,
    sets_ui_filter: list,
    names_ui_filter: list, # Puede ser lista de nombres base o nombres completos
    rarities_ui_filter: list,
    sort_direction: str,
    # DataFrame de metadatos completo para filtrado local
    full_metadata_df: pd.DataFrame
    ) -> pd.DataFrame:

    # Paso 1: Filtrar el DataFrame de metadatos localmente para obtener IDs
    ids_to_query_df = full_metadata_df.copy()

    if supertype_ui_filter and supertype_ui_filter != "Todos":
        ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]

    if sets_ui_filter:
        ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]

    if rarities_ui_filter:
        ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]

    # Filtrado por nombre (usa 'base_pokemon_name' si el supertype es Pokémon, sino 'name')
    if names_ui_filter:
        if supertype_ui_filter == 'Pokémon':
            ids_to_query_df = ids_to_query_df[ids_to_query_df['base_pokemon_name'].isin(names_ui_filter)]
        else: # Para Entrenador, Energía, o si supertype_ui_filter es "Todos"
            ids_to_query_df = ids_to_query_df[ids_to_query_df['name'].isin(names_ui_filter)]
            # Considerar: si supertype es "Todos", ¿el filtro de nombre debe aplicar a base_pokemon_name para los Pokémon
            # y a 'name' para el resto? Esto se vuelve complejo. La implementación actual es más simple.

    if ids_to_query_df.empty:
        logging.info("El filtrado local de metadatos no produjo IDs.")
        return pd.DataFrame()

    list_of_card_ids = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids:
        return pd.DataFrame()

    # Paso 2: Construir la consulta SQL solo con los IDs filtrados
    query_sql = f"""
    SELECT
        meta.id,
        meta.name AS pokemon_name, -- Nombre completo de la carta
        meta.supertype,
        meta.set_name,
        meta.rarity,
        meta.artist,
        meta.images_large AS image_url,
        prices.cm_trendPrice AS price
    FROM
        `{CARD_METADATA_TABLE}` AS meta
    JOIN
        `{latest_table_path}` AS prices ON meta.id = prices.id
    WHERE meta.id IN UNNEST(@card_ids_param)
    ORDER BY prices.cm_trendPrice {sort_direction}
    """

    query_params = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids)]
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)

    param_details_str = f"card_ids_param: ARRAY<STRING> with {len(list_of_card_ids)} IDs"
    logging.info(f"Ejecutando consulta: {query_sql} con parámetros: {param_details_str}")

    try:
        query_job = _client.query(query_sql, job_config=job_config)
        results_df = query_job.to_dataframe()
        results_df['price'] = pd.to_numeric(results_df['price'], errors='coerce')
        logging.info(f"Consulta ejecutada. Se obtuvieron {len(results_df)} filas.")
        return results_df
    except Exception as e:
        if "db-dtypes" in str(e).lower():
             st.error("Error: Falta 'db-dtypes'. Añádelo a requirements.txt.")
        else:
            st.error(f"Error al ejecutar consulta principal: {e}.")
        return pd.DataFrame()

# Ejecutar la consulta
results_df = fetch_card_data(
    bq_client,
    LATEST_SNAPSHOT_TABLE,
    selected_supertype,
    selected_sets,
    selected_names_to_filter, # Esta es la lista de nombres base o completos seleccionados
    selected_rarities,
    sort_sql,
    all_card_metadata_df # Pasamos el DF de metadatos completo
)

# --- Área Principal: Visualización de Resultados ---
st.header("Resultados")

if not results_df.empty:
    display_columns_mapping = {
        'id': 'ID',
        'pokemon_name': 'Nombre Carta',
        'supertype': 'Categoría',
        'set_name': 'Set',
        'rarity': 'Rareza',
        'artist': 'Artista',
        'price': 'Precio (Trend €)'
    }
    actual_columns_to_display = [col for col in display_columns_mapping.keys() if col in results_df.columns]
    display_df = results_df[actual_columns_to_display].copy()
    display_df.rename(columns=display_columns_mapping, inplace=True)

    price_display_column = display_columns_mapping.get('price')
    if price_display_column and price_display_column in display_df.columns:
         display_df[price_display_column] = display_df[price_display_column].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    st.header("Detalle de Carta")

    if 'pokemon_name' in results_df.columns and 'id' in results_df.columns and not results_df.empty:
        card_options_for_detail_view = results_df['pokemon_name'].astype(str) + " (" + results_df['id'].astype(str) + ")"
        # Añadir una opción por defecto o manejar el caso de que no haya opciones
        if not card_options_for_detail_view.empty:
            selected_card_display_for_detail = st.selectbox(
                "Selecciona una carta de los resultados para ver detalles:",
                options=card_options_for_detail_view,
                index=0 # Seleccionar la primera por defecto si hay resultados
            )

            if selected_card_display_for_detail:
                selected_card_id_str = selected_card_display_for_detail[selected_card_display_for_detail.rfind("(")+1:selected_card_display_for_detail.rfind(")")]
                card_details_row = results_df[results_df['id'] == selected_card_id_str].iloc[0]

                col1, col2 = st.columns([1, 2])
                with col1:
                    if 'image_url' in card_details_row and pd.notna(card_details_row['image_url']):
                        st.image(card_details_row['image_url'], caption=f"{card_details_row['pokemon_name']} - {card_details_row['set_name']}", width=300)
                    else:
                        st.warning("Imagen no disponible.")
                with col2:
                    st.subheader(f"{card_details_row['pokemon_name']}")
                    if 'id' in card_details_row: st.markdown(f"**ID:** `{card_details_row['id']}`")
                    if 'supertype' in card_details_row: st.markdown(f"**Categoría:** {card_details_row['supertype']}")
                    if 'set_name' in card_details_row: st.markdown(f"**Set:** {card_details_row['set_name']}")
                    if 'rarity' in card_details_row: st.markdown(f"**Rareza:** {card_details_row['rarity']}")
                    if 'artist' in card_details_row and pd.notna(card_details_row['artist']) and card_details_row['artist']:
                         st.markdown(f"**Artista:** {card_details_row['artist']}")
                    if 'price' in card_details_row:
                         st.metric(label="Precio (Trend €)", value=f"€{card_details_row['price']:.2f}" if pd.notna(card_details_row['price']) else "N/A")
        else:
            st.info("No hay cartas en los resultados para mostrar detalles.")
    else:
        # Este caso puede ocurrir si results_df está vacío pero las columnas existen (poco probable)
        # o si las columnas clave no existen (más probable si la consulta falló y devolvió df vacío sin esas columnas).
        st.info("Selecciona filtros para ver cartas y sus detalles.")


else: # Si results_df está vacío
    if bq_client and LATEST_SNAPSHOT_TABLE: # Solo si los prerrequisitos para consultar estaban bien
        st.info("No se encontraron cartas con los filtros seleccionados.")

# --- Footer opcional ---
st.sidebar.info("Pokémon TCG Explorer v2")
