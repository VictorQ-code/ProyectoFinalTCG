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

@st.cache_data(ttl=3600)
def get_card_metadata_with_base_names(_client: bigquery.Client) -> pd.DataFrame:
    """Obtiene todos los metadatos y añade una columna 'base_pokemon_name'."""
    query = f"SELECT id, name, supertype, subtypes, rarity, set_id, set_name, artist, images_large FROM `{CARD_METADATA_TABLE}`"
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            st.warning("No se pudo cargar metadatos de cartas.")
            return pd.DataFrame()

        # Heurística para el nombre base del Pokémon: primera palabra
        # Podría necesitar mejoras para nombres compuestos como "Mr. Mime" o con guiones.
        df['base_pokemon_name'] = df.apply(
            lambda row: row['name'].split(' ')[0] if row['supertype'] == 'Pokémon' and pd.notna(row['name']) else None,
            axis=1
        )
        # Para Entrenadores y Objetos (Energía), el "nombre base" es el nombre completo
        df.loc[df['supertype'] != 'Pokémon', 'base_pokemon_name'] = df['name']

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

# 1. Filtro por Supertype (Categoría Principal)
supertype_options = sorted(all_card_metadata_df['supertype'].dropna().unique().tolist())
selected_supertype = st.sidebar.selectbox("Selecciona Categoría:", ["Todos"] + supertype_options, index=0)

# Filtrar metadata_df basado en supertype seleccionado
if selected_supertype == "Todos":
    filtered_metadata_by_supertype_df = all_card_metadata_df
else:
    filtered_metadata_by_supertype_df = all_card_metadata_df[all_card_metadata_df['supertype'] == selected_supertype]


# 2. Filtro por Set (basado en el supertype ya filtrado)
set_options = sorted(filtered_metadata_by_supertype_df['set_name'].dropna().unique().tolist())
selected_sets = st.sidebar.multiselect("Filtrar por Set:", set_options)


# 3. Filtro por Nombre Base (Pokémon, Entrenador, Objeto)
# El nombre a mostrar dependerá del supertype
if selected_supertype == 'Pokémon':
    name_options_col = 'base_pokemon_name'
    name_label = "Filtrar por Pokémon (Nombre Base):"
else: # Para Entrenador, Energía, etc., usar el nombre completo
    name_options_col = 'name'
    name_label = f"Filtrar por Nombre ({selected_supertype if selected_supertype != 'Todos' else 'Carta'}):"

# Obtener opciones de nombre del dataframe ya filtrado por supertype (y potencialmente set si se implementara filtrado en cascada más complejo)
name_options = sorted(filtered_metadata_by_supertype_df[name_options_col].dropna().unique().tolist())
selected_base_names = st.sidebar.multiselect(name_label, name_options)


# 4. Filtro por Rareza (basado en el supertype ya filtrado)
rarity_options = sorted(filtered_metadata_by_supertype_df['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Filtrar por Rareza:", rarity_options)


# 5. Ordenamiento
sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1)
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"


# --- Construcción y Ejecución de la Consulta Principal ---
@st.cache_data(ttl=600)
def fetch_card_data(
    _client: bigquery.Client,
    latest_table: str,
    supertype_filter: str | None,
    set_filters: list,
    base_name_filters: list, # Puede ser nombre base de Pokémon o nombre completo de Entrenador/Energía
    rarity_filters: list,
    sort: str,
    all_meta_df: pd.DataFrame # Pasamos el dataframe completo de metadatos
    ) -> pd.DataFrame:

    # Paso 1: Filtrar el DataFrame de metadatos localmente ANTES de construir la consulta SQL
    # Esto es más eficiente para el filtrado de nombre base si `base_name_filters` se refiere a 'base_pokemon_name'
    query_ids_df = all_meta_df.copy()

    if supertype_filter and supertype_filter != "Todos":
        query_ids_df = query_ids_df[query_ids_df['supertype'] == supertype_filter]

    if set_filters:
        query_ids_df = query_ids_df[query_ids_df['set_name'].isin(set_filters)]

    if rarity_filters:
        query_ids_df = query_ids_df[query_ids_df['rarity'].isin(rarity_filters)]

    # Filtrado por nombre (base_pokemon_name o name completo)
    if base_name_filters:
        if supertype_filter == 'Pokémon':
            query_ids_df = query_ids_df[query_ids_df['base_pokemon_name'].isin(base_name_filters)]
        else: # Para Entrenador, Energía, etc., o si supertype es "Todos" y se filtra por un nombre específico
            query_ids_df = query_ids_df[query_ids_df['name'].isin(base_name_filters)]


    if query_ids_df.empty:
        logging.info("El filtrado local de metadatos no produjo IDs, no se ejecutará consulta a BQ.")
        return pd.DataFrame()

    # Obtener la lista de IDs para la consulta SQL
    list_of_ids_to_query = query_ids_df['id'].unique().tolist()
    if not list_of_ids_to_query:
        return pd.DataFrame() # No hay IDs, no hay nada que consultar

    # Paso 2: Construir la consulta SQL solo con los IDs filtrados
    base_query = f"""
    SELECT
        meta.id,
        meta.name AS pokemon_name, -- Este será el nombre completo de la carta
        meta.supertype,
        meta.set_name,
        meta.rarity,
        meta.artist,
        meta.images_large AS image_url,
        prices.cm_trendPrice AS price
    FROM
        `{CARD_METADATA_TABLE}` AS meta
    JOIN
        `{latest_table}` AS prices ON meta.id = prices.id
    WHERE meta.id IN UNNEST(@card_ids) -- Filtramos por los IDs preseleccionados
    """
    base_query += f" ORDER BY prices.cm_trendPrice {sort}"

    params = [bigquery.ArrayQueryParameter("card_ids", "STRING", list_of_ids_to_query)]
    job_config = bigquery.QueryJobConfig(query_parameters=params)

    logging.info(f"Ejecutando consulta con {len(list_of_ids_to_query)} IDs. Orden: {sort}")

    try:
        query_job = _client.query(base_query, job_config=job_config)
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
    selected_base_names,
    selected_rarities,
    sort_sql,
    all_card_metadata_df # Pasamos el DF de metadatos completo
)

# --- Área Principal: Visualización de Resultados ---
st.header("Resultados")

if not results_df.empty:
    display_columns_mapping = {
        'id': 'ID',
        'pokemon_name': 'Nombre Completo', # Mostramos el nombre completo de la carta
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

    if 'pokemon_name' in results_df.columns and 'id' in results_df.columns:
        # Usamos el nombre completo de la carta (pokemon_name) para el selector de detalles
        card_options = results_df['pokemon_name'].astype(str) + " (" + results_df['id'].astype(str) + ")"
        selected_card_display = st.selectbox("Selecciona una carta para ver detalles:", options=card_options)

        if selected_card_display:
            selected_card_id_str = selected_card_display[selected_card_display.rfind("(")+1:selected_card_display.rfind(")")]
            card_details = results_df[results_df['id'] == selected_card_id_str].iloc[0]

            col1, col2 = st.columns([1, 2])
            with col1:
                if 'image_url' in card_details and pd.notna(card_details['image_url']):
                    st.image(card_details['image_url'], caption=f"{card_details['pokemon_name']} - {card_details['set_name']}", width=300)
                else:
                    st.warning("Imagen no disponible.")
            with col2:
                st.subheader(f"{card_details['pokemon_name']}") # Nombre completo
                if 'id' in card_details: st.markdown(f"**ID:** `{card_details['id']}`")
                if 'supertype' in card_details: st.markdown(f"**Categoría:** {card_details['supertype']}")
                if 'set_name' in card_details: st.markdown(f"**Set:** {card_details['set_name']}")
                if 'rarity' in card_details: st.markdown(f"**Rareza:** {card_details['rarity']}")
                if 'artist' in card_details and pd.notna(card_details['artist']) and card_details['artist']:
                     st.markdown(f"**Artista:** {card_details['artist']}")
                if 'price' in card_details:
                     st.metric(label="Precio (Trend €)", value=f"€{card_details['price']:.2f}" if pd.notna(card_details['price']) else "N/A")
    else:
        st.warning("No se pueden mostrar detalles de cartas.")
else:
    if bq_client and LATEST_SNAPSHOT_TABLE:
        st.info("No se encontraron cartas con los filtros seleccionados o hubo un error al consultar los datos.")

# --- Footer opcional ---
st.sidebar.info("Aplicación Pokémon TCG Explorer")
