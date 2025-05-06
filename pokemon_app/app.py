import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re
import logging
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

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
MAX_ROWS_NO_FILTER = 200 # Límite de filas a mostrar en AgGrid si no hay filtros específicos

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
@st.cache_data(ttl=3600) # Cachear por 1 hora
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    query = f"SELECT table_id FROM `{_client.project}.{BIGQUERY_DATASET}`.__TABLES__ WHERE STARTS_WITH(table_id, 'monthly_') ORDER BY table_id DESC LIMIT 1"
    try:
        results = _client.query(query).result()
        if results.total_rows > 0:
            latest_table_id = list(results)[0].table_id
            logging.info(f"Usando tabla de precios: {latest_table_id}")
            return f"{_client.project}.{BIGQUERY_DATASET}.{latest_table_id}"
        st.warning(f"No se encontraron tablas 'monthly_...' en '{BIGQUERY_DATASET}'.")
        return None
    except Exception as e:
        st.error(f"Error buscando tabla snapshot: {e}.")
        return None

POKEMON_SUFFIXES_TO_REMOVE = [
    ' VMAX', ' VSTAR', ' V-UNION', ' V', ' GX', ' EX', ' BREAK', ' Prism Star', ' Star',
    ' Radiant', ' δ', ' Tag Team', ' & ', ' Light', ' Dark', ' ◇', ' ☆',
    # Podrías añadir más aquí, ej. ' Lv.X', ' Prime', ' LEGEND'
    # Formas específicas podrían requerir lógica más compleja que simples sufijos
]
MULTI_WORD_BASE_NAMES = [ # Nombres que deben ser tratados como una unidad
    "Mr. Mime", "Mime Jr.", "Farfetch'd", "Sirfetch'd", "Ho-Oh", "Porygon-Z", "Type: Null", 
    "Tapu Koko", "Tapu Lele", "Tapu Bulu", "Tapu Fini", "Mr. Rime", 
    "Indeedee M", "Indeedee F", "Great Tusk", "Iron Treads", # Ejemplos recientes
]

def get_true_base_name(name_str, supertype, suffixes, multi_word_bases):
    if not isinstance(name_str, str) or supertype != 'Pokémon':
        return name_str

    # Verificar si el nombre completo es un nombre base multi-palabra conocido
    for mw_base in multi_word_bases:
        if name_str.startswith(mw_base):
            # Si es un multi-palabra, ver si tiene sufijos DESPUÉS de este base.
            # Ejemplo: "Mr. Mime GX" -> base "Mr. Mime"
            # Ejemplo: "Tapu Koko V" -> base "Tapu Koko"
            # Lo que queda después del multi_word_base es lo que se compara con los sufijos.
            # Esta parte no es necesaria si el objetivo es solo "Mr. Mime" de "Mr. Mime GX".
            # La lógica actual devolverá `mw_base`.
            return mw_base 
            
    cleaned_name = name_str
    for suffix in suffixes:
        if cleaned_name.endswith(suffix):
            cleaned_name = cleaned_name[:-len(suffix)].strip()
            # No hacer break, para permitir quitar múltiples sufijos si el orden es correcto
            # o si se quieren quitar todos los que coincidan.
            # Ejemplo: "Pikachu VMAX Special Art" podría necesitar quitar " VMAX" y " Special Art"
            # (si " Special Art" estuviera en sufijos)

    # Si después de quitar sufijos conocidos, aún quedan espacios y no es un multi-word base
    # (ej. "Urshifu Single Strike VMAX"), tomar la primera palabra podría ser una opción,
    # pero puede ser demasiado agresivo. Por ahora, devolvemos el nombre limpiado de sufijos.
    # if ' ' in cleaned_name and cleaned_name not in multi_word_bases:
    #     cleaned_name = cleaned_name.split(' ')[0]
        
    return cleaned_name if cleaned_name else name_str # Devolver original si se vació

@st.cache_data(ttl=3600) # Cachear por 1 hora
def get_card_metadata_with_base_names(_client: bigquery.Client) -> pd.DataFrame:
    query = f"""
    SELECT id, name, supertype, subtypes, rarity, set_id, set_name, artist, images_large 
    FROM `{CARD_METADATA_TABLE}`
    """ # Seleccionar solo columnas necesarias
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            st.warning("No se pudo cargar metadatos de cartas desde BigQuery.")
            return pd.DataFrame()
        
        df['base_pokemon_name'] = df.apply(
            lambda row: get_true_base_name(row['name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES),
            axis=1
        )
        logging.info(f"Metadatos de cartas cargados y procesados: {len(df)} filas.")
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower():
             st.error("Error: Falta el paquete 'db-dtypes'. Añádelo a tu requirements.txt y reinicia.")
        else:
            st.error(f"Error al obtener metadatos de cartas: {e}.")
        return pd.DataFrame()

# --- Carga de Datos Inicial ---
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
all_card_metadata_df = get_card_metadata_with_base_names(bq_client)

if not LATEST_SNAPSHOT_TABLE or all_card_metadata_df.empty:
    st.error("No se pueden cargar datos esenciales (tabla de precios o metadatos de cartas). La aplicación no puede continuar.")
    st.stop()

# --- Lógica Principal de la Aplicación Streamlit ---
st.title("Explorador de Cartas Pokémon TCG")

# --- Barra Lateral: Filtros y Controles ---
st.sidebar.header("Filtros y Opciones")
options_df_for_filters = all_card_metadata_df.copy() # Usar para poblar opciones de filtros

supertype_options_list = sorted(options_df_for_filters['supertype'].dropna().unique().tolist())
select_supertype_options = ["Todos"] + supertype_options_list if supertype_options_list else ["Todos"]
selected_supertype = st.sidebar.selectbox("Categoría:", select_supertype_options, index=0, key="sb_supertype_filter")

if selected_supertype != "Todos":
    options_df_for_filters = options_df_for_filters[options_df_for_filters['supertype'] == selected_supertype]

set_options_list = sorted(options_df_for_filters['set_name'].dropna().unique().tolist())
selected_sets = st.sidebar.multiselect("Set(s):", set_options_list, key="ms_sets_filter")

if selected_sets:
    options_df_for_filters = options_df_for_filters[options_df_for_filters['set_name'].isin(selected_sets)]

name_label = "Nombre de Carta:"
name_col_for_options = 'name' # Columna del DataFrame a usar para las opciones de nombre
if selected_supertype == 'Pokémon':
    name_col_for_options = 'base_pokemon_name'
    name_label = "Pokémon (Nombre Base):"
elif selected_supertype != "Todos": # Entrenador, Energía
    name_label = f"Nombre ({selected_supertype}):"
# Si es "Todos", se queda con 'name' y "Nombre de Carta:", o podríamos ajustar más

# Asegurarse que la columna exista antes de intentar obtener unique values
if name_col_for_options in options_df_for_filters.columns:
    name_options_list = sorted(options_df_for_filters[name_col_for_options].dropna().unique().tolist())
else:
    name_options_list = []
    logging.warning(f"Columna '{name_col_for_options}' no encontrada para filtro de nombres.")

selected_names_to_filter = st.sidebar.multiselect(name_label, name_options_list, key="ms_names_filter")

if selected_names_to_filter: # Aplicar este filtro para el siguiente selector (rareza)
    if name_col_for_options in options_df_for_filters.columns:
        options_df_for_filters = options_df_for_filters[options_df_for_filters[name_col_for_options].isin(selected_names_to_filter)]

rarity_options_list = sorted(options_df_for_filters['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Rareza(s):", rarity_options_list, key="ms_rarities_filter")

sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1, key="rd_sort_order")
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"


@st.cache_data(ttl=600) # Cachear resultados de consulta por 10 minutos
def fetch_card_data(_client: bigquery.Client, latest_table_path: str, supertype_ui_filter: str | None,
                    sets_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
                    sort_direction: str, full_metadata_df: pd.DataFrame) -> pd.DataFrame:
    
    ids_to_query_df = full_metadata_df.copy() # Empezar con todos los metadatos

    # Aplicar filtros secuencialmente al DataFrame de metadatos para obtener IDs
    if supertype_ui_filter and supertype_ui_filter != "Todos":
        ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter:
        ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]
    if rarities_ui_filter:
        ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    
    if names_ui_filter:
        # Determinar la columna correcta para filtrar nombres ('base_pokemon_name' o 'name')
        actual_name_col_to_filter = 'base_pokemon_name' if supertype_ui_filter == 'Pokémon' else 'name'
        if actual_name_col_to_filter in ids_to_query_df.columns:
            ids_to_query_df = ids_to_query_df[ids_to_query_df[actual_name_col_to_filter].isin(names_ui_filter)]
        else: # Fallback si la columna no existe (no debería pasar con la lógica actual)
            logging.warning(f"Columna '{actual_name_col_to_filter}' para filtro de nombre no existe en ids_to_query_df.")

    if ids_to_query_df.empty: 
        logging.info("Filtrado local de metadatos no produjo IDs.")
        return pd.DataFrame()
    
    list_of_card_ids = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids: 
        return pd.DataFrame()

    query_sql = f"""
    SELECT meta.id, meta.name AS pokemon_name, meta.supertype, meta.set_name, meta.rarity, meta.artist, 
           meta.images_large AS image_url, prices.cm_trendPrice AS price
    FROM `{CARD_METADATA_TABLE}` AS meta 
    JOIN `{latest_table_path}` AS prices ON meta.id = prices.id
    WHERE meta.id IN UNNEST(@card_ids_param) 
    ORDER BY prices.cm_trendPrice {sort_direction}
    """ # Agregado prices.cm_trendPrice para desambiguar si 'price' existiera en meta
    
    query_params = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids)]
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    
    logging.info(f"Ejecutando consulta SQL para {len(list_of_card_ids)} IDs. Orden: {sort_direction}")
    try:
        query_job = _client.query(query_sql, job_config=job_config)
        results_df = query_job.to_dataframe()
        results_df['price'] = pd.to_numeric(results_df['price'], errors='coerce')
        logging.info(f"Consulta a BigQuery ejecutada. Se obtuvieron {len(results_df)} filas.")
        return results_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): st.error("Error: Falta 'db-dtypes'. Revisa requirements.txt.")
        else: st.error(f"Error en consulta a BigQuery: {e}.")
        logging.error(f"Error en fetch_card_data (consulta BQ): {e}", exc_info=True)
        return pd.DataFrame()

# Obtener resultados basados en filtros
results_df = fetch_card_data(bq_client, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets,
                             selected_names_to_filter, selected_rarities, sort_sql, all_card_metadata_df)

# ... (Todo el código anterior hasta la sección de AgGrid) ...

# --- Área Principal: Visualización de Resultados ---
st.header("Resultados")

# Inicializar estado de sesión para la carta seleccionada si no existe
if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
if 'last_selected_card_from_grid_for_rerun_check' not in st.session_state: # Para evitar reruns si AgGrid devuelve lo mismo
    st.session_state.last_selected_card_from_grid_for_rerun_check = None


# DataFrame para mostrar en AgGrid (potencialmente limitado si no hay filtros)
results_df_for_aggrid_display = results_df 
is_initial_unfiltered_load = (
    not selected_sets and not selected_names_to_filter and not selected_rarities and
    (selected_supertype == "Todos" or not selected_supertype)
)
if is_initial_unfiltered_load and len(results_df) > MAX_ROWS_NO_FILTER:
    st.info(f"Mostrando los primeros {MAX_ROWS_NO_FILTER} de {len(results_df)} resultados. Aplica filtros para una búsqueda más específica.")
    results_df_for_aggrid_display = results_df.head(MAX_ROWS_NO_FILTER)


if not results_df_for_aggrid_display.empty:
    display_columns_mapping = {
        'id': 'ID', 'pokemon_name': 'Nombre Carta', 'supertype': 'Categoría',
        'set_name': 'Set', 'rarity': 'Rareza', 'artist': 'Artista', 'price': 'Precio (Trend €)'
    }
    cols_in_df = [col for col in display_columns_mapping.keys() if col in results_df_for_aggrid_display.columns]
    final_display_df = results_df_for_aggrid_display[cols_in_df].copy()
    final_display_df.rename(columns=display_columns_mapping, inplace=True)
    
    price_display_col_name = display_columns_mapping.get('price')
    if price_display_col_name and price_display_col_name in final_display_df.columns:
         final_display_df[price_display_col_name] = final_display_df[price_display_col_name].apply(
             lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A"
         )

    gb = GridOptionsBuilder.from_dataframe(final_display_df)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_grid_options(domLayout='normal')
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
    gridOptions = gb.build()

    st.write("Haz clic en una fila de la tabla para ver sus detalles:")
    grid_response = AgGrid(
        final_display_df,
        gridOptions=gridOptions,
        height=500, 
        width='100%',
        data_return_mode=DataReturnMode.AS_INPUT, 
        update_mode=GridUpdateMode.SELECTION_CHANGED, # Cambiado para que solo actualice en selección
        fit_columns_on_grid_load=False, 
        allow_unsafe_jscode=True, 
        key='pokemon_aggrid_sel_optimized', # Nueva key
    )

    # --- LÓGICA DE ACTUALIZACIÓN DE SELECCIÓN ---
    current_selection_in_grid = None
    if grid_response and isinstance(grid_response.get('selected_rows'), list) and grid_response['selected_rows']:
        try: 
            # 'ID' es el nombre de la columna en final_display_df (que es results_df['id'])
            current_selection_in_grid = grid_response['selected_rows'][0]['ID'] 
        except (KeyError, IndexError) as e:
            logging.warning(f"Error al acceder a la fila seleccionada de AgGrid: {e}")
    
    # Si la selección del grid ha cambiado REALMENTE desde la última vez que actualizamos el estado de sesión
    if current_selection_in_grid is not None and \
       current_selection_in_grid != st.session_state.get('selected_card_id_from_grid'): # Usar .get para evitar KeyError
        logging.info(f"AgGrid selection changed. New selection ID: {current_selection_in_grid}")
        st.session_state.selected_card_id_from_grid = current_selection_in_grid
        # No es necesario un rerun aquí si el resto del script lee de session_state y AgGrid
        # se actualiza correctamente por sí mismo. Vamos a probar sin rerun explícito primero.
        # Si los detalles no se actualizan, se puede volver a añadir st.experimental_rerun().
        # Sin embargo, un rerun puede hacer que AgGrid pierda su estado de scroll/página.
        # El cambio a update_mode=GridUpdateMode.SELECTION_CHANGED podría ayudar.
        st.experimental_rerun() # <-- RE-AÑADIDO. Es probable que sea necesario para que la UI se actualice.

    # --- FIN LÓGICA DE ACTUALIZACIÓN ---

    st.divider()
    st.header("Detalle de Carta")

    card_to_display_details = None # Esta será una Serie de Pandas de la carta a mostrar
    
    # Priorizar la selección del grid guardada en session_state
    selected_id_to_show = st.session_state.get('selected_card_id_from_grid') # Usar .get()

    if selected_id_to_show:
        # Buscar en el results_df ORIGINAL (no el limitado/transformado para AgGrid)
        matched_rows_in_results_df = results_df[results_df['id'] == selected_id_to_show]
        if not matched_rows_in_results_df.empty:
            card_to_display_details = matched_rows_in_results_df.iloc[0]
            logging.info(f"Mostrando detalles para la carta ID (desde grid/session): {selected_id_to_show}")
        else:
            logging.warning(f"ID {selected_id_to_show} de session_state no encontrado en results_df actual.")
            # Podría pasar si results_df cambió y la carta seleccionada ya no está.
            # En este caso, se intentará el fallback de abajo.
            
    # Fallback: si no hay selección de grid válida, pero hay resultados, mostrar el primero del results_df
    if card_to_display_details is None and not results_df.empty:
        card_to_display_details = results_df.iloc[0]
        default_id_to_show = card_to_display_details.get('id')
        logging.info(f"Mostrando detalles para la primera carta por defecto ID: {default_id_to_show}")
        # Sincronizar session_state si mostramos el primero por defecto y no había selección válida previa
        # O si el ID seleccionado previamente ya no es válido.
        if selected_id_to_show is None or \
           (selected_id_to_show and card_to_display_details.get('id') != selected_id_to_show):
            if default_id_to_show and pd.notna(default_id_to_show):
                 st.session_state.selected_card_id_from_grid = default_id_to_show


    if card_to_display_details is not None and isinstance(card_to_display_details, pd.Series) and not card_to_display_details.empty:
        # ... (El código para mostrar los detalles de card_to_display_details es el mismo) ...
        card_name_detail = card_to_display_details.get('pokemon_name', "N/A")
        card_id_detail = card_to_display_details.get('id', "N/A")
        card_set_detail = card_to_display_details.get('set_name', "N/A")
        card_image_url_detail = card_to_display_details.get('image_url', None)
        card_supertype_detail = card_to_display_details.get('supertype', "N/A")
        card_rarity_detail = card_to_display_details.get('rarity', "N/A")
        card_artist_detail = card_to_display_details.get('artist', None)
        card_price_detail = card_to_display_details.get('price', None)

        col1, col2 = st.columns([1, 2])
        with col1:
            if pd.notna(card_image_url_detail):
                st.image(card_image_url_detail, caption=f"{card_name_detail} - {card_set_detail}", width=300)
            else:
                st.warning("Imagen no disponible.")
        with col2:
            st.subheader(f"{card_name_detail}")
            st.markdown(f"**ID:** `{card_id_detail}`")
            st.markdown(f"**Categoría:** {card_supertype_detail}")
            st.markdown(f"**Set:** {card_set_detail}")
            st.markdown(f"**Rareza:** {card_rarity_detail}")
            if pd.notna(card_artist_detail) and card_artist_detail:
                 st.markdown(f"**Artista:** {card_artist_detail}")
            if pd.notna(card_price_detail):
                 st.metric(label="Precio (Trend €)", value=f"€{card_price_detail:.2f}")
            else:
                 st.markdown("**Precio (Trend €):** N/A")
    else: 
        st.info("Haz clic en una carta en la tabla de resultados para ver sus detalles o aplica filtros para ver cartas.")

elif not results_df.empty and results_df_for_aggrid_display.empty : 
    st.info(f"Se encontraron {len(results_df)} resultados. Aplica filtros más específicos para visualizarlos.")

else: 
    if bq_client and LATEST_SNAPSHOT_TABLE:
        st.info("No se encontraron cartas con los filtros seleccionados.")

# ... (El resto del código, como el footer, permanece igual) ...
