import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re
import logging
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Pokémon TCG Explorer")
# Configurar el logging para que salga en los logs de Streamlit Cloud
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Asegura que vaya a stdout/stderr
)
logger = logging.getLogger(__name__)


# --- Constantes y Configuración de GCP ---
try:
    GCP_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
except KeyError:
    logger.error("CRITICAL: 'project_id' no encontrado en los secrets de Streamlit.")
    st.error("Error: 'project_id' no encontrado en los secrets de Streamlit.")
    st.stop()
except Exception as e:
    logger.error(f"CRITICAL: Error inesperado al leer secrets: {e}")
    st.error(f"Error inesperado al leer secrets: {e}")
    st.stop()

BIGQUERY_DATASET = "pokemon_dataset"
CARD_METADATA_TABLE = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.card_metadata"
MAX_ROWS_NO_FILTER = 200

# --- Conexión Segura a BigQuery ---
@st.cache_resource
def connect_to_bigquery():
    try:
        # ... (código de conexión igual que antes, usando logger.info/error) ...
        if "gcp_service_account" not in st.secrets:
            logger.error("CONNECT_BQ: Sección [gcp_service_account] no encontrada en los secrets.")
            st.error("Error: Sección [gcp_service_account] no encontrada en los secrets.")
            return None
        creds_json = dict(st.secrets["gcp_service_account"])
        required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"]
        missing_keys = [key for key in required_keys if key not in creds_json or not creds_json[key]]
        if missing_keys:
             logger.error(f"CONNECT_BQ: Faltan claves en [gcp_service_account]: {', '.join(missing_keys)}")
             st.error(f"Error: Faltan claves en [gcp_service_account]: {', '.join(missing_keys)}")
             return None
        creds = service_account.Credentials.from_service_account_info(creds_json)
        client = bigquery.Client(credentials=creds, project=GCP_PROJECT_ID)
        logger.info("CONNECT_BQ: Conexión a BigQuery establecida.")
        return client
    except Exception as e:
        logger.error(f"CONNECT_BQ: Error al conectar con BigQuery: {e}", exc_info=True)
        st.error(f"Error al conectar con BigQuery: {e}.")
        return None

bq_client = connect_to_bigquery()
if bq_client is None:
    st.stop()

# --- Funciones Auxiliares ---
@st.cache_data(ttl=3600)
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    # ... (código igual, usando logger.info/warning/error) ...
    query = f"SELECT table_id FROM `{_client.project}.{BIGQUERY_DATASET}`.__TABLES__ WHERE STARTS_WITH(table_id, 'monthly_') ORDER BY table_id DESC LIMIT 1"
    try:
        results = _client.query(query).result()
        if results.total_rows > 0:
            latest_table_id = list(results)[0].table_id
            logger.info(f"SNAPSHOT_TABLE: Usando tabla de precios: {latest_table_id}")
            return f"{_client.project}.{BIGQUERY_DATASET}.{latest_table_id}"
        logger.warning(f"SNAPSHOT_TABLE: No se encontraron tablas 'monthly_...' en '{BIGQUERY_DATASET}'.")
        st.warning(f"No se encontraron tablas 'monthly_...' en '{BIGQUERY_DATASET}'.")
        return None
    except Exception as e:
        logger.error(f"SNAPSHOT_TABLE: Error buscando tabla snapshot: {e}", exc_info=True)
        st.error(f"Error buscando tabla snapshot: {e}.")
        return None


POKEMON_SUFFIXES_TO_REMOVE = [
    ' VMAX', ' VSTAR', ' V-UNION', ' V', ' GX', ' EX', ' BREAK', ' Prism Star', ' Star',
    ' Radiant', ' δ', ' Tag Team', ' & ', ' Light', ' Dark', ' ◇', ' ☆',
]
MULTI_WORD_BASE_NAMES = ["Mr. Mime", "Mime Jr.", "Farfetch'd", "Sirfetch'd", "Ho-Oh", "Porygon-Z", "Type: Null", "Tapu Koko", "Tapu Lele", "Tapu Bulu", "Tapu Fini", "Mr. Rime", "Indeedee M", "Indeedee F", "Great Tusk", "Iron Treads"]

def get_true_base_name(name_str, supertype, suffixes, multi_word_bases):
    # ... (código igual) ...
    if not isinstance(name_str, str) or supertype != 'Pokémon':
        return name_str
    for mw_base in multi_word_bases:
        if name_str.startswith(mw_base):
            return mw_base 
    cleaned_name = name_str
    for suffix in suffixes:
        if cleaned_name.endswith(suffix):
            cleaned_name = cleaned_name[:-len(suffix)].strip()
    return cleaned_name if cleaned_name else name_str

@st.cache_data(ttl=3600)
def get_card_metadata_with_base_names(_client: bigquery.Client) -> pd.DataFrame:
    # ... (código igual, usando logger.info/warning/error) ...
    query = f"SELECT id, name, supertype, subtypes, rarity, set_id, set_name, artist, images_large FROM `{CARD_METADATA_TABLE}`"
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            logger.warning("METADATA: No se pudo cargar metadatos de cartas desde BigQuery (DataFrame vacío).")
            st.warning("No se pudo cargar metadatos de cartas desde BigQuery.")
            return pd.DataFrame()
        df['base_pokemon_name'] = df.apply(
            lambda row: get_true_base_name(row['name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES),
            axis=1
        )
        logger.info(f"METADATA: Metadatos de cartas cargados y procesados: {len(df)} filas.")
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower():
             logger.error("METADATA: Error de dependencia 'db-dtypes'.", exc_info=True)
             st.error("Error: Falta el paquete 'db-dtypes'. Añádelo a tu requirements.txt y reinicia.")
        else:
            logger.error(f"METADATA: Error al obtener metadatos de cartas: {e}", exc_info=True)
            st.error(f"Error al obtener metadatos de cartas: {e}.")
        return pd.DataFrame()

# --- Carga de Datos Inicial ---
logger.info("APP_INIT: Iniciando carga de datos.")
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
all_card_metadata_df = get_card_metadata_with_base_names(bq_client)

if not LATEST_SNAPSHOT_TABLE or all_card_metadata_df.empty:
    logger.critical("APP_INIT: No se pueden cargar datos esenciales. Deteniendo aplicación.")
    st.error("No se pueden cargar datos esenciales (tabla de precios o metadatos de cartas). La aplicación no puede continuar.")
    st.stop()
logger.info("APP_INIT: Carga de datos inicial completada.")

# --- Lógica Principal de la Aplicación Streamlit ---
st.title("Explorador de Cartas Pokémon TCG")

# --- Barra Lateral: Filtros y Controles ---
st.sidebar.header("Filtros y Opciones")
options_df_for_filters = all_card_metadata_df.copy()

supertype_options_list = sorted(options_df_for_filters['supertype'].dropna().unique().tolist())
select_supertype_options = ["Todos"] + supertype_options_list if supertype_options_list else ["Todos"]
selected_supertype = st.sidebar.selectbox("Categoría:", select_supertype_options, index=0, key="sb_supertype_filter")
logger.debug(f"SIDEBAR: Supertype seleccionado: {selected_supertype}")

if selected_supertype != "Todos":
    options_df_for_filters = options_df_for_filters[options_df_for_filters['supertype'] == selected_supertype]

set_options_list = sorted(options_df_for_filters['set_name'].dropna().unique().tolist())
selected_sets = st.sidebar.multiselect("Set(s):", set_options_list, key="ms_sets_filter")
logger.debug(f"SIDEBAR: Sets seleccionados: {selected_sets}")


name_label = "Nombre de Carta:"
name_col_for_options = 'name'
if selected_supertype == 'Pokémon':
    name_col_for_options = 'base_pokemon_name'
    name_label = "Pokémon (Nombre Base):"
elif selected_supertype != "Todos":
    name_label = f"Nombre ({selected_supertype}):"

if name_col_for_options in options_df_for_filters.columns:
    name_options_list = sorted(options_df_for_filters[name_col_for_options].dropna().unique().tolist())
else:
    name_options_list = []
    logger.warning(f"SIDEBAR: Columna '{name_col_for_options}' no encontrada para filtro de nombres.")
selected_names_to_filter = st.sidebar.multiselect(name_label, name_options_list, key="ms_names_filter")
logger.debug(f"SIDEBAR: Nombres seleccionados: {selected_names_to_filter} (usando columna '{name_col_for_options}')")


if selected_names_to_filter:
    if name_col_for_options in options_df_for_filters.columns:
        options_df_for_filters = options_df_for_filters[options_df_for_filters[name_col_for_options].isin(selected_names_to_filter)]

rarity_options_list = sorted(options_df_for_filters['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Rareza(s):", rarity_options_list, key="ms_rarities_filter")
logger.debug(f"SIDEBAR: Rarezas seleccionadas: {selected_rarities}")


sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1, key="rd_sort_order")
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"
logger.debug(f"SIDEBAR: Orden de precio: {sort_order}")


@st.cache_data(ttl=600)
def fetch_card_data(_client: bigquery.Client, latest_table_path: str, supertype_ui_filter: str | None,
                    sets_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
                    sort_direction: str, full_metadata_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"FETCH_DATA: Iniciando. Supertype: {supertype_ui_filter}, Sets: {len(sets_ui_filter)}, Names: {len(names_ui_filter)}, Rarities: {len(rarities_ui_filter)}")
    ids_to_query_df = full_metadata_df.copy()
    if supertype_ui_filter and supertype_ui_filter != "Todos":
        ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter:
        ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]
    if rarities_ui_filter:
        ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:
        actual_name_col_to_filter = 'base_pokemon_name' if supertype_ui_filter == 'Pokémon' else 'name'
        if actual_name_col_to_filter in ids_to_query_df.columns:
            ids_to_query_df = ids_to_query_df[ids_to_query_df[actual_name_col_to_filter].isin(names_ui_filter)]
        else:
            logger.warning(f"FETCH_DATA: Columna '{actual_name_col_to_filter}' para filtro de nombre no existe.")

    if ids_to_query_df.empty: 
        logger.info("FETCH_DATA: Filtrado local no produjo IDs.")
        return pd.DataFrame()
    list_of_card_ids = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids: 
        logger.info("FETCH_DATA: Lista de IDs vacía después de unique().")
        return pd.DataFrame()

    query_sql = f"""
    SELECT meta.id, meta.name AS pokemon_name, meta.supertype, meta.set_name, meta.rarity, meta.artist, 
           meta.images_large AS image_url, prices.cm_trendPrice AS price
    FROM `{CARD_METADATA_TABLE}` AS meta JOIN `{latest_table_path}` AS prices ON meta.id = prices.id
    WHERE meta.id IN UNNEST(@card_ids_param) ORDER BY prices.cm_trendPrice {sort_direction}"""
    query_params = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids)]
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    logger.info(f"FETCH_DATA: Ejecutando consulta SQL para {len(list_of_card_ids)} IDs. Orden: {sort_direction}")
    try:
        query_job = _client.query(query_sql, job_config=job_config)
        results_df = query_job.to_dataframe()
        results_df['price'] = pd.to_numeric(results_df['price'], errors='coerce')
        logger.info(f"FETCH_DATA: Consulta BQ OK. Filas: {len(results_df)}.")
        return results_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): 
            logger.error("FETCH_DATA: Error dependencia 'db-dtypes'.", exc_info=True)
            st.error("Error: Falta 'db-dtypes'. Revisa requirements.txt.")
        else: 
            logger.error(f"FETCH_DATA: Error en consulta BQ: {e}", exc_info=True)
            st.error(f"Error en consulta a BigQuery: {e}.")
        return pd.DataFrame()

results_df = fetch_card_data(bq_client, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets,
                             selected_names_to_filter, selected_rarities, sort_sql, all_card_metadata_df)
logger.info(f"MAIN_APP: results_df cargado con {len(results_df)} filas.")

# --- Área Principal: Visualización de Resultados ---
st.header("Resultados")

if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
    logger.info("SESSION_STATE: 'selected_card_id_from_grid' inicializado a None.")

logger.info(f"AGGRID_PREP: ID en session_state ANTES de AgGrid: {st.session_state.selected_card_id_from_grid}")

results_df_for_aggrid_display = results_df 
is_initial_unfiltered_load = (not selected_sets and not selected_names_to_filter and not selected_rarities and (selected_supertype == "Todos" or not selected_supertype))
if is_initial_unfiltered_load and len(results_df) > MAX_ROWS_NO_FILTER:
    logger.info(f"AGGRID_PREP: Limitando display a {MAX_ROWS_NO_FILTER} filas de {len(results_df)}.")
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
             lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")

    gb = GridOptionsBuilder.from_dataframe(final_display_df)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_grid_options(domLayout='normal')
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
    gridOptions = gb.build()

    st.write("Haz clic en una fila de la tabla para ver sus detalles:")
    grid_response = AgGrid(
        final_display_df, gridOptions=gridOptions, height=500, width='100%',
        data_return_mode=DataReturnMode.AS_INPUT, 
        update_mode=GridUpdateMode.SELECTION_CHANGED, # Clave para que responda a clics
        fit_columns_on_grid_load=False, allow_unsafe_jscode=True, 
        key='pokemon_aggrid_logger_test', 
    )

    newly_selected_id_from_grid = None
    if grid_response and isinstance(grid_response.get('selected_rows'), list) and grid_response['selected_rows']:
        try: 
            newly_selected_id_from_grid = grid_response['selected_rows'][0]['ID'] 
            logger.info(f"AGGRID_CLICK: Fila seleccionada en AgGrid. ID: {newly_selected_id_from_grid}")
        except (KeyError, IndexError) as e:
            logger.warning(f"AGGRID_CLICK: Error al acceder a fila seleccionada de AgGrid: {e}")
    else:
        logger.debug("AGGRID_CLICK: No hay filas seleccionadas en AgGrid o grid_response es inválido.")
    
    # Condición para actualizar el estado y re-ejecutar
    # Comprobar si newly_selected_id_from_grid tiene un valor Y es diferente del estado actual
    if newly_selected_id_from_grid is not None and \
       newly_selected_id_from_grid != st.session_state.get('selected_card_id_from_grid'):
        logger.info(f"SESSION_STATE_UPDATE: Cambiando selected_card_id de '{st.session_state.get('selected_card_id_from_grid')}' a '{newly_selected_id_from_grid}'. Llamando a rerun.")
        st.session_state.selected_card_id_from_grid = newly_selected_id_from_grid
        st.experimental_rerun() # ESENCIAL para actualizar la sección de detalles
    else:
        logger.debug(f"SESSION_STATE_UPDATE: No hay cambio en la selección de AgGrid o es la misma. ID actual en sesión: {st.session_state.get('selected_card_id_from_grid')}, ID de grid: {newly_selected_id_from_grid}")


    st.divider()
    st.header("Detalle de Carta")

    card_to_display_details = None 
    selected_id_for_details = st.session_state.get('selected_card_id_from_grid')
    logger.info(f"DETAIL_VIEW: Intentando mostrar detalles para ID (de session_state): {selected_id_for_details}")
    
    if selected_id_for_details:
        matched_rows_in_results_df = results_df[results_df['id'] == selected_id_for_details]
        if not matched_rows_in_results_df.empty:
            card_to_display_details = matched_rows_in_results_df.iloc[0]
            logger.info(f"DETAIL_VIEW: Carta encontrada en results_df para ID: {selected_id_for_details}. Nombre: {card_to_display_details.get('pokemon_name')}")
        else:
            logger.warning(f"DETAIL_VIEW: ID {selected_id_for_details} no encontrado en results_df actual (quizás results_df cambió).")
            
    if card_to_display_details is None and not results_df.empty:
        card_to_display_details = results_df.iloc[0] # Fallback al primero
        fallback_id = card_to_display_details.get('id')
        logger.info(f"DETAIL_VIEW: Fallback a la primera carta de results_df. ID: {fallback_id}. Nombre: {card_to_display_details.get('pokemon_name')}")
        # Sincronizar session_state si estamos mostrando el fallback
        if selected_id_for_details is None or (fallback_id and selected_id_for_details != fallback_id):
            if fallback_id and pd.notna(fallback_id):
                logger.info(f"DETAIL_VIEW: Actualizando session_state con ID de fallback: {fallback_id}")
                st.session_state.selected_card_id_from_grid = fallback_id
                # Podríamos necesitar un rerun aquí si queremos que el estado se refleje consistentemente
                # si el usuario luego interactúa con filtros y este fallback se activa.
                # Por ahora, se asume que el próximo rerun (por filtro o clic) lo corregirá.

    if card_to_display_details is not None and isinstance(card_to_display_details, pd.Series) and not card_to_display_details.empty:
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
            # ... (resto de los st.markdown y st.metric)
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
        logger.info("DETAIL_VIEW: No hay carta para mostrar en detalles (card_to_display_details es None o vacío).")
        st.info("Haz clic en una carta en la tabla de resultados para ver sus detalles o aplica filtros para ver cartas.")

elif not results_df.empty and results_df_for_aggrid_display.empty : 
    logger.info(f"MAIN_APP: results_df tiene {len(results_df)} filas, pero display está limitado/vacío.")
    st.info(f"Se encontraron {len(results_df)} resultados. Aplica filtros más específicos para visualizarlos.")
else: 
    logger.info("MAIN_APP: results_df está vacío.")
    if bq_client and LATEST_SNAPSHOT_TABLE:
        st.info("No se encontraron cartas con los filtros seleccionados.")

st.sidebar.info("Pokémon TCG Explorer - Debug Logs")
