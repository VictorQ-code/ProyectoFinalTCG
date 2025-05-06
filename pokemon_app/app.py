import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re
import logging
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Pokémon TCG Explorer")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Constantes y Configuración de GCP ---
try:
    GCP_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
except KeyError:
    logger.critical("CRITICAL: 'project_id' no encontrado en los secrets de Streamlit.")
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
        if "gcp_service_account" not in st.secrets:
            logger.error("CONNECT_BQ: Sección [gcp_service_account] no encontrada.")
            st.error("Error: Sección [gcp_service_account] no encontrada.")
            return None
        creds_json = dict(st.secrets["gcp_service_account"])
        required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"]
        missing_keys = [key for key in required_keys if key not in creds_json or not creds_json[key]]
        if missing_keys:
             logger.error(f"CONNECT_BQ: Faltan claves: {', '.join(missing_keys)}")
             st.error(f"Error: Faltan claves: {', '.join(missing_keys)}")
             return None
        creds = service_account.Credentials.from_service_account_info(creds_json)
        client = bigquery.Client(credentials=creds, project=GCP_PROJECT_ID)
        logger.info("CONNECT_BQ: Conexión OK.")
        return client
    except Exception as e:
        logger.error(f"CONNECT_BQ: Error: {e}", exc_info=True)
        st.error(f"Error al conectar con BigQuery: {e}.")
        return None

bq_client = connect_to_bigquery()
if bq_client is None: st.stop()

# --- Funciones Auxiliares ---
@st.cache_data(ttl=3600)
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    query = f"SELECT table_id FROM `{_client.project}.{BIGQUERY_DATASET}`.__TABLES__ WHERE STARTS_WITH(table_id, 'monthly_') ORDER BY table_id DESC LIMIT 1"
    try:
        results = _client.query(query).result()
        if results.total_rows > 0:
            latest_table_id = list(results)[0].table_id
            logger.info(f"SNAPSHOT_TABLE: Usando: {latest_table_id}")
            return f"{_client.project}.{BIGQUERY_DATASET}.{latest_table_id}"
        logger.warning(f"SNAPSHOT_TABLE: No se encontraron tablas 'monthly_...'.")
        st.warning(f"No se encontraron tablas 'monthly_...'.")
        return None
    except Exception as e:
        logger.error(f"SNAPSHOT_TABLE: Error: {e}", exc_info=True)
        st.error(f"Error buscando tabla snapshot: {e}.")
        return None

POKEMON_SUFFIXES_TO_REMOVE = [' VMAX', ' VSTAR', ' V-UNION', ' V', ' GX', ' EX', ' BREAK', ' Prism Star', ' Star', ' Radiant', ' δ', ' Tag Team', ' & ', ' Light', ' Dark', ' ◇', ' ☆']
MULTI_WORD_BASE_NAMES = ["Mr. Mime", "Mime Jr.", "Farfetch'd", "Sirfetch'd", "Ho-Oh", "Porygon-Z", "Type: Null", "Tapu Koko", "Tapu Lele", "Tapu Bulu", "Tapu Fini", "Mr. Rime", "Indeedee M", "Indeedee F", "Great Tusk", "Iron Treads"]

def get_true_base_name(name_str, supertype, suffixes, multi_word_bases):
    if not isinstance(name_str, str) or supertype != 'Pokémon': return name_str
    for mw_base in multi_word_bases:
        if name_str.startswith(mw_base): return mw_base 
    cleaned_name = name_str
    for suffix in suffixes:
        if cleaned_name.endswith(suffix): cleaned_name = cleaned_name[:-len(suffix)].strip()
    return cleaned_name if cleaned_name else name_str

@st.cache_data(ttl=3600)
def get_card_metadata_with_base_names(_client: bigquery.Client) -> pd.DataFrame:
    query = f"SELECT id, name, supertype, subtypes, rarity, set_id, set_name, artist, images_large FROM `{CARD_METADATA_TABLE}`"
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            logger.warning("METADATA: DataFrame vacío de BQ.")
            st.warning("No se pudo cargar metadatos.")
            return pd.DataFrame()
        df['base_pokemon_name'] = df.apply(lambda row: get_true_base_name(row['name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES), axis=1)
        logger.info(f"METADATA: Cargados {len(df)} metadatos.")
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("METADATA: Error 'db-dtypes'.", exc_info=True); st.error("Error: Falta 'db-dtypes'.")
        else: logger.error(f"METADATA: Error: {e}", exc_info=True); st.error(f"Error metadatos: {e}.")
        return pd.DataFrame()

logger.info("APP_INIT: Cargando datos iniciales.")
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
all_card_metadata_df = get_card_metadata_with_base_names(bq_client)
if not LATEST_SNAPSHOT_TABLE or all_card_metadata_df.empty:
    logger.critical("APP_INIT: Datos esenciales no cargados. Stop.")
    st.error("Datos esenciales no cargados. Stop.")
    st.stop()
logger.info("APP_INIT: Datos iniciales OK.")

st.title("Explorador de Cartas Pokémon TCG")
st.sidebar.header("Filtros y Opciones")
options_df_for_filters = all_card_metadata_df.copy()

supertype_options_list = sorted(options_df_for_filters['supertype'].dropna().unique().tolist())
select_supertype_options = ["Todos"] + supertype_options_list if supertype_options_list else ["Todos"]
selected_supertype = st.sidebar.selectbox("Categoría:", select_supertype_options, index=0, key="sb_supertype_filter")
if selected_supertype != "Todos": options_df_for_filters = options_df_for_filters[options_df_for_filters['supertype'] == selected_supertype]

set_options_list = sorted(options_df_for_filters['set_name'].dropna().unique().tolist())
selected_sets = st.sidebar.multiselect("Set(s):", set_options_list, key="ms_sets_filter")
if selected_sets: options_df_for_filters = options_df_for_filters[options_df_for_filters['set_name'].isin(selected_sets)]

name_label = "Nombre de Carta:"; name_col_for_options = 'name'
if selected_supertype == 'Pokémon': name_col_for_options = 'base_pokemon_name'; name_label = "Pokémon (Nombre Base):"
elif selected_supertype != "Todos": name_label = f"Nombre ({selected_supertype}):"
if name_col_for_options in options_df_for_filters.columns: name_options_list = sorted(options_df_for_filters[name_col_for_options].dropna().unique().tolist())
else: name_options_list = []; logger.warning(f"SIDEBAR: Columna '{name_col_for_options}' no encontrada.")
selected_names_to_filter = st.sidebar.multiselect(name_label, name_options_list, key="ms_names_filter")
if selected_names_to_filter and name_col_for_options in options_df_for_filters.columns: options_df_for_filters = options_df_for_filters[options_df_for_filters[name_col_for_options].isin(selected_names_to_filter)]

rarity_options_list = sorted(options_df_for_filters['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Rareza(s):", rarity_options_list, key="ms_rarities_filter")

sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1, key="rd_sort_order")
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"

@st.cache_data(ttl=600)
def fetch_card_data(_client: bigquery.Client, latest_table_path: str, supertype_ui_filter: str | None,
                    sets_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
                    sort_direction: str, full_metadata_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"FETCH_DATA: Ini. SType:{supertype_ui_filter}, Sets:{len(sets_ui_filter)}, Names:{len(names_ui_filter)}, Rars:{len(rarities_ui_filter)}")
    ids_to_query_df = full_metadata_df.copy()
    if supertype_ui_filter and supertype_ui_filter != "Todos": ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]
    if rarities_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:
        actual_name_col_to_filter = 'base_pokemon_name' if supertype_ui_filter == 'Pokémon' else 'name'
        if actual_name_col_to_filter in ids_to_query_df.columns: ids_to_query_df = ids_to_query_df[ids_to_query_df[actual_name_col_to_filter].isin(names_ui_filter)]
        else: logger.warning(f"FETCH_DATA: Col '{actual_name_col_to_filter}' no existe.")
    if ids_to_query_df.empty: logger.info("FETCH_DATA: No IDs post-filter."); return pd.DataFrame()
    list_of_card_ids = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids: logger.info("FETCH_DATA: Lista IDs vacía."); return pd.DataFrame()
    query_sql = f"""SELECT meta.id, meta.name AS pokemon_name, meta.supertype, meta.set_name, meta.rarity, meta.artist, meta.images_large AS image_url, prices.cm_trendPrice AS price FROM `{CARD_METADATA_TABLE}` AS meta JOIN `{latest_table_path}` AS prices ON meta.id = prices.id WHERE meta.id IN UNNEST(@card_ids_param) ORDER BY prices.cm_trendPrice {sort_direction}"""
    query_params = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids)]
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    logger.info(f"FETCH_DATA: SQL BQ para {len(list_of_card_ids)} IDs. Orden: {sort_direction}")
    try:
        results_df = _client.query(query_sql, job_config=job_config).to_dataframe()
        results_df['price'] = pd.to_numeric(results_df['price'], errors='coerce')
        logger.info(f"FETCH_DATA: BQ OK. Filas: {len(results_df)}.")
        return results_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("FETCH_DATA: Err 'db-dtypes'.", exc_info=True); st.error("Error: Falta 'db-dtypes'.")
        else: logger.error(f"FETCH_DATA: Err BQ: {e}", exc_info=True); st.error(f"Error BQ: {e}.")
        return pd.DataFrame()

results_df = fetch_card_data(bq_client, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets, selected_names_to_filter, selected_rarities, sort_sql, all_card_metadata_df)
logger.info(f"MAIN_APP: results_df cargado con {len(results_df)} filas.")

# --- Área Principal: Visualización de Resultados ---
st.header("Resultados")

if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
    logger.info("SESSION_STATE_INIT: 'selected_card_id_from_grid' inicializado a None.")

logger.info(f"AGGRID_RENDERING: ID en session_state ANTES de renderizar AgGrid: {st.session_state.get('selected_card_id_from_grid')}")

results_df_for_aggrid_display = results_df 
is_initial_unfiltered_load = (not selected_sets and not selected_names_to_filter and not selected_rarities and (selected_supertype == "Todos" or not selected_supertype))
if is_initial_unfiltered_load and len(results_df) > MAX_ROWS_NO_FILTER:
    logger.info(f"AGGRID_RENDERING: Limitando display a {MAX_ROWS_NO_FILTER} filas de {len(results_df)}.")
    st.info(f"Mostrando los primeros {MAX_ROWS_NO_FILTER} de {len(results_df)} resultados. Aplica filtros.")
    results_df_for_aggrid_display = results_df.head(MAX_ROWS_NO_FILTER)

grid_response = None 
if not results_df_for_aggrid_display.empty:
    display_columns_mapping = {'id': 'ID', 'pokemon_name': 'Nombre Carta', 'supertype': 'Categoría', 'set_name': 'Set', 'rarity': 'Rareza', 'artist': 'Artista', 'price': 'Precio (Trend €)'}
    cols_in_df = [col for col in display_columns_mapping.keys() if col in results_df_for_aggrid_display.columns]
    final_display_df = results_df_for_aggrid_display[cols_in_df].copy()
    final_display_df.rename(columns=display_columns_mapping, inplace=True)
    price_display_col_name = display_columns_mapping.get('price')
    if price_display_col_name and price_display_col_name in final_display_df.columns:
         final_display_df[price_display_col_name] = final_display_df[price_display_col_name].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")

    gb = GridOptionsBuilder.from_dataframe(final_display_df)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_grid_options(domLayout='normal')
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
    gridOptions = gb.build()

    st.write("Haz clic en una fila de la tabla para ver sus detalles:")
    grid_response = AgGrid(
        final_display_df, gridOptions=gridOptions, height=500, width='100%',
        data_return_mode=DataReturnMode.AS_INPUT, 
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=False, allow_unsafe_jscode=True, 
        key=f'pokemon_aggrid_key_{st.session_state.get("selected_card_id_from_grid", "initial")}',
    )
else:
    logger.info("AGGRID_RENDERING: No hay datos para mostrar en AgGrid.")

# --- LÓGICA DE MANEJO DE CLIC EN AGGRID Y ACTUALIZACIÓN DE ESTADO (CORREGIDA) ---
if grid_response: 
    newly_selected_id_from_grid = None
    selected_rows_list = grid_response.get('selected_rows')
    
    if isinstance(selected_rows_list, list) and selected_rows_list: 
        try: 
            # 'ID' es el nombre de la columna en final_display_df (que es results_df['id'])
            row_data = selected_rows_list[0]
            if isinstance(row_data, dict): # Asegurarse que la fila es un diccionario
                newly_selected_id_from_grid = row_data.get('ID') 
                if newly_selected_id_from_grid:
                     logger.info(f"AGGRID_HANDLER: Fila seleccionada en AgGrid. ID: {newly_selected_id_from_grid}")
                else:
                     logger.warning("AGGRID_HANDLER: Fila seleccionada en AgGrid pero el 'ID' es None o no existe en la fila.")
            else:
                logger.warning(f"AGGRID_HANDLER: Fila seleccionada no es un diccionario: {type(row_data)}")
        except (IndexError) as e: # IndexError si selected_rows_list está vacía (aunque ya lo chequeamos)
            logger.warning(f"AGGRID_HANDLER: Error de índice al acceder a fila seleccionada de AgGrid: {e}")
        except Exception as e: # Captura general para otros errores inesperados
            logger.error(f"AGGRID_HANDLER: Error inesperado al procesar fila seleccionada: {e}", exc_info=True)
            newly_selected_id_from_grid = None 
    else:
        logger.debug(f"AGGRID_HANDLER: No hay filas seleccionadas válidas en AgGrid. selected_rows: {selected_rows_list}")
    
    current_session_id = st.session_state.get('selected_card_id_from_grid')
    
    if newly_selected_id_from_grid is not None and newly_selected_id_from_grid != current_session_id:
        logger.info(f"AGGRID_HANDLER: Nueva selección! Cambiando ID en sesión de '{current_session_id}' a '{newly_selected_id_from_grid}'. Re-ejecutando script.")
        st.session_state.selected_card_id_from_grid = newly_selected_id_from_grid
        st.experimental_rerun()
    else:
        logger.debug(f"AGGRID_HANDLER: Selección de AgGrid no cambió o es None. ID en sesión: {current_session_id}, ID de grid: {newly_selected_id_from_grid}")
# --- FIN DE LÓGICA DE MANEJO DE CLIC ---

st.divider()
st.header("Detalle de Carta")

card_to_display_details = None 
id_from_session = st.session_state.get('selected_card_id_from_grid')
logger.info(f"DETAIL_DISPLAY: Intentando mostrar detalles para ID (desde session_state): {id_from_session}")

if id_from_session:
    matched_rows_in_results_df = results_df[results_df['id'] == id_from_session]
    if not matched_rows_in_results_df.empty:
        card_to_display_details = matched_rows_in_results_df.iloc[0]
        logger.info(f"DETAIL_DISPLAY: Carta encontrada en results_df para ID: {id_from_session}. Nombre: {card_to_display_details.get('pokemon_name')}")
    else:
        logger.warning(f"DETAIL_DISPLAY: ID {id_from_session} de session_state NO encontrado en results_df actual (total {len(results_df)} filas). Los filtros pueden haber cambiado.")
        # No limpiar session_state aquí, podría ser un estado intermedio antes de que AgGrid se actualice
        # con los nuevos resultados. Si AgGrid se vacía, la selección persistirá pero no se encontrará.
        
if card_to_display_details is None and not results_df.empty:
    card_to_display_details = results_df.iloc[0]
    fallback_id = card_to_display_details.get('id')
    logger.info(f"DETAIL_DISPLAY: Fallback a la primera carta de results_df. ID: {fallback_id}. Nombre: {card_to_display_details.get('pokemon_name')}")
    if id_from_session is None or (fallback_id and id_from_session != fallback_id):
        if fallback_id and pd.notna(fallback_id):
            logger.info(f"DETAIL_DISPLAY: Actualizando session_state con ID de fallback: {fallback_id} (porque id_from_session era {id_from_session})")
            if st.session_state.get('selected_card_id_from_grid') != fallback_id : # Solo actualizar si es diferente para evitar bucle de rerun si no hay rerun explícito aquí
                st.session_state.selected_card_id_from_grid = fallback_id
                # Considerar un rerun aquí si es necesario para que este fallback se refleje inmediatamente
                # st.experimental_rerun()

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
    logger.info("DETAIL_DISPLAY: No hay carta para mostrar en detalles al final de la lógica.")
    st.info("Haz clic en una carta en la tabla de resultados para ver sus detalles o aplica filtros para ver cartas.")

if not results_df_for_aggrid_display.empty: pass
elif not results_df.empty and results_df_for_aggrid_display.empty : 
    logger.info(f"DISPLAY_MSG: results_df tiene {len(results_df)} filas, pero display está limitado/vacío.")
    st.info(f"Se encontraron {len(results_df)} resultados. Aplica filtros más específicos para visualizarlos.")
else: 
    logger.info("DISPLAY_MSG: results_df está vacío.")
    if bq_client and LATEST_SNAPSHOT_TABLE:
        st.info("No se encontraron cartas con los filtros seleccionados.")

st.sidebar.info("Pokémon TCG Explorer - Debug v3")
