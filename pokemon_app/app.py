logger.error(f"CONNECT_BQ: Faltan claves: {python
import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service', '.join(missing_keys)}")
             st.error(f"Error: Faltan claves: {_account
import re
import logging
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdate', '.join(missing_keys)}")
             return None
        creds = service_account.Credentials.fromMode, DataReturnMode

# --- Configuración Inicial ---
st.set_page_config(layout="wide_service_account_info(creds_json)
        client = bigquery.Client(credentials=cre", page_title="Pokémon TCG Explorer")
logging.basicConfig(
    level=logging.INFO,
ds, project=GCP_PROJECT_ID)
        logger.info("CONNECT_BQ: Conexión OK.")    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger
        return client
    except Exception as e:
        logger.error(f"CONNECT_BQ: Error = logging.getLogger(__name__)

# --- Constantes y Configuración de GCP ---
try:
    G: {e}", exc_info=True)
        st.error(f"Error al conectar con BigQueryCP_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
except KeyError: {e}.")
        return None

bq_client = connect_to_bigquery()
if bq:
    logger.critical("CRITICAL: 'project_id' no encontrado en los secrets.")
    st_client is None: st.stop()

# --- Funciones Auxiliares ---
@st.cache_.error("Error: 'project_id' no encontrado en los secrets.")
    st.stop()
exceptdata(ttl=3600)
def get_latest_snapshot_table(_client: bigquery. Exception as e:
    logger.error(f"CRITICAL: Error inesperado al leer secrets: {eClient) -> str | None:
    # ... (código igual) ...
    query = f"SELECT}")
    st.error(f"Error inesperado al leer secrets: {e}")
    st.stop table_id FROM `{_client.project}.{BIGQUERY_DATASET}`.__TABLES__ WHERE STARTS()

BIGQUERY_DATASET = "pokemon_dataset"
CARD_METADATA_TABLE = f"{_WITH(table_id, 'monthly_') ORDER BY table_id DESC LIMIT 1"
    tryGCP_PROJECT_ID}.{BIGQUERY_DATASET}.card_metadata"
MAX_ROWS_NO:
        results = _client.query(query).result()
        if results.total_rows > _FILTER = 200

# --- Conexión Segura a BigQuery ---
@st.cache0:
            latest_table_id = list(results)[0].table_id
            logger.info_resource
def connect_to_bigquery():
    try:
        if "gcp_service_account" not in st.secrets:
            logger.error("CONNECT_BQ: Sección [gcp_service_account](f"SNAPSHOT_TABLE: Usando: {latest_table_id}")
            return f"{_client no encontrada.")
            st.error("Error: Sección [gcp_service_account] no encontrada.")
.project}.{BIGQUERY_DATASET}.{latest_table_id}"
        logger.warning(f"SNAPSHOT_TABLE: No se encontraron tablas 'monthly_...'.")
        st.warning(f"No            return None
        creds_json = dict(st.secrets["gcp_service_account"])
        required_keys = ["type", "project_id", "private_key_id", "private_key se encontraron tablas 'monthly_...'.")
        return None
    except Exception as e:
        logger.", "client_email", "client_id"]
        missing_keys = [key for key in required_error(f"SNAPSHOT_TABLE: Error: {e}", exc_info=True)
        st.errorkeys if key not in creds_json or not creds_json[key]]
        if missing_keys(f"Error buscando tabla snapshot: {e}.")
        return None

POKEMON_SUFFIXES_TO_REMOVE = [' VMAX', ' VSTAR', ' V-UNION', ' V', ' GX', ' EX:
             logger.error(f"CONNECT_BQ: Faltan claves: {', '.join(missing_keys)}")', ' BREAK', ' Prism Star', ' Star', ' Radiant', ' δ', ' Tag Team', ' & ',
             st.error(f"Error: Faltan claves: {', '.join(missing_keys)}")
             return None
        creds = service_account.Credentials.from_service_account_info(cre ' Light', ' Dark', ' ◇', ' ☆']
MULTI_WORD_BASE_NAMES = ["Mr.ds_json)
        client = bigquery.Client(credentials=creds, project=GCP_PROJECT Mime", "Mime Jr.", "Farfetch'd", "Sirfetch'd", "Ho-Oh_ID)
        logger.info("CONNECT_BQ: Conexión OK.")
        return client
    ", "Porygon-Z", "Type: Null", "Tapu Koko", "Tapu Leleexcept Exception as e:
        logger.error(f"CONNECT_BQ: Error: {e}", exc_info=", "Tapu Bulu", "Tapu Fini", "Mr. Rime", "Indeedee M",True)
        st.error(f"Error al conectar con BigQuery: {e}.")
        return None "Indeedee F", "Great Tusk", "Iron Treads"]

def get_true_base_name(name_str, supertype, suffixes, multi_word_bases):
    # ... (código igual) ...


bq_client = connect_to_bigquery()
if bq_client is None: st.stop()

# --- Funciones Auxiliares ---
@st.cache_data(ttl=3600)
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    query    if not isinstance(name_str, str) or supertype != 'Pokémon': return name_str
    for mw_ = f"SELECT table_id FROM `{_client.project}.{BIGQUERY_DATASET}`.__TABLESbase in multi_word_bases:
        if name_str.startswith(mw_base): return mw___ WHERE STARTS_WITH(table_id, 'monthly_') ORDER BY table_id DESC LIMIT 1base 
    cleaned_name = name_str
    for suffix in suffixes:
        if cleaned_name.endswith(suffix): cleaned_name = cleaned_name[:-len(suffix)].strip()
    return cleaned_"
    try:
        results = _client.query(query).result()
        if results.totalname if cleaned_name else name_str

@st.cache_data(ttl=3600)_rows > 0:
            latest_table_id = list(results)[0].table_id
            logger.info(f"SNAPSHOT_TABLE: Usando: {latest_table_id}")
            return
def get_card_metadata_with_base_names(_client: bigquery.Client) -> pd.DataFrame:
    # ... (código igual) ...
    query = f"SELECT id, name, super f"{_client.project}.{BIGQUERY_DATASET}.{latest_table_id}"
        loggertype, subtypes, rarity, set_id, set_name, artist, images_large FROM `{CARD_MET.warning(f"SNAPSHOT_TABLE: No se encontraron tablas 'monthly_...'.")
        st.warning(f"No se encontraron tablas 'monthly_...'.")
        return None
    except Exception as e:ADATA_TABLE}`"
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            logger.warning("METADATA: DataFrame vacío de BQ.")
            st.warning("No se pudo cargar metadatos.")
            return pd.DataFrame()
        df['base_
        logger.error(f"SNAPSHOT_TABLE: Error: {e}", exc_info=True)
        st.error(f"Error buscando tabla snapshot: {e}.")
        return None

POKEMON_pokemon_name'] = df.apply(lambda row: get_true_base_name(row['name'],SUFFIXES_TO_REMOVE = [' VMAX', ' VSTAR', ' V-UNION', ' V', ' GX', ' EX', ' BREAK', ' Prism Star', ' Star', ' Radiant', ' δ', ' Tag Team', ' & ', ' Light', ' Dark', ' ◇', ' ☆']
MULTI_WORD_BASE_NAMES row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES), axis=1)
        logger.info(f"METADATA: Cargados {len(df)} metadatos.")
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("METADATA: Error 'db-dtypes'.", exc_info=True); = ["Mr. Mime", "Mime Jr.", "Farfetch'd", "Sirfetch'd", "Ho-Oh", "Porygon-Z", "Type: Null", "Tapu Koko", "Tapu Lele", "Tapu Bulu", "Tapu Fini", "Mr. Rime", " st.error("Error: Falta 'db-dtypes'.")
        else: logger.error(f"METADATA: Error: {e}", exc_info=True); st.error(f"Error metadatos: {e}.")
        return pd.DataFrame()

logger.info("APP_INIT: Cargando datos inicialesIndeedee M", "Indeedee F", "Great Tusk", "Iron Treads"]

def get_true_base_name(name_str, supertype, suffixes, multi_word_bases):
    if not.")
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
all_card_metadata isinstance(name_str, str) or supertype != 'Pokémon': return name_str
    for mw_base in multi_word_bases:
        if name_str.startswith(mw_base): return mw__df = get_card_metadata_with_base_names(bq_client)
if not LATESTbase 
    cleaned_name = name_str
    for suffix in suffixes:
        if cleaned_name.endswith(suffix): cleaned_name = cleaned_name[:-len(suffix)].strip()
    return cleaned__SNAPSHOT_TABLE or all_card_metadata_df.empty:
    logger.critical("APP_INIT: Datos esenciales no cargados. Stop.")
    st.error("Datos esenciales no cargados. Stop.")
    st.name if cleaned_name else name_str

@st.cache_data(ttl=3600)
def get_card_metadata_with_base_names(_client: bigquery.Client) -> pd.stop()
logger.info("APP_INIT: Datos iniciales OK.")

st.title("Explorador de Cartas Pokémon TCG")
st.sidebar.header("Filtros y Opciones")
options_df_forDataFrame:
    query = f"SELECT id, name, supertype, subtypes, rarity, set_id,_filters = all_card_metadata_df.copy()
# ... (Sidebar filters igual que antes) ...
supertype_options_list = sorted(options_df_for_filters['supertype'].dropna().unique().tolist()) set_name, artist, images_large FROM `{CARD_METADATA_TABLE}`"
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            
select_supertype_options = ["Todos"] + supertype_options_list if supertype_options_list elselogger.warning("METADATA: DataFrame vacío de BQ.")
            st.warning("No se pudo cargar metadatos.")
            return pd.DataFrame()
        df['base_pokemon_name'] = df. ["Todos"]
selected_supertype = st.sidebar.selectbox("Categoría:", select_supertype_options, index=0, key="sb_supertype_filter_v2")
if selected_supertype != "Todosapply(lambda row: get_true_base_name(row['name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES), axis=1)
        logger.info(f"METADATA: Cargados {len(df)} metadatos.")
        return": options_df_for_filters = options_df_for_filters[options_df_for_filters['supertype'] == selected_supertype]
set_options_list = sorted(options_df_for df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error_filters['set_name'].dropna().unique().tolist())
selected_sets = st.sidebar.multiselect("Set("METADATA: Error 'db-dtypes'.", exc_info=True); st.error("Error: Falta 'db-dtypes'.")
        else: logger.error(f"METADATA: Error: {e(s):", set_options_list, key="ms_sets_filter_v2")
if selected_sets: options_df_for_filters = options_df_for_filters[options_df_for}", exc_info=True); st.error(f"Error metadatos: {e}.")
        return_filters['set_name'].isin(selected_sets)]
name_label = "Nombre de Carta:"; name_col_ pd.DataFrame()

logger.info("APP_INIT: Cargando datos iniciales.")
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
all_card_metadata_df = getfor_options = 'name'
if selected_supertype == 'Pokémon': name_col_for_options = 'base_pokemon_name'; name_label = "Pokémon (Nombre Base):"
elif selected_super_card_metadata_with_base_names(bq_client)
if not LATEST_SNAPSHOT_TABLEtype != "Todos": name_label = f"Nombre ({selected_supertype}):"
if name_col_for_options in options_df_for_filters.columns: name_options_list = sorted(options or all_card_metadata_df.empty:
    logger.critical("APP_INIT: Datos esenciales no cargados._df_for_filters[name_col_for_options].dropna().unique().tolist())
else: Stop.")
    st.error("Datos esenciales no cargados. Stop.")
    st.stop()
logger name_options_list = []; logger.warning(f"SIDEBAR: Columna '{name_col_for_options}'.info("APP_INIT: Datos iniciales OK.")

st.title("Explorador de Cartas Pokémon TCG")
st.sidebar.header("Filtros y Opciones")
options_df_for_filters = all_card_metadata_df.copy()

supertype_options_list = sorted(options_ no encontrada.")
selected_names_to_filter = st.sidebar.multiselect(name_label, name_options_list, key="ms_names_filter_v2")
if selected_names_todf_for_filters['supertype'].dropna().unique().tolist())
select_supertype_options = ["_filter and name_col_for_options in options_df_for_filters.columns: options_dfTodos"] + supertype_options_list if supertype_options_list else ["Todos"]
selected_super_for_filters = options_df_for_filters[options_df_for_filters[name_col_for_options].isin(selected_names_to_filter)]
rarity_options_list = sortedtype = st.sidebar.selectbox("Categoría:", select_supertype_options, index=0, key="sb_super(options_df_for_filters['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Rareza(s):", rarity_options_list, key="ms_type_filter_v3") 
if selected_supertype != "Todos": options_df_for_rarities_filter_v2")
sort_order = st.sidebar.radio("Ordenar por Precio (Trendfilters = options_df_for_filters[options_df_for_filters['supertype'] == selected_supertype]
set_options_list = sorted(options_df_for_filters['set_name'].):", ("Ascendente", "Descendente"), index=1, key="rd_sort_order_v2dropna().unique().tolist())
selected_sets = st.sidebar.multiselect("Set(s):",")
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"


@st.cache_data(ttl=600)
def fetch_card_data(_client: bigquery.Client, latest set_options_list, key="ms_sets_filter_v3") 
if selected_sets: options__table_path: str, supertype_ui_filter: str | None,
                    sets_ui_df_for_filters = options_df_for_filters[options_df_for_filters['set_filter: list, names_ui_filter: list, rarities_ui_filter: list,
                    sortname'].isin(selected_sets)]
name_label = "Nombre de Carta:"; name_col_for__direction: str, full_metadata_df: pd.DataFrame) -> pd.DataFrame:
    # ...options = 'name'
if selected_supertype == 'Pokémon': name_col_for_options = ' (fetch_card_data igual que antes) ...
    logger.info(f"FETCH_DATA: Ini. SType:{supertype_ui_filter}, Sets:{len(sets_ui_filter)}, Names:{lenbase_pokemon_name'; name_label = "Pokémon (Nombre Base):"
elif selected_supertype !=(names_ui_filter)}, Rars:{len(rarities_ui_filter)}")
    ids_to "Todos": name_label = f"Nombre ({selected_supertype}):"
if name_col_for_query_df = full_metadata_df.copy()
    if supertype_ui_filter and super_options in options_df_for_filters.columns: name_options_list = sorted(options_df_for_filters[name_col_for_options].dropna().unique().tolist())
else: name_options_list = []; logger.warning(f"SIDEBAR: Columna '{name_col_for_optionstype_ui_filter != "Todos": ids_to_query_df = ids_to_query_df}' no encontrada.")
selected_names_to_filter = st.sidebar.multiselect(name_label, name_options_list, key="ms_names_filter_v3") 
if selected_names_to_[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter: ids_to_query_df = ids_to_query_df[ids_tofilter and name_col_for_options in options_df_for_filters.columns: options_df__query_df['set_name'].isin(sets_ui_filter)]
    if rarities_uifor_filters = options_df_for_filters[options_df_for_filters[name_col__filter: ids_to_query_df = ids_to_query_df[ids_to_queryfor_options].isin(selected_names_to_filter)]
rarity_options_list = sorted(options_df_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:_for_filters['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Rareza(s):", rarity_options_list, key="ms_rarities_filter
        actual_name_col_to_filter = 'base_pokemon_name' if supertype_ui_filter == 'Pokémon' else 'name'
        if actual_name_col_to_filter in ids_to_query_df.columns: ids_to_query_df = ids_to_query_df_v3") 
sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1, key="rd_sort_order_v3") 
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"

@st.[ids_to_query_df[actual_name_col_to_filter].isin(names_ui_filter)]
        else: logger.warning(f"FETCH_DATA: Col '{actual_name_col_to_filter}' no existe.")
    if ids_to_query_df.empty: logger.info("FETCHcache_data(ttl=600)
def fetch_card_data(_client: bigquery.Client, latest_table_path: str, supertype_ui_filter: str | None,
                    sets__DATA: No IDs post-filter."); return pd.DataFrame()
    list_of_card_ids = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids: logger.info("FETCH_DATA: Lista IDs vacía."); return pd.DataFrame()
    query_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
                    sort_direction: str, full_metadata_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"FETCH_DATA: Ini. SType:{supertype_ui_filter}, Sets:{len(setssql = f"""SELECT meta.id, meta.name AS pokemon_name, meta.supertype, meta.set_name, meta.rarity, meta.artist, meta.images_large AS image_url, prices.cm_trendPrice AS price FROM `{CARD_METADATA_TABLE}` AS meta JOIN `{latest_table__ui_filter)}, Names:{len(names_ui_filter)}, Rars:{len(rarities_ui_filter)}")
    ids_to_query_df = full_metadata_df.copy()
    if supertype_ui_filter and supertype_ui_filter != "Todos": ids_to_query_dfpath}` AS prices ON meta.id = prices.id WHERE meta.id IN UNNEST(@card_ids_param) ORDER BY prices.cm_trendPrice {sort_direction}"""
    query_params = [big = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filterquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids)]
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    logger.info(f"FETCH_DATA: SQL BQ para {len(list_of_card_ids)} IDs)]
    if rarities_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:
        actual_name_col_to_filter = 'base_pokemon. Orden: {sort_direction}")
    try:
        results_df = _client.query(query_name' if supertype_ui_filter == 'Pokémon' else 'name'
        if actual_name_col_to_filter in ids_to_query_df.columns: ids_to_query_df_sql, job_config=job_config).to_dataframe()
        results_df['price'] = pd.to_numeric(results_df['price'], errors='coerce')
        logger.info(f" = ids_to_query_df[ids_to_query_df[actual_name_col_toFETCH_DATA: BQ OK. Filas: {len(results_df)}.")
        return results_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger_filter].isin(names_ui_filter)]
        else: logger.warning(f"FETCH_DATA: Col '{actual_name_col_to_filter}' no existe.")
    if ids_to_query.error("FETCH_DATA: Err 'db-dtypes'.", exc_info=True); st.error("Error: Falta 'db-dtypes'.")
        else: logger.error(f"FETCH_DATA:_df.empty: logger.info("FETCH_DATA: No IDs post-filter."); return pd.DataFrame()
    list_of_card_ids = ids_to_query_df['id'].unique().tolist()
     Err BQ: {e}", exc_info=True); st.error(f"Error BQ: {e}.")
        return pd.DataFrame()


results_df = fetch_card_data(bq_client, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets, selected_names_to_filter, selected_if not list_of_card_ids: logger.info("FETCH_DATA: Lista IDs vacía."); returnrarities, sort_sql, all_card_metadata_df)
logger.info(f"MAIN_APP: results_df cargado con {len(results_df)} filas.")

# --- Área Principal: Visualización de Resultados pd.DataFrame()
    query_sql = f"""SELECT meta.id, meta.name AS pokemon_name, meta.supertype, meta.set_name, meta.rarity, meta.artist, meta.images_large AS image_url, prices.cm_trendPrice AS price FROM `{CARD_METADATA_TABLE}` AS meta JOIN ---
st.header("Resultados")

if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
    logger.info("SESSION_STATE_INIT: 'selected_card_id_from_grid' `{latest_table_path}` AS prices ON meta.id = prices.id WHERE meta.id IN UNNEST(@card_ids_param) ORDER BY prices.cm_trendPrice {sort_direction}"""
    query_params = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card inicializado a None.")

logger.info(f"AGGRID_RENDERING: ID en session_state ANTES de renderizar AgGrid: {st.session_state.get('selected_card_id_from_grid')}")

results_df_for_aggrid_display = results_df 
is_initial__ids)]
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    logger.info(f"FETCH_DATA: SQL BQ para {len(list_of_card_ids)} IDs. Orden: {sort_direction}")
    try:
        results_df = _clientunfiltered_load = (not selected_sets and not selected_names_to_filter and not selected_rarities and (selected_supertype == "Todos" or not selected_supertype))
if is_initial_unfiltered_load and len(results_df) > MAX_ROWS_NO_FILTER:
    logger.info(f"AGGRID_RENDERING: Limitando display a {MAX_ROWS_NO_FILTER} filas.query(query_sql, job_config=job_config).to_dataframe()
        results_df['price'] = pd.to_numeric(results_df['price'], errors='coerce')
        logger.info(f"FETCH_DATA: BQ OK. Filas: {len(results_df)}.")
        return results_df
    except Exception as e:
        if "db-dtypes" in str(e de {len(results_df)}.")
    st.info(f"Mostrando los primeros {MAX_ROWS_NO_FILTER} de {len(results_df)} resultados. Aplica filtros.")
    results_df_for_aggrid_display = results_df.head(MAX_ROWS_NO_FILTER)

).lower(): logger.error("FETCH_DATA: Err 'db-dtypes'.", exc_info=True); st.error("Error: Falta 'db-dtypes'.")
        else: logger.error(f"grid_response = None 
if not results_df_for_aggrid_display.empty:
    display_columns_mapping = {'id': 'ID', 'pokemon_name': 'Nombre Carta', 'supertype': 'Categoría', 'set_name': 'Set', 'rarity': 'Rareza', 'artist': 'Artista', 'price': 'Precio (Trend €)'}
    cols_in_df = [col forFETCH_DATA: Err BQ: {e}", exc_info=True); st.error(f"Error BQ: {e}.")
        return pd.DataFrame()

results_df = fetch_card_data(bq_client col in display_columns_mapping.keys() if col in results_df_for_aggrid_display.columns]
    final_display_df = results_df_for_aggrid_display[cols_in, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets, selected_names_to_filter, selected__df].copy()
    final_display_df.rename(columns=display_columns_mapping, inplace=True)
    price_display_col_name = display_columns_mapping.get('price')
rarities, sort_sql, all_card_metadata_df)
logger.info(f"MAIN_APP: results_df cargado con {len(results_df)} filas.")

# --- Área Principal: Visualización de Resultados    if price_display_col_name and price_display_col_name in final_display_df.columns:
         final_display_df[price_display_col_name] = final_display_df[price_ ---
st.header("Resultados")

if 'selected_card_id_from_grid' not indisplay_col_name].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")

    gb = GridOptionsBuilder.from_dataframe(final_ st.session_state:
    st.session_state.selected_card_id_from_grid = None
    logger.info("SESSION_STATE_INIT: 'selected_card_id_from_grid'display_df)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_grid_options(domLayout='normal')
    gb.configure_pagination(enabled inicializado a None.")

logger.info(f"AGGRID_RENDERING: ID en session_state ANTES de renderizar AgGrid: {st.session_state.get('selected_card_id_from_=True, paginationAutoPageSize=False, paginationPageSize=25)
    gridOptions = gb.build()

    st.write("Haz clic en una fila de la tabla para ver sus detalles:")
    gridgrid')}")

results_df_for_aggrid_display = results_df 
is_initial__response = AgGrid(
        final_display_df, gridOptions=gridOptions, height=50unfiltered_load = (not selected_sets and not selected_names_to_filter and not selected_rarities and (selected_supertype == "Todos" or not selected_supertype))
if is_initial_0, width='100%',
        data_return_mode=DataReturnMode.AS_INPUT, 
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=False, allow_unsafe_jscode=True, 
        key='pokemon_aggrid_staticunfiltered_load and len(results_df) > MAX_ROWS_NO_FILTER:
    logger.info(f"AGGRID_RENDERING: Limitando display a {MAX_ROWS_NO_FILTER} filas de {len(results_df)}.")
    st.info(f"Mostrando los primeros {MAX__key_v2', # KEY ESTÁTICA v2
    )
else:
    logger.info("AGGRID_RENDERING: No hay datos para mostrar en AgGrid.")

# --- LÓGICA DE MANROWS_NO_FILTER} de {len(results_df)} resultados. Aplica filtros.")
    results_df_for_aggrid_display = results_df.head(MAX_ROWS_NO_FILTER)

EJO DE CLIC EN AGGRID Y ACTUALIZACIÓN DE ESTADO (CORREGIDA PARA DATAFRAME) ---
if grid_response: 
    logger.info(f"AGGRID_HANDLER_ENTRY: Verificando grid_response.")
    
    newly_selected_id_from_grid = None
grid_response = None 
if not results_df_for_aggrid_display.empty:
    display_columns_mapping = {'id': 'ID', 'pokemon_name': 'Nombre Carta', 'supertype    selected_rows_data = grid_response.get('selected_rows') 
    logger.info(f"AGGRID_HANDLER_ENTRY: Tipo de grid_response.selected_rows: {type(selected_rows_data)}")': 'Categoría', 'set_name': 'Set', 'rarity': 'Rareza', 'artist': 'Artista', 'price': 'Precio (Trend €)'}
    cols_in_df = [col for
    
    if isinstance(selected_rows_data, pd.DataFrame) and not selected_rows_data.empty:
        try: 
            first_row = selected_rows_data.iloc[0] # Acceder a la primera col in display_columns_mapping.keys() if col in results_df_for_aggrid_display.columns]
    final_display_df = results_df_for_aggrid_display[cols_in_df].copy()
    final_display_df.rename(columns=display_columns_mapping, inplace fila (Serie)
            # Verificar si 'ID' existe en el índice o columnas de la Serie
            if '=True)
    price_display_col_name = display_columns_mapping.get('price')
    if price_display_col_name and price_display_col_name in final_display_df.columns:
ID' in first_row:
                newly_selected_id_from_grid = first_row['         final_display_df[price_display_col_name] = final_display_df[price_ID'] 
            else:
                logger.warning("AGGRID_HANDLER: Columna 'ID' no encontrada en ladisplay_col_name].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")

    gb = GridOptionsBuilder.from_dataframe(final_ fila seleccionada del DataFrame.")

            if newly_selected_id_from_grid:
                 logger.info(f"display_df)
    gb.configure_selection(selection_mode='single', use_checkbox=False)AGGRID_HANDLER: Fila seleccionada válida (DataFrame). ID: {newly_selected_id_from_grid}")
            else:
                 logger.warning("AGGRID_HANDLER: 'ID' es None o vacío después de intentar extraerlo.")
                 newly_selected_id_from_grid = None 
                 
        except IndexError
    gb.configure_grid_options(domLayout='normal')
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
    gridOptions = gb.build()

    st.write("Haz clic en una fila de la tabla para ver sus detalles:")
    #:
            logger.warning("AGGRID_HANDLER: Error de índice al acceder a .iloc[0] (DataFrame vacío inesperadamente?).")
            newly_selected_id_from_grid = None
        except Exception as e:
 Volver a KEY ESTÁTICA para simplificar
    grid_response = AgGrid(
        final_display_df, gridOptions=gridOptions, height=500, width='100%',
        data_return_mode=DataReturnMode.AS_INPUT, 
        update_mode=GridUpdateMode.SELECTION            logger.error(f"AGGRID_HANDLER: Error inesperado procesando fila (DataFrame): {e}", exc__CHANGED,
        fit_columns_on_grid_load=False, allow_unsafe_jscode=True, 
        key='pokemon_aggrid_static_key_v2', 
    )
info=True)
            newly_selected_id_from_grid = None 
    else:
        logger.debug(f"AGGRID_HANDLER: No hay filas seleccionadas válidas (selected_rows no es DataFrameelse:
    logger.info("AGGRID_RENDERING: No hay datos para mostrar en AgGrid.")

# --- LÓGICA DE MANEJO DE CLIC EN AGGRID Y ACTUALIZACIÓN DE ESTADO (Adaptada a DataFrame) ---
if grid_response: 
    logger.info(f"AGGRID_HANDLER_ENTRY o está vacío).")
    
    current_session_id = st.session_state.get('selected_card_id_from_grid')
    logger.info(f"AGGRID_HANDLER: Current session ID: {current_session_id}, Newly selected from grid: {newly_selected_id_from_grid}"): Verificando grid_response. Tipo de selected_rows: {type(grid_response.get('selected_rows'))}")
    
    newly_selected_id_from_grid = None
    selected_rows_data = grid_response.get('selected_rows') 
    
    # --- LÓGICA AD
    
    if newly_selected_id_from_grid is not None and newly_selected_id_from_grid != current_session_id:
        logger.info(f"AGGRID_HANDLER: DETECTADO CAMBIO DE SELECCIÓN! De '{current_session_id}' a '{newly_selected_id_APTADA PARA DataFrame o Lista ---
    if isinstance(selected_rows_data, pd.DataFrame) and not selected_rows_data.empty:
        try:
            first_selected_row_series = selected_rows_datafrom_grid}'. Actualizando session_state y RE-EJECUTANDO.")
        st.session_state..iloc[0] 
            newly_selected_id_from_grid = first_selected_row_series.get('ID') 
            if newly_selected_id_from_grid:
                 logger.infoselected_card_id_from_grid = newly_selected_id_from_grid
        st.experimental_rerun() 
    else:
        logger.debug(f"AGGRID_HANDLER: Sin cambio de selección o nueva selección es None. No se re-ejecuta por esta vía.")
# --- FIN DE LÓGICA DE MANEJO DE CLIC ---

st.divider()
st.header("Detalle de Carta")

(f"AGGRID_HANDLER (DataFrame): Fila seleccionada válida. ID: {newly_selected_id_from_grid}")
            else:
                 logger.warning("AGGRID_HANDLER (DataFrame): 'ID' no encontrado en la fila seleccionada o es None.")
        except IndexError:
            logger.warning("AGcard_to_display_details = None 
id_from_session = st.session_state.get('selected_card_id_from_grid')
logger.info(f"DETAIL_DISPLAY_ENTRY: ID para detalles (desde session_state): {id_from_session}")

if id_from_session:
GRID_HANDLER (DataFrame): selected_rows es DataFrame pero vacío.")
        except Exception as e:
            logger.error    if not results_df.empty:
        matched_rows_in_results_df = results_df[results_df['id'] == id_from_session]
        if not matched_rows_in_results_df(f"AGGRID_HANDLER (DataFrame): Error inesperado procesando fila: {e}", exc_info=True)
            newly_selected_id_from_grid = None 
    elif isinstance(selected_rows_data, list) and selected_rows_data: # Fallback por si devuelve lista
        try: .empty:
            card_to_display_details = matched_rows_in_results_df.iloc[0]
            logger.info(f"DETAIL_DISPLAY: Carta encontrada para ID '{id_from_session}'.
            row_data = selected_rows_data[0]
            if isinstance(row_data, dict Nombre: {card_to_display_details.get('pokemon_name')}")
        else:
            logger.warning(f"DETAIL_DISPLAY: ID '{id_from_session}' NO ENCONTRADO en results_df):
                newly_selected_id_from_grid = row_data.get('ID') 
                if newly_selected_id_from_grid:
                     logger.info(f"AGGRID_HANDLER actual (total {len(results_df)} filas).")
            # Limpiar el estado si el ID ya no es válido en los resultados actuales?
            # st.session_state.selected_card_id_from_grid = None (List): Fila seleccionada válida. ID: {newly_selected_id_from_grid}")
                else:
                     logger.warning("AGGRID_HANDLER (List): 'ID' no encontrado o None.")
            else 
    else:
        logger.warning(f"DETAIL_DISPLAY: results_df está vacío, no:
                logger.warning(f"AGGRID_HANDLER (List): Fila no es dict: {type(row_data)}")
        except IndexError: 
            logger.warning("AGGRID_HANDLER (List): selected_rows vac se puede buscar ID '{id_from_session}'.")
        
if card_to_display_details is None and not results_df.empty:
    card_to_display_details = results_df.iloc[0]
    fallback_id = card_to_display_details.get('id')
    logger.info(f"DETAIL_DISPLAY: Usando FALLBACK a la primera carta de results_df. ID: {fallbackía.")
        except Exception as e: 
            logger.error(f"AGGRID_HANDLER (List): Error inesperado procesando fila: {e}", exc_info=True)
            newly_selected_id_id}. Nombre: {card_to_display_details.get('pokemon_name')}")
    if id_from_session is None or (fallback_id and id_from_session != fallback_id):
        _from_grid = None 
    else:
        logger.debug(f"AGGRID_HANDLER: No hay filas seleccionadas válidas (selected_rows no es DataFrame/lista no vacía o es None).")
if fallback_id and pd.notna(fallback_id):
            if st.session_state.get    # --- FIN LÓGICA ADAPTADA ---
    
    current_session_id = st.session_state.get('selected_card_id_from_grid')
    logger.info(f"AG('selected_card_id_from_grid') != fallback_id : 
                logger.info(f"DETAIL_DISPLAY: Actualizando session_state con ID de fallback '{fallback_id}' (era '{id_fromGRID_HANDLER: Current session ID: {current_session_id}, Newly selected from grid: {newly_selected_id_from_grid}")
    
    if newly_selected_id_from_grid is not None and_session}').")
                st.session_state.selected_card_id_from_grid = fallback_id

if card_to_display_details is not None and isinstance(card_to_display_details, pd.Series) and not card_to_display_details.empty:
    logger.info(f newly_selected_id_from_grid != current_session_id:
        logger.info(f"AGGRID_HANDLER: DETECTADO CAMBIO DE SELECCIÓN! De '{current_session_id}' a"DETAIL_DISPLAY: RENDERIZANDO detalles para: {card_to_display_details.get('pokemon_name')} (ID: {card_to_display_details.get('id')})")
    # ... (código para mostrar detalles igual que antes) ...
    card_name_detail = card_to_display_details '{newly_selected_id_from_grid}'. Actualizando session_state y RE-EJECUTANDO.")
        st.session_state.selected_card_id_from_grid = newly_selected_id_from_grid
        st.experimental_rerun() 
    else:
        logger.debug(f"AGGRID_HANDLER: Sin cambio de selección o nueva selección es None. No se re-ejecuta por esta vía.")
.get('pokemon_name', "N/A")
    card_id_detail = card_to_display_details.get('id', "N/A")
    card_set_detail = card_to_display_details.get('set_name', "N/A")
    card_image_url_detail = card_to_display_details.get('image_url', None)
    card_supertype# --- FIN DE LÓGICA DE MANEJO DE CLIC ---

st.divider()
st.header("Detalle de Carta")

card_to_display_details = None 
id_from_session_detail = card_to_display_details.get('supertype', "N/A")
    card_rarity_detail = card_to_display_details.get('rarity', "N/A") = st.session_state.get('selected_card_id_from_grid')
logger.info(f"DETAIL_DISPLAY_ENTRY: ID para detalles (de session_state): {id_from_session}")

if id
    card_artist_detail = card_to_display_details.get('artist', None)
    card_price_detail = card_to_display_details.get('price', None)
    col1,_from_session:
    if not results_df.empty: 
        matched_rows_in_results_df = results_df[results_df['id'] == id_from_session]
        if not matched_rows_in_results_df.empty:
            card_to_display_details = matched col2 = st.columns([1, 2])
    with col1:
        if pd.notna(card_image_url_detail): st.image(card_image_url_detail, caption=f"{card_name_detail} - {card_set_detail}", width=300)
        else: st_rows_in_results_df.iloc[0]
            logger.info(f"DETAIL_DISPLAY: Carta encontrada para ID '{id_from_session}'. Nombre: {card_to_display_details.get.warning("Imagen no disponible.")
    with col2:
        st.subheader(f"{card_name_detail}")
        st.markdown(f"**ID:** `{card_id_detail}`")
        ('pokemon_name')}")
        else:
            logger.warning(f"DETAIL_DISPLAY: ID '{id_from_session}' NO ENCONTRADO en results_df actual.")
            # No limpiar session_state aquí para evitar posibles bucles si el rerun es rápido
    else:
        logger.warning(f"DETAIL_st.markdown(f"**Categoría:** {card_supertype_detail}")
        st.markdown(f"**Set:** {card_set_detail}")
        st.markdown(f"**Rareza:** {card_rarity_detail}")
        if pd.notna(card_artist_detail) and card_artist_detail: st.markdown(f"**Artista:** {card_artist_detail}")
        if pd.notna(card_price_detail): st.metric(label="Precio (Trend €)", value=DISPLAY: results_df está vacío, no se puede buscar ID '{id_from_session}'.")
        
if card_to_display_details is None and not results_df.empty:
    card_to_display_details = results_df.iloc[0] 
    fallback_id = card_to_display_details.get('id')
    logger.info(f"DETAIL_DISPLAY: Usando FALLBACK a la primera carta. ID: {fallback_id}.")
    if id_from_session is None or (fallback_id andf"€{card_price_detail:.2f}")
        else: st.markdown("**Precio (Trend €):** N/A")
else: 
    logger.info("DETAIL_DISPLAY: No hay carta para mostrar en detalles al final.")
    st.info("Haz clic en una carta en la tabla de resultados para id_from_session != fallback_id):
        if fallback_id and pd.notna(fallback_id):
            if st.session_state.get('selected_card_id_from_grid') != fallback_id : 
                logger.info(f"DETAIL_DISPLAY: Actualizando session_state con ID de fallback '{fallback_id}' (era '{st.session_state.get('selected_card_id_from_grid ver sus detalles o aplica filtros.")

if not results_df_for_aggrid_display.empty: pass
elif not results_df.empty and results_df_for_aggrid_display.empty : 
    logger.info(f"DISPLAY_MSG: results_df tiene {len(results_df)} filas, pero display limitado/vacío.")
    st.info(f"Se encontraron {len(results_df)} resultados. Aplica filtros más específicos.")
else: 
    logger.info("DISPLAY_MSG: results_df está vacío.")')}').")
                st.session_state.selected_card_id_from_grid = fallback_id
                # No llamar rerun desde el fallback
            
if card_to_display_details is not None and isinstance(card_to_display_details, pd.Series) and not card_to_display_details.empty:

    if bq_client and LATEST_SNAPSHOT_TABLE:
        st.info("No se encontraron cartas con los filtros seleccionados.")

st.sidebar.info("Pokémon TCG Explorer - Debug v5")
