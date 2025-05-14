import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re
import logging
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import numpy as np
import os
import tensorflow as tf # Mantenido por si cambias de modelo MLP a TF
import joblib # Para cargar los pipelines .pkl
import typing
import random
import json # Para cargar el threshold

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Pokémon TCG Explorer")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info(f"TensorFlow Version: {tf.__version__}")
logger.info(f"Keras Version (via TF): {tf.keras.__version__}")


# --- Constantes y Configuración de GCP ---
try:
    GCP_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
    logger.info(f"CONFIG: GCP Project ID '{GCP_PROJECT_ID}' cargado.")
except KeyError:
    logger.critical("CRITICAL_CONFIG: 'project_id' o [gcp_service_account] no encontrado en secrets.")
    st.error("Error Crítico: Configuración de 'project_id' no encontrada. Revisa Secrets.")
    st.stop()
except Exception as e:
    logger.critical(f"CRITICAL_CONFIG: Error inesperado leyendo secrets: {e}", exc_info=True)
    st.error(f"Error Crítico leyendo Secrets: {e}")
    st.stop()

BIGQUERY_DATASET = "pokemon_dataset"
CARD_METADATA_TABLE = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.card_metadata"
MAX_ROWS_NO_FILTER = 200

# --- RUTAS Y NOMBRES DE ARCHIVOS DEL MODELO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_files")

# LightGBM (Pipelines) - Modelo activo
LGBM_MODEL_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "lgbm_models")
PIPE_LOW_PKL_FILENAME = "modelo_pipe_low.pkl"
PIPE_HIGH_PKL_FILENAME = "modelo_pipe_high.pkl"
THRESHOLD_JSON_FILENAME = "threshold.json"
PIPE_LOW_LGBM_PATH = os.path.join(LGBM_MODEL_DIR, PIPE_LOW_PKL_FILENAME)
PIPE_HIGH_LGBM_PATH = os.path.join(LGBM_MODEL_DIR, PIPE_HIGH_PKL_FILENAME)
THRESHOLD_LGBM_PATH = os.path.join(LGBM_MODEL_DIR, THRESHOLD_JSON_FILENAME)


# --- CONFIGURACIÓN DE FEATURES PARA LGBM ---
_LGBM_NUMERIC_FEATURES_APP = [
    'prev_price', 'days_since_prev_snapshot',
    'cm_avg1', 'cm_avg7', 'cm_avg30', 'cm_trendPrice'
]
_LGBM_CATEGORICAL_FEATURES_APP = [
    'artist_name', 'pokemon_name', 'rarity',
    'set_name', 'types', 'supertype', 'subtypes'
]
_LGBM_ALL_FEATURES_APP = _LGBM_NUMERIC_FEATURES_APP + _LGBM_CATEGORICAL_FEATURES_APP
_LGBM_THRESHOLD_COLUMN_APP = 'cm_avg7'

# --- CONFIGURACIÓN PARA CARTAS DESTACADAS ---
FEATURED_RARITY = 'Special Illustration Rare'
NUM_FEATURED_CARDS_TO_DISPLAY = 5

# --- Conexión Segura a BigQuery ---
@st.cache_resource
def connect_to_bigquery():
    try:
        creds_json = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(creds_json)
        client = bigquery.Client(credentials=creds, project=GCP_PROJECT_ID)
        logger.info("CONNECT_BQ: Conexión OK.")
        return client
    except Exception as e: logger.error(f"CONNECT_BQ: Error: {e}", exc_info=True); st.error(f"Error BQ: {e}."); return None

bq_client = connect_to_bigquery()
if bq_client is None: st.stop()

# --- FUNCIONES DE CARGA DE MODELOS Y PREPROCESADORES ---
@st.cache_resource
def load_joblib_object(file_path, object_name="Objeto Joblib"):
    if not os.path.exists(file_path):
        logger.error(f"LOAD_JOBLIB: Archivo '{object_name}' no en: {file_path}")
        st.error(f"Error Crítico: Archivo '{object_name}' no en '{file_path}'.")
        return None
    try:
        loaded_object = joblib.load(file_path)
        logger.info(f"LOAD_JOBLIB: {object_name} cargado desde: {file_path}")
        return loaded_object
    except Exception as e:
        logger.error(f"LOAD_JOBLIB: Error cargando {object_name} desde {file_path}: {e}", exc_info=True)
        st.error(f"Error Crítico al Cargar {object_name}: {e}")
        return None

@st.cache_data
def load_threshold_from_json(file_path):
    if not os.path.exists(file_path):
        logger.error(f"LOAD_THRESHOLD: Archivo de umbral no en: {file_path}")
        st.error(f"Error Crítico: Archivo de umbral no en '{file_path}'.")
        return 30.0
    try:
        with open(file_path, "r") as f: data = json.load(f)
        threshold_value = data.get("threshold")
        if threshold_value is None:
            logger.error(f"LOAD_THRESHOLD: Clave 'threshold' no en {file_path}.")
            st.error("Error Crítico: Formato incorrecto de archivo de umbral.")
            return 30.0
        logger.info(f"LOAD_THRESHOLD: Umbral {threshold_value} cargado desde: {file_path}")
        return float(threshold_value)
    except Exception as e: logger.error(f"LOAD_THRESHOLD: Error: {e}", exc_info=True); st.error(f"Error Cargar Umbral: {e}"); return 30.0

# --- Carga de Modelos LGBM y Umbral ---
logger.info("APP_INIT: Iniciando carga de modelos LGBM y umbral.")
pipe_low_lgbm_app = load_joblib_object(PIPE_LOW_LGBM_PATH, "Pipeline LGBM Precios Bajos")
pipe_high_lgbm_app = load_joblib_object(PIPE_HIGH_LGBM_PATH, "Pipeline LGBM Precios Altos")
threshold_lgbm_app = load_threshold_from_json(THRESHOLD_LGBM_PATH)


# --- FUNCIONES UTILITARIAS DE DATOS ---
@st.cache_data(ttl=3600)
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    query = f"SELECT table_id FROM `{_client.project}.{BIGQUERY_DATASET}`.__TABLES__ WHERE STARTS_WITH(table_id, 'monthly_') ORDER BY table_id DESC LIMIT 1"
    try:
        results = _client.query(query).result()
        if results.total_rows > 0:
            latest_table_id = list(results)[0].table_id
            logger.info(f"SNAPSHOT_TABLE: Usando tabla snapshot: {latest_table_id}")
            return f"{_client.project}.{BIGQUERY_DATASET}.{latest_table_id}"
        logger.warning("SNAPSHOT_TABLE: No se encontraron tablas snapshot 'monthly_...'.")
        st.warning("Advertencia: No se encontraron tablas de precios ('monthly_...').")
        return None
    except Exception as e: logger.error(f"SNAPSHOT_TABLE: Error: {e}", exc_info=True); st.error(f"Error al buscar tabla snapshot: {e}."); return None

POKEMON_SUFFIXES_TO_REMOVE = [' VMAX', ' VSTAR', ' V-UNION', ' V', ' GX', ' EX', ' BREAK', ' Prism Star', ' Star', ' Radiant', ' δ', ' Tag Team', ' & ', ' Light', ' Dark', ' ◇', ' ☆']
MULTI_WORD_BASE_NAMES = ["Mr. Mime", "Mime Jr.", "Farfetch'd", "Sirfetch'd", "Ho-Oh", "Porygon-Z", "Type: Null", "Tapu Koko", "Tapu Lele", "Tapu Bulu", "Tapu Fini", "Mr. Rime", "Indeedee M", "Indeedee F", "Great Tusk", "Iron Treads"] # yapf: disable

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
    query = f"""
    SELECT
        id,
        name         AS pokemon_name,
        supertype,
        subtypes,
        types,
        rarity,
        set_name,
        artist       AS artist_name,
        images_large AS image_url,
        cardmarket_url,
        tcgplayer_url
    FROM `{CARD_METADATA_TABLE}`
    """
    logger.info(f"METADATA_BQ: Ejecutando query para metadatos: {query[:100]}...")
    try:
        df = _client.query(query).to_dataframe()
        if df.empty: logger.warning("METADATA_BQ: DataFrame de metadatos vacío."); st.warning("No se pudo cargar metadatos."); return pd.DataFrame()
        expected_cols = ['artist_name', 'pokemon_name', 'rarity', 'set_name', 'types', 'supertype', 'subtypes', 'cardmarket_url', 'tcgplayer_url']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 'Unknown_Placeholder' if col not in ['cardmarket_url', 'tcgplayer_url'] else None
                logger.warning(f"METADATA_BQ: Columna '{col}' no en metadatos, añadida con placeholder/None.")
        df['base_pokemon_name_display'] = df.apply(lambda row: get_true_base_name(row['pokemon_name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES), axis=1)
        logger.info(f"METADATA_BQ: Metadatos cargados y procesados. Total filas: {len(df)}.")
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("METADATA_BQ: Error de 'db-dtypes'.", exc_info=True); st.error("Error de Dependencia: Falta 'db-dtypes'.")
        else: logger.error(f"METADATA_BQ: Error al cargar metadatos: {e}", exc_info=True); st.error(f"Error al cargar metadatos: {e}.")
        return pd.DataFrame()

# --- FUNCIÓN DE CONSULTA DE DATOS DE PRECIOS Y METADATOS COMBINADOS ---
@st.cache_data(ttl=600)
def fetch_card_data_from_bq(
    _client: bigquery.Client, latest_table_path: str, supertype_ui_filter: str | None,
    sets_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
    sort_direction: str, full_metadata_df_param: pd.DataFrame
) -> pd.DataFrame:
    logger.info(f"FETCH_BQ_DATA: Ini. SType:{supertype_ui_filter}, Sets:{len(sets_ui_filter)}, Names:{len(names_ui_filter)}, Rars:{len(rarities_ui_filter)}")
    if not latest_table_path: logger.error("FETCH_BQ_DATA_FAIL: 'latest_table_path' es None."); st.error("Error Interno: No se pudo determinar la tabla de precios."); return pd.DataFrame()
    
    ids_to_query_df = full_metadata_df_param.copy()
    if supertype_ui_filter and supertype_ui_filter != "Todos": ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]
    if rarities_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:
        name_col_to_use_for_filter = 'base_pokemon_name_display' if supertype_ui_filter == 'Pokémon' and 'base_pokemon_name_display' in ids_to_query_df.columns else 'pokemon_name'
        if name_col_to_use_for_filter in ids_to_query_df.columns: ids_to_query_df = ids_to_query_df[ids_to_query_df[name_col_to_use_for_filter].isin(names_ui_filter)]
    
    if ids_to_query_df.empty: logger.info("FETCH_BQ_DATA: No hay IDs que coincidan."); return pd.DataFrame()
    list_of_card_ids_to_query = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids_to_query: logger.info("FETCH_BQ_DATA: Lista IDs vacía."); return pd.DataFrame()

    # Query SQL actualizada con los nombres de columna correctos de las tablas de BQ
    query_sql_template = f"""
    SELECT
        meta.id,
        meta.name AS pokemon_name,         -- CORRECCIÓN: Usar meta.name y darle alias
        meta.supertype,
        meta.subtypes,
        meta.types,
        meta.set_name,
        meta.rarity,
        meta.artist AS artist_name,        -- CORRECCIÓN: Usar meta.artist y darle alias
        meta.images_large AS image_url,
        meta.cardmarket_url,
        meta.tcgplayer_url,
        prices.cm_averageSellPrice AS precio,
        prices.cm_trendPrice,
        prices.cm_avg1,
        prices.cm_avg7,
        prices.cm_avg30,
        PARSE_DATE('%Y_%m_%d', prices._TABLE_SUFFIX) AS fecha_snapshot
    FROM `{CARD_METADATA_TABLE}` AS meta
    LEFT JOIN `{latest_table_path}` AS prices ON meta.id = prices.id
    WHERE meta.id IN UNNEST(@card_ids_param)
    ORDER BY prices.cm_averageSellPrice {sort_direction}
    """
    query_params_bq = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids_to_query)]
    job_config_bq = bigquery.QueryJobConfig(query_parameters=query_params_bq)
    logger.info(f"FETCH_BQ_DATA: SQL BQ para {len(list_of_card_ids_to_query)} IDs. Orden: {sort_direction}")
    try:
        results_from_bq_df = _client.query(query_sql_template, job_config=job_config_bq).to_dataframe()
        price_cols_to_convert = ['precio', 'cm_trendPrice', 'cm_avg1', 'cm_avg7', 'cm_avg30']
        for pcol in price_cols_to_convert:
            if pcol in results_from_bq_df.columns:
                 results_from_bq_df[pcol] = pd.to_numeric(results_from_bq_df[pcol], errors='coerce')
        logger.info(f"FETCH_BQ_DATA: Consulta a BQ OK. Filas: {len(results_from_bq_df)}.")
        logger.debug(f"FETCH_BQ_DATA: Columnas en results_df: {results_from_bq_df.columns.tolist()}")
        return results_from_bq_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("FETCH_BQ_DATA_FAIL: Error de 'db-dtypes'.", exc_info=True); st.error("Error de Dependencia: Falta 'db-dtypes'.")
        else: logger.error(f"FETCH_BQ_DATA_FAIL: Error BQ: {e}", exc_info=True); st.error(f"Error al obtener datos de cartas: {e}.")
        return pd.DataFrame()

# --- FUNCIÓN DE PREDICCIÓN CON MODELOS LGBM ---
def predict_price_with_lgbm_pipelines_app(
    pipe_low_lgbm_loaded, pipe_high_lgbm_loaded, threshold_lgbm_value: float,
    card_data_for_prediction: pd.Series
) -> float | None:
    logger.info(f"LGBM_PRED_APP: Iniciando predicción para carta ID: {card_data_for_prediction.get('id', 'N/A')}")
    if not pipe_low_lgbm_loaded or not pipe_high_lgbm_loaded or threshold_lgbm_value is None:
        logger.error("LGBM_PRED_APP: Pipelines LGBM o umbral no cargados.")
        st.error("Error Interno: Modelos LGBM o umbral no disponibles.")
        return None
    try:
        input_dict = {}
        current_price_val = card_data_for_prediction.get('precio')
        input_dict['prev_price'] = float(current_price_val) if pd.notna(current_price_val) else 0.0
        if pd.isna(current_price_val): logger.warning(f"LGBM_PRED_APP: 'prev_price' (de 'precio' actual) es NaN. Usando 0.0.")
        input_dict['days_since_prev_snapshot'] = 30.0

        for col_name in ['cm_avg1', 'cm_avg7', 'cm_avg30', 'cm_trendPrice']:
            val = card_data_for_prediction.get(col_name)
            input_dict[col_name] = float(val) if pd.notna(val) else 0.0
            if pd.isna(val): logger.warning(f"LGBM_PRED_APP: Feature numérica '{col_name}' es NaN. Usando 0.0.")
        
        input_dict['artist_name'] = str(card_data_for_prediction.get('artist_name', 'Unknown_Artist'))
        input_dict['pokemon_name'] = str(card_data_for_prediction.get('pokemon_name', 'Unknown_Pokemon'))
        input_dict['rarity'] = str(card_data_for_prediction.get('rarity', 'Unknown_Rarity'))
        input_dict['set_name'] = str(card_data_for_prediction.get('set_name', 'Unknown_Set'))
        input_dict['supertype'] = str(card_data_for_prediction.get('supertype', 'Unknown_Supertype'))
        types_val = card_data_for_prediction.get('types'); input_dict['types'] = str(types_val[0]) if isinstance(types_val, list) and types_val and pd.notna(types_val[0]) else (str(types_val) if pd.notna(types_val) else 'Unknown_Type')
        subtypes_val = card_data_for_prediction.get('subtypes'); input_dict['subtypes'] = ', '.join(sorted(list(set(str(s) for s in subtypes_val if pd.notna(s))))) if isinstance(subtypes_val, list) and subtypes_val else (str(subtypes_val) if pd.notna(subtypes_val) else 'None')

        X_new_predict_df = pd.DataFrame([input_dict])
        missing_cols = [col for col in _LGBM_ALL_FEATURES_APP if col not in X_new_predict_df.columns]
        if missing_cols:
            logger.error(f"LGBM_PRED_APP: Faltan columnas en X_new_predict_df: {missing_cols}")
            st.error(f"Error Interno: Faltan datos para predicción LGBM ({', '.join(missing_cols)}).")
            return None
        
        X_new_predict_for_pipe = X_new_predict_df[_LGBM_ALL_FEATURES_APP] # Asegurar orden y selección
        threshold_feature_value = X_new_predict_for_pipe.loc[0, _LGBM_THRESHOLD_COLUMN_APP]
        active_pipe = pipe_low_lgbm_loaded if threshold_feature_value < threshold_lgbm_value else pipe_high_lgbm_loaded
        model_type_used = "Low-Price Pipe" if threshold_feature_value < threshold_lgbm_value else "High-Price Pipe"
        logger.info(f"LGBM_PRED_APP: Usando {model_type_used} basado en {_LGBM_THRESHOLD_COLUMN_APP}={threshold_feature_value}")
        pred_log = active_pipe.predict(X_new_predict_for_pipe)
        prediction_final = np.expm1(pred_log[0])
        logger.info(f"LGBM_PRED_APP: Predicción (escala original): {prediction_final:.2f}€")
        return float(prediction_final)
    except KeyError as e_key: logger.error(f"LGBM_PRED_APP: KeyError: {e_key}", exc_info=True); st.error(f"Error Datos: Falta '{e_key}'."); return None
    except Exception as e: logger.error(f"LGBM_PRED_APP_EXC: Excepción: {e}", exc_info=True); st.error(f"Error Predicción LGBM: {e}"); return None

# --- Carga de Datos Inicial de BigQuery ---
logger.info("APP_INIT: Cargando datos iniciales de BigQuery.")
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
all_card_metadata_df = get_card_metadata_with_base_names(bq_client)
if not LATEST_SNAPSHOT_TABLE or all_card_metadata_df.empty:
    logger.critical("APP_INIT_FAIL: Datos esenciales de BigQuery no cargados.")
    st.error("Error Crítico: No se pudieron cargar los datos esenciales de BigQuery.")
    st.stop()
logger.info("APP_INIT: Datos iniciales de BigQuery cargados OK.")


# --- Sidebar y Filtros ---
st.sidebar.header("Filtros y Opciones")
options_df_for_filters = all_card_metadata_df.copy()
supertype_options_list = sorted(options_df_for_filters['supertype'].dropna().unique().tolist())
select_supertype_options = ["Todos"] + supertype_options_list if supertype_options_list else ["Todos"]
selected_supertype = st.sidebar.selectbox("Categoría:", select_supertype_options, index=0, key="sb_supertype_filter_v3")
if selected_supertype != "Todos": options_df_for_filters = options_df_for_filters[options_df_for_filters['supertype'] == selected_supertype]
set_options_list = sorted(options_df_for_filters['set_name'].dropna().unique().tolist())
selected_sets = st.sidebar.multiselect("Set(s):", set_options_list, key="ms_sets_filter_v3")
if selected_sets: options_df_for_filters = options_df_for_filters[options_df_for_filters['set_name'].isin(selected_sets)]
name_label = "Nombre de Carta:"; name_col_for_options = 'pokemon_name'
if selected_supertype == 'Pokémon': name_col_for_options = 'base_pokemon_name_display'; name_label = "Pokémon (Nombre Base):"
elif selected_supertype != "Todos": name_label = f"Nombre ({selected_supertype}):"
if name_col_for_options in options_df_for_filters.columns: name_options_list = sorted(options_df_for_filters[name_col_for_options].dropna().unique().tolist())
else: name_options_list = []; logger.warning(f"SIDEBAR: Columna '{name_col_for_options}' no para filtro nombre.")
selected_names_to_filter = st.sidebar.multiselect(name_label, name_options_list, key="ms_names_filter_v3")
if selected_names_to_filter and name_col_for_options in options_df_for_filters.columns: options_df_for_filters = options_df_for_filters[options_df_for_filters[name_col_for_options].isin(selected_names_to_filter)]
rarity_options_list = sorted(options_df_for_filters['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Rareza(s):", rarity_options_list, key="ms_rarities_filter_v3")
sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1, key="rd_sort_order_v3")
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"

# --- Carga de results_df ---
logger.info("MAIN_APP: Fetcheando resultados principales de BigQuery (basado en filtros de sidebar).")
results_df = fetch_card_data_from_bq(
    bq_client, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets,
    selected_names_to_filter, selected_rarities, sort_sql, all_card_metadata_df
)
logger.info(f"MAIN_APP: 'results_df' cargado con {len(results_df)} filas.")

# --- Inicializar session_state y Lógica de Fallback ---
if 'selected_card_id_from_grid' not in st.session_state: st.session_state.selected_card_id_from_grid = None
if st.session_state.selected_card_id_from_grid is None and not results_df.empty:
    cards_with_price = results_df[pd.notna(results_df['precio'])]
    if not cards_with_price.empty:
        random_card_id = cards_with_price.sample(1).iloc[0].get('id')
        if random_card_id and pd.notna(random_card_id):
            st.session_state.selected_card_id_from_grid = random_card_id
            logger.info(f"FALLBACK_SELECT: Seleccionando carta aleatoria '{random_card_id}'.")
    else: logger.warning("FALLBACK_SELECT: No hay cartas con precio en results_df.")

is_initial_unfiltered_load = (not selected_sets and not selected_names_to_filter and not selected_rarities and (selected_supertype == "Todos" or not selected_supertype))

# --- SECCIÓN PRINCIPAL DE CONTENIDO ---
st.title("Explorador de Cartas Pokémon TCG")

if is_initial_unfiltered_load and not all_card_metadata_df.empty:
    st.header("Cartas Destacadas")
    special_illustration_rares = all_card_metadata_df[all_card_metadata_df['rarity'] == FEATURED_RARITY].copy()
    if not special_illustration_rares.empty:
        num_cards_to_show = min(len(special_illustration_rares), NUM_FEATURED_CARDS_TO_DISPLAY)
        if len(special_illustration_rares) > 0 and num_cards_to_show > 0 :
             display_cards_df = special_illustration_rares.sample(n=num_cards_to_show, random_state=1).reset_index(drop=True)
        else: display_cards_df = pd.DataFrame()
        if not display_cards_df.empty:
             cols = st.columns(num_cards_to_show)
             for i, card in display_cards_df.iterrows():
                 with cols[i]:
                     card_name_featured = card.get('pokemon_name', 'N/A')
                     card_set_featured = card.get('set_name', 'N/A')
                     image_url_featured = card.get('images_large')
                     if pd.notna(image_url_featured): st.image(image_url_featured, width=150, caption=card_set_featured)
                     else: st.warning("Imagen no disp."); st.caption(f"{card_name_featured} ({card_set_featured})")
             st.markdown("---")
    if special_illustration_rares.empty and results_df.empty and is_initial_unfiltered_load and bq_client and LATEST_SNAPSHOT_TABLE:
         st.info("No se encontraron cartas con la rareza destacada o con precio en la base de datos actual.")

elif not is_initial_unfiltered_load:
    st.header("Resultados de Cartas")
    results_df_for_aggrid_display = results_df
    if len(results_df) > MAX_ROWS_NO_FILTER and is_initial_unfiltered_load :
        st.info(f"Mostrando los primeros {MAX_ROWS_NO_FILTER} de {len(results_df)} resultados. Aplica filtros.")
        results_df_for_aggrid_display = results_df.head(MAX_ROWS_NO_FILTER)
    if not results_df_for_aggrid_display.empty:
        display_columns_mapping = {'id': 'ID', 'pokemon_name': 'Nombre Carta', 'supertype': 'Categoría', 'set_name': 'Set', 'rarity': 'Rareza', 'artist_name': 'Artista', 'precio': 'Precio (€)'}
        cols_in_df_for_display = [col for col in display_columns_mapping.keys() if col in results_df_for_aggrid_display.columns]
        final_display_df_aggrid = results_df_for_aggrid_display[cols_in_df_for_display].copy()
        final_display_df_aggrid.rename(columns=display_columns_mapping, inplace=True)
        price_display_col_name_in_aggrid = display_columns_mapping.get('Precio (€)')
        if price_display_col_name_in_aggrid and price_display_col_name_in_aggrid in final_display_df_aggrid.columns:
             final_display_df_aggrid[price_display_col_name_in_aggrid] = final_display_df_aggrid[price_display_col_name_in_aggrid].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")
        gb = GridOptionsBuilder.from_dataframe(final_display_df_aggrid)
        gb.configure_selection(selection_mode='single', use_checkbox=False); gb.configure_grid_options(domLayout='normal')
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
        gridOptions = gb.build()
        st.write("Haz clic en una fila de la tabla para ver sus detalles:")
        grid_response = AgGrid( final_display_df_aggrid, gridOptions=gridOptions, height=500, width='100%', data_return_mode=DataReturnMode.AS_INPUT, update_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=False, allow_unsafe_jscode=True, key='pokemon_aggrid_main_display_vFINAL_LGBM_2') # Changed key
        if grid_response:
            selected_rows_data = grid_response.get('selected_rows')
            if selected_rows_data and not selected_rows_data.empty:
                newly_selected_id = selected_rows_data.iloc[0]['ID']
                if newly_selected_id != st.session_state.selected_card_id_from_grid:
                    st.session_state.selected_card_id_from_grid = newly_selected_id; st.rerun()
    else: st.info("No hay cartas que coincidan con los filtros aplicados.")

# --- Sección de Detalle de Carta Seleccionada y Predicción ---
if st.session_state.selected_card_id_from_grid is not None:
    st.divider(); st.header("Detalle de Carta Seleccionada")
    card_to_display_in_detail_section = None
    id_for_detail_view = st.session_state.selected_card_id_from_grid
    source_df = results_df if (not results_df.empty and id_for_detail_view in results_df['id'].values) else all_card_metadata_df
    if not source_df.empty:
        matched_rows = source_df[source_df['id'] == id_for_detail_view]
        if not matched_rows.empty: card_to_display_in_detail_section = matched_rows.iloc[0]
        else: st.session_state.selected_card_id_from_grid = None; st.rerun()
    else: st.error("Error: No se cargaron metadatos."); st.stop()

    if card_to_display_in_detail_section is not None:
        card_name_render = card_to_display_in_detail_section.get('pokemon_name', "N/A")
        card_set_render = card_to_display_in_detail_section.get('set_name', "N/A")
        card_image_url_render = card_to_display_in_detail_section.get('image_url', None)
        card_supertype_render = card_to_display_in_detail_section.get('supertype', "N/A")
        card_rarity_render = card_to_display_in_detail_section.get('rarity', "N/A")
        card_artist_render = card_to_display_in_detail_section.get('artist_name', None)
        card_price_actual_render = card_to_display_in_detail_section.get('precio', None)
        cardmarket_url_render = card_to_display_in_detail_section.get('cardmarket_url', None)
        tcgplayer_url_render = card_to_display_in_detail_section.get('tcgplayer_url', None)
        col_img, col_info = st.columns([1, 2])
        with col_img:
            if pd.notna(card_image_url_render): st.image(card_image_url_render, caption=f"{card_name_render} ({card_set_render})", width=300)
            else: st.warning("Imagen no disponible.")
            links_html = []
            if pd.notna(cardmarket_url_render) and cardmarket_url_render.startswith("http"): links_html.append(f"<a href='{cardmarket_url_render}' target='_blank' style='...'>Cardmarket</a>")
            if pd.notna(tcgplayer_url_render) and tcgplayer_url_render.startswith("http"): links_html.append(f"<a href='{tcgplayer_url_render}' target='_blank' style='...'>TCGplayer</a>")
            if links_html: st.markdown(" ".join(links_html), unsafe_allow_html=True)
            else: st.caption("Links no disponibles.")
        with col_info:
            st.subheader(f"{card_name_render}")
            st.markdown(f"**ID:** `{card_to_display_in_detail_section.get('id')}`"); st.markdown(f"**Categoría:** {card_supertype_render}"); st.markdown(f"**Set:** {card_set_render}"); st.markdown(f"**Rareza:** {card_rarity_render}")
            if pd.notna(card_artist_render): st.markdown(f"**Artista:** {card_artist_render}")
            if pd.notna(card_price_actual_render): st.metric(label="Precio Actual (Trend €)", value=f"€{card_price_actual_render:.2f}")
            else: st.markdown("**Precio Actual (Trend €):** N/A")
            st.markdown("---"); st.subheader("Predicción de Precio (Modelo LGBM Estimado)")
            
            if pipe_low_lgbm_app and pipe_high_lgbm_app and threshold_lgbm_app is not None:
                if pd.notna(card_price_actual_render) and pd.notna(card_to_display_in_detail_section.get('cm_avg7')):
                     if st.button("⚡ Estimar Precio Futuro (LGBM)", key=f"predict_lgbm_btn_{card_to_display_in_detail_section.get('id')}"):
                         with st.spinner("Calculando estimación (LGBM)..."):
                             pred_price = predict_price_with_lgbm_pipelines_app(
                                 pipe_low_lgbm_app, pipe_high_lgbm_app, threshold_lgbm_app,
                                 card_to_display_in_detail_section
                             )
                         if pred_price is not None:
                             delta = pred_price - card_price_actual_render
                             delta_color = "normal" if delta < -0.01 else ("inverse" if delta > 0.01 else "off")
                             st.metric(label="Precio Estimado (LGBM)", value=f"€{pred_price:.2f}", delta=f"{delta:+.2f}€", delta_color=delta_color)
                         else: st.warning("No se pudo obtener estimación (LGBM).")
                else:
                     st.info("Datos insuficientes (precio actual o cm_avg7) para la estimación con LGBM.")
            else: st.warning("Modelos LGBM o umbral no cargados.")
else:
    if results_df.empty and not is_initial_unfiltered_load: st.info("No se encontraron cartas con los filtros seleccionados.")
    elif results_df.empty and is_initial_unfiltered_load:
        if not all_card_metadata_df.empty and not all_card_metadata_df[all_card_metadata_df['rarity'] == FEATURED_RARITY].empty: pass
        else: st.info("No se encontraron cartas destacadas ni otros resultados iniciales.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Pokémon TCG Explorer v1.11 | LGBM")
