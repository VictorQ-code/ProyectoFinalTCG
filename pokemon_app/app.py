import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re
import logging
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import numpy as np
import os
import tensorflow as tf
import joblib
import typing
import random
import json

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

# --- RUTAS Y NOMBRES DE ARCHIVOS DE MODELOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES_DIR = os.path.join(BASE_DIR, "model_files")

# MLP (Predicción Futura) - Verifica que esta sea tu estructura real
MLP_ARTIFACTS_SUBDIR = "mlp_v1" # Subcarpeta donde están los artefactos del MLP
MLP_SAVED_MODEL_PATH = os.path.join(MODEL_FILES_DIR, MLP_ARTIFACTS_SUBDIR)
MLP_OHE_PKL_FILENAME = "ohe_mlp_cat.pkl"
MLP_SCALER_PKL_FILENAME = "scaler_mlp_num.pkl"
MLP_OHE_PATH = os.path.join(MLP_SAVED_MODEL_PATH, MLP_OHE_PKL_FILENAME)
MLP_SCALER_PATH = os.path.join(MLP_SAVED_MODEL_PATH, MLP_SCALER_PKL_FILENAME)

# LightGBM Pipelines (Precio Justo)
LGBM_HIGH_PKL_PATH = os.path.join(MODEL_FILES_DIR, "High/modelo_pipe_high.pkl")
LGBM_LOW_PKL_PATH = os.path.join(MODEL_FILES_DIR, "Low/modelo_pipe_low.pkl")
LGBM_THRESHOLD_JSON_PATH = os.path.join(MODEL_FILES_DIR, "threshold.json")


# --- CONFIGURACIÓN DEL MODELO MLP ---
_MLP_NUM_COLS = ['price_t0_log', 'days_diff']
_MLP_CAT_COLS = ['artist_name', 'pokemon_name', 'rarity', 'set_name', 'types', 'supertype', 'subtypes']
_MLP_INPUT_KEY = 'inputs'
_MLP_OUTPUT_KEY = 'output_0'
_MLP_TARGET_LOG = True
_MLP_DAYS_DIFF = 29.0

# --- CONFIGURACIÓN DE PIPELINES LGBM ---
_LGBM_NUM_FEATURES_INPUT = ['cm_avg1', 'cm_avg7', 'cm_avg30', 'cm_trendPrice']
_LGBM_CAT_FEATURES_INPUT = ['rarity', 'supertype', 'subtypes', 'types', 'set_name']
_LGBM_ALL_INPUT_COLS = _LGBM_NUM_FEATURES_INPUT + _LGBM_CAT_FEATURES_INPUT
_LGBM_TARGET_LOG = True

# --- CONFIGURACIÓN PARA CARTAS DESTACADAS ---
FEATURED_RARITY = 'Special Illustration Rare'
NUM_FEATURED_CARDS_TO_DISPLAY = 5

# --- INICIALIZAR SESSION STATE TEMPRANO ---
if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
    logger.info("SESSION_STATE_INIT: 'selected_card_id_from_grid' inicializado a None (temprano).")


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
if bq_client is None:
    logger.critical("APP_STOP: No se pudo conectar a BigQuery.")
    st.stop()

# --- FUNCIONES DE CARGA DE ARTEFACTOS ---
@st.cache_resource
def load_tf_model_as_layer(model_path): # Para el MLP SavedModel
    saved_model_pb_path = os.path.join(model_path, "saved_model.pb")
    if not os.path.exists(saved_model_pb_path):
        logger.error(f"LOAD_TF_LAYER: 'saved_model.pb' no encontrado en la ruta del modelo: {model_path}")
        st.error(f"Error Crítico: El archivo 'saved_model.pb' no se encuentra en '{model_path}'. Verifica la constante MLP_SAVED_MODEL_PATH y tu estructura de archivos en GitHub.")
        return None
    try:
        logger.info(f"LOAD_TF_LAYER: Intentando cargar SavedModel como TFSMLayer desde: {model_path}")
        model_as_layer_obj = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        logger.info(f"LOAD_TF_LAYER: SavedModel cargado exitosamente como TFSMLayer.")
        try: logger.info(f"LOAD_TF_LAYER: Call Signature (info interna): {model_as_layer_obj._call_signature}")
        except AttributeError: logger.warning("LOAD_TF_LAYER: No se pudo acceder a '_call_signature'.")
        return model_as_layer_obj
    except Exception as e:
        logger.error(f"LOAD_TF_LAYER: Error crítico al cargar SavedModel como TFSMLayer desde {model_path}: {e}", exc_info=True)
        st.error(f"Error Crítico al Cargar Modelo MLP: {e}.")
        return None

@st.cache_resource
def load_sklearn_pipeline(file_path, pipeline_name="Sklearn Pipeline"): # Para los LGBM .pkl
    if not os.path.exists(file_path):
        logger.warning(f"LOAD_SKL_PIPE: Archivo '{pipeline_name}' no encontrado en: {file_path}")
        return None
    try:
        pipeline = joblib.load(file_path)
        logger.info(f"LOAD_SKL_PIPE: {pipeline_name} cargado exitosamente desde: {file_path}")
        return pipeline
    except Exception as e:
        logger.error(f"LOAD_SKL_PIPE: Error crítico al cargar {pipeline_name} desde {file_path}: {e}", exc_info=True)
        st.error(f"Error Crítico al Cargar Pipeline '{pipeline_name}': {e}")
        return None

@st.cache_resource
def load_joblib_preprocessor(file_path, preprocessor_name="Preprocessor"): # Para OHE y Scaler del MLP
    if not os.path.exists(file_path):
        logger.error(f"LOAD_PREPROC: Archivo '{preprocessor_name}' no encontrado en: {file_path}")
        st.error(f"Error Crítico: El archivo preprocesador '{preprocessor_name}' no se encuentra en '{file_path}'. Verifica las constantes de ruta del MLP.")
        return None
    try:
        preprocessor = joblib.load(file_path)
        logger.info(f"LOAD_PREPROC: {preprocessor_name} cargado exitosamente desde: {file_path}")
        return preprocessor
    except Exception as e:
        logger.error(f"LOAD_PREPROC: Error crítico al cargar {preprocessor_name} desde {file_path}: {e}", exc_info=True)
        st.error(f"Error Crítico al Cargar Preprocesador '{preprocessor_name}': {e}")
        return None

@st.cache_resource
def load_json_config(file_path, config_name="JSON Config"):
    if not os.path.exists(file_path):
        logger.warning(f"LOAD_JSON: Archivo '{config_name}' no encontrado en: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        logger.info(f"LOAD_JSON: {config_name} cargado exitosamente desde: {file_path}")
        return config
    except Exception as e:
        logger.error(f"LOAD_JSON: Error crítico al cargar {config_name} desde {file_path}: {e}", exc_info=True)
        st.error(f"Error Crítico al Cargar Configuración '{config_name}': {e}")
        return None

# --- Carga de TODOS los Modelos y Preprocesadores ---
logger.info("APP_INIT: Iniciando carga de artefactos de TODOS los modelos.")
# MLP
mlp_model_layer = load_tf_model_as_layer(MLP_SAVED_MODEL_PATH)
mlp_ohe = load_joblib_preprocessor(MLP_OHE_PATH, "MLP OneHotEncoder")
mlp_scaler = load_joblib_preprocessor(MLP_SCALER_PATH, "MLP ScalerNumérico")

# LightGBM Pipelines
lgbm_pipeline_high = load_sklearn_pipeline(LGBM_HIGH_PKL_PATH, "Pipeline LGBM High")
lgbm_pipeline_low = load_sklearn_pipeline(LGBM_LOW_PKL_PATH, "Pipeline LGBM Low")
lgbm_threshold_config = load_json_config(LGBM_THRESHOLD_JSON_PATH, "LGBM Threshold Config")
if lgbm_threshold_config is None: lgbm_threshold_config = {"threshold": 30.0} # Default si no carga


# --- Funciones Auxiliares de Datos ---
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
        st.warning("Advertencia: No se encontraron tablas de precios ('monthly_...'). La información de precios podría no estar disponible.")
        return None
    except Exception as e:
        logger.error(f"SNAPSHOT_TABLE: Error buscando tabla snapshot: {e}", exc_info=True)
        st.error(f"Error al buscar la tabla de precios más reciente: {e}.")
        return None

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
        id, name, name as pokemon_name, -- Añadir pokemon_name para consistencia con MLP
        supertype, subtypes, types,
        rarity, set_id, set_name,
        artist, artist as artist_name, -- Añadir artist_name para consistencia con MLP
        images_large, cardmarket_url, tcgplayer_url
    FROM `{CARD_METADATA_TABLE}`
    """
    logger.info(f"METADATA_BQ: Ejecutando query para metadatos: {query[:100]}...")
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            logger.warning("METADATA_BQ: DataFrame de metadatos vacío devuelto por BigQuery.")
            st.warning("No se pudieron cargar los metadatos de las cartas desde BigQuery.")
            return pd.DataFrame()
        for col_to_check in ['cardmarket_url', 'tcgplayer_url', 'types', 'subtypes', 'artist_name', 'pokemon_name']:
            if col_to_check not in df.columns:
                if col_to_check == 'artist_name' and 'artist' in df.columns: df['artist_name'] = df['artist']
                elif col_to_check == 'pokemon_name' and 'name' in df.columns: df['pokemon_name'] = df['name']
                else: df[col_to_check] = None
                logger.warning(f"METADATA_BQ: Columna '{col_to_check}' no encontrada o creada en metadatos, añadida como None/placeholder.")
        df['base_pokemon_name'] = df.apply(lambda row: get_true_base_name(row['name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES), axis=1)
        logger.info(f"METADATA_BQ: Metadatos cargados y procesados. Total filas: {len(df)}.")
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("METADATA_BQ: Error de 'db-dtypes'.", exc_info=True); st.error("Error de Dependencia: Falta 'db-dtypes'.")
        else: logger.error(f"METADATA_BQ: Error al cargar metadatos de BigQuery: {e}", exc_info=True); st.error(f"Error al cargar metadatos de cartas: {e}.")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_card_data_from_bq(
    _client: bigquery.Client, latest_table_path: str, supertype_ui_filter: str | None,
    sets_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
    sort_direction: str, full_metadata_df_param: pd.DataFrame
) -> pd.DataFrame:
    logger.info(f"FETCH_BQ_DATA: Ini. SType:{supertype_ui_filter}, Sets:{len(sets_ui_filter)}, Names:{len(names_ui_filter)}, Rars:{len(rarities_ui_filter)}")
    if not latest_table_path: logger.error("FETCH_BQ_DATA_FAIL: 'latest_table_path' es None."); st.error("Error Interno: No se pudo determinar la tabla de precios."); return pd.DataFrame()
    ids_to_query_df = full_metadata_df_param.copy()
    # Aplicar filtros de sidebar a los metadatos para obtener la lista de IDs a consultar
    if supertype_ui_filter and supertype_ui_filter != "Todos": ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]
    if rarities_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:
        actual_name_col_to_filter_on = 'base_pokemon_name' if supertype_ui_filter == 'Pokémon' else 'name'
        if actual_name_col_to_filter_on in ids_to_query_df.columns: ids_to_query_df = ids_to_query_df[ids_to_query_df[actual_name_col_to_filter_on].isin(names_ui_filter)]
    if ids_to_query_df.empty: logger.info("FETCH_BQ_DATA: No hay IDs que coincidan con filtros de metadatos."); return pd.DataFrame()
    list_of_card_ids_to_query = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids_to_query: logger.info("FETCH_BQ_DATA: Lista IDs vacía después de filtrar metadatos."); return pd.DataFrame()

    # Query para obtener datos de la tabla de precios y unir con metadatos seleccionados
    query_sql_template = f"""
    SELECT
        meta.id, meta.name, meta.name AS pokemon_name, meta.supertype,
        meta.subtypes, meta.types,
        meta.set_name, meta.rarity, meta.artist, meta.artist as artist_name,
        meta.images_large AS image_url,
        meta.cardmarket_url, meta.tcgplayer_url,
        prices.cm_averageSellPrice AS price,
        prices.cm_avg1, prices.cm_avg7, prices.cm_avg30, prices.cm_trendPrice
    FROM `{CARD_METADATA_TABLE}` AS meta
    JOIN `{latest_table_path}` AS prices ON meta.id = prices.id
    WHERE meta.id IN UNNEST(@card_ids_param)
    ORDER BY prices.cm_trendPrice {sort_direction}
    """
    query_params_bq = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids_to_query)]
    job_config_bq = bigquery.QueryJobConfig(query_parameters=query_params_bq)
    logger.info(f"FETCH_BQ_DATA: SQL BQ para {len(list_of_card_ids_to_query)} IDs. Orden: {sort_direction}")
    try:
        results_from_bq_df = _client.query(query_sql_template, job_config=job_config_bq).to_dataframe()
        price_cols_to_convert = ['price', 'cm_avg1', 'cm_avg7', 'cm_avg30', 'cm_trendPrice']
        for p_col in price_cols_to_convert:
            if p_col in results_from_bq_df.columns:
                results_from_bq_df[p_col] = pd.to_numeric(results_from_bq_df[p_col], errors='coerce')
        logger.info(f"FETCH_BQ_DATA: Consulta a BQ OK. Filas devueltas: {len(results_from_bq_df)}.")
        return results_from_bq_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("FETCH_BQ_DATA_FAIL: Error de 'db-dtypes'.", exc_info=True); st.error("Error de Dependencia: Falta 'db-dtypes'.")
        else: logger.error(f"FETCH_BQ_DATA_FAIL: Error BQ: {e}", exc_info=True); st.error(f"Error al obtener datos de cartas: {e}.")
        return pd.DataFrame()


# --- FUNCIONES DE PREDICCIÓN ---
# MLP
def predict_price_with_local_tf_layer(
    model_layer: tf.keras.layers.TFSMLayer, ohe_mlp: typing.Any, scaler_mlp: typing.Any,
    card_data_series: pd.Series
) -> float | None:
    logger.info(f"PREDICT_MLP: Iniciando para ID: {card_data_series.get('id', 'N/A')}")
    if not model_layer or not ohe_mlp or not scaler_mlp: return None
    try:
        data_dict = {}
        price_val = card_data_series.get('price')
        data_dict['price_t0_log'] = np.log1p(price_val) if pd.notna(price_val) and price_val > 0 else np.log1p(0)
        data_dict['days_diff'] = float(_MLP_DAYS_DIFF)
        for col in _MLP_CAT_COLS:
            val = card_data_series.get(col)
            if col == 'types':
                if isinstance(val, list) and val: data_dict[col] = str(val[0]) if pd.notna(val[0]) else 'Unknown_Type'
                elif pd.notna(val): data_dict[col] = str(val)
                else: data_dict[col] = 'Unknown_Type'
            elif col == 'subtypes':
                if isinstance(val, list) and val: cleaned = [str(s) for s in val if pd.notna(s)]; data_dict[col] = ', '.join(sorted(list(set(cleaned)))) if cleaned else 'None'
                elif pd.notna(val): data_dict[col] = str(val)
                else: data_dict[col] = 'None'
            else: data_dict[col] = str(val) if pd.notna(val) else f'Unknown_{col}'

        df_prep = pd.DataFrame([data_dict])
        df_prep = df_prep[_MLP_NUM_COLS + _MLP_CAT_COLS]

        num_feat = scaler_mlp.transform(df_prep[_MLP_NUM_COLS].fillna(0))
        cat_feat = mlp_ohe.transform(df_prep[_MLP_CAT_COLS].astype(str))
        X_final = np.concatenate([num_feat, cat_feat], axis=1)

        EXPECTED_MLP_FEATURES = 4865 # Actualizar si es diferente
        if X_final.shape[1] != EXPECTED_MLP_FEATURES:
            logger.error(f"MLP SHAPE MISMATCH! Expected {EXPECTED_MLP_FEATURES}, got {X_final.shape[1]}")
            return None
        
        pred_raw = model_layer(**{_MLP_INPUT_KEY: tf.convert_to_tensor(X_final, dtype=tf.float32)})
        pred_tensor = pred_raw[_MLP_OUTPUT_KEY]
        pred_numeric = pred_tensor.numpy()[0][0]
        return np.expm1(pred_numeric) if _MLP_TARGET_LOG else pred_numeric
    except Exception as e: logger.error(f"PREDICT_MLP_EXC: {e}", exc_info=True); return None

# LightGBM
def predict_price_with_lgbm_pipelines_app(
    input_df_row: pd.DataFrame, config: dict, pipe_high, pipe_low
) -> typing.Tuple[typing.Optional[float], str]:
    chosen_pipeline_name = "None"; final_prediction = None
    try: X_new_predict = input_df_row[_LGBM_ALL_INPUT_COLS]
    except KeyError as e: logger.error(f"LGBM_PRED_FAIL: Faltan cols en input: {e}"); return None, "Error_Input_Cols"
    
    # Asegurarse que las columnas numéricas sean numéricas para la decisión y predicción
    for num_col in _LGBM_NUM_FEATURES_INPUT:
        X_new_predict[num_col] = pd.to_numeric(X_new_predict[num_col], errors='coerce').fillna(0) # Imputar NaNs con 0
        
    cm_avg7_val = X_new_predict['cm_avg7'].iloc[0]
    threshold = config.get('threshold', 30.0)
    active_pipe = None
    if pd.notna(cm_avg7_val) and cm_avg7_val >= threshold:
        if pipe_high: active_pipe = pipe_high; chosen_pipeline_name = "LGBM High"
        elif pipe_low: active_pipe = pipe_low; chosen_pipeline_name = "LGBM Low (Fallback H)"
    else:
        if pipe_low: active_pipe = pipe_low; chosen_pipeline_name = "LGBM Low"
        elif pipe_high: active_pipe = pipe_high; chosen_pipeline_name = "LGBM High (Fallback L)"
    if active_pipe:
        try:
            logger.debug(f"LGBM PRED INPUT DF for {chosen_pipeline_name}:\n{X_new_predict.to_string(index=False)}")
            # Asegurar que las columnas categóricas sean strings antes de OHE dentro del pipeline
            for cat_col in _LGBM_CAT_FEATURES_INPUT:
                 X_new_predict[cat_col] = X_new_predict[cat_col].astype(str).fillna('Missing_Value')

            pred_log = active_pipe.predict(X_new_predict)
            pred_numeric = pred_log[0]
            final_prediction = np.expm1(pred_numeric) if _LGBM_TARGET_LOG else pred_numeric
        except Exception as e_pipe: logger.error(f"LGBM_PRED_PIPE_EXC: {e_pipe}", exc_info=True); final_prediction = None
    else: logger.error("LGBM_PRED_FAIL: Ningún pipeline LGBM seleccionado o cargado.")
    return final_prediction, chosen_pipeline_name


# --- Carga de Datos Inicial de BigQuery ---
logger.info("APP_INIT: Cargando datos iniciales de BigQuery.")
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
all_card_metadata_df = get_card_metadata_with_base_names(bq_client) # Contiene types y subtypes

if not LATEST_SNAPSHOT_TABLE or all_card_metadata_df.empty:
    logger.critical("APP_INIT_FAIL: Datos esenciales de BigQuery no cargados.")
    st.error("Error Crítico: No se pudieron cargar los datos esenciales de BigQuery. La aplicación se detendrá.")
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
name_label = "Nombre de Carta:"; name_col_for_options = 'name'
if selected_supertype == 'Pokémon': name_col_for_options = 'base_pokemon_name'; name_label = "Pokémon (Nombre Base):"
elif selected_supertype != "Todos": name_label = f"Nombre ({selected_supertype}):"
if name_col_for_options in options_df_for_filters.columns: name_options_list = sorted(options_df_for_filters[name_col_for_options].dropna().unique().tolist())
else: name_options_list = []; logger.warning(f"SIDEBAR: Columna '{name_col_for_options}' no encontrada para filtro de nombre.")
selected_names_to_filter = st.sidebar.multiselect(name_label, name_options_list, key="ms_names_filter_v3")
if selected_names_to_filter and name_col_for_options in options_df_for_filters.columns: options_df_for_filters = options_df_for_filters[options_df_for_filters[name_col_for_options].isin(selected_names_to_filter)]
rarity_options_list = sorted(options_df_for_filters['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Rareza(s):", rarity_options_list, key="ms_rarities_filter_v3")
sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1, key="rd_sort_order_v3")
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"

# --- Carga de results_df (después de filtros sidebar) ---
logger.info("MAIN_APP: Fetcheando resultados principales de BigQuery (basado en filtros de sidebar).")
results_df = fetch_card_data_from_bq(
    bq_client, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets,
    selected_names_to_filter, selected_rarities, sort_sql, all_card_metadata_df
)
logger.info(f"MAIN_APP: 'results_df' cargado con {len(results_df)} filas (reflejando filtros).")


# --- Inicializar session_state key para la carta seleccionada ---
# Esta inicialización ahora está antes de la lógica de fallback
logger.info(f"SESSION_STATE (pre-init check): ID en session_state: {st.session_state.get('selected_card_id_from_grid')}")
if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
    logger.info("SESSION_STATE_INIT: 'selected_card_id_from_grid' inicializado a None.")
logger.info(f"SESSION_STATE (post-init check): ID en session_state: {st.session_state.get('selected_card_id_from_grid')}")


# --- Lógica para establecer la carta seleccionada al inicio si no hay nada ---
if st.session_state.selected_card_id_from_grid is None and not results_df.empty:
    cards_with_price = results_df[pd.notna(results_df['price'])]
    if not cards_with_price.empty:
        random_card_row = cards_with_price.sample(1).iloc[0]
        random_card_id = random_card_row.get('id')
        if random_card_id and pd.notna(random_card_id):
            st.session_state.selected_card_id_from_grid = random_card_id
            logger.info(f"FALLBACK_SELECT: Seleccionando carta aleatoria con precio como fallback: '{random_card_id}'.")
    else:
        logger.warning("FALLBACK_SELECT: No se encontraron cartas con precio en los resultados filtrados para seleccionar un fallback.")
logger.info(f"SESSION_STATE (post-fallback): ID en session_state: {st.session_state.get('selected_card_id_from_grid')}")


# --- Determinar si estamos en la carga inicial sin filtros ---
is_initial_unfiltered_load = (not selected_sets and not selected_names_to_filter and not selected_rarities and (selected_supertype == "Todos" or not selected_supertype))


# --- SECCIÓN PRINCIPAL DE CONTENIDO ---
st.title("Explorador de Cartas Pokémon TCG")

# --- Mostrar Cartas Destacadas O Tabla de Resultados ---
if is_initial_unfiltered_load and not all_card_metadata_df.empty:
    st.header("Cartas Destacadas")
    special_illustration_rares = all_card_metadata_df[all_card_metadata_df['rarity'] == FEATURED_RARITY].copy()
    if not special_illustration_rares.empty:
        num_cards_to_show = min(len(special_illustration_rares), NUM_FEATURED_CARDS_TO_DISPLAY)
        if len(special_illustration_rares) > 0 and num_cards_to_show > 0 :
             sample_indices = random.sample(special_illustration_rares.index.tolist(), num_cards_to_show)
             display_cards_df = special_illustration_rares.loc[sample_indices].reset_index(drop=True)
        else: display_cards_df = pd.DataFrame()
        if not display_cards_df.empty:
             cols = st.columns(num_cards_to_show)
             for i, card in display_cards_df.iterrows():
                 with cols[i]:
                     card_name_featured = card.get('name', 'N/A'); card_set_featured = card.get('set_name', 'N/A')
                     image_url_featured = card.get('images_large')
                     if pd.notna(image_url_featured): st.image(image_url_featured, width=150, caption=card_set_featured)
                     else: st.warning("Imagen no disponible"); st.caption(f"{card_name_featured} ({card_set_featured})")
             st.markdown("---")
        elif special_illustration_rares.empty: logger.info(f"FEATURED_CARDS: No hay cartas '{FEATURED_RARITY}'.")
    if special_illustration_rares.empty and results_df.empty and is_initial_unfiltered_load and bq_client and LATEST_SNAPSHOT_TABLE:
         st.info("No se encontraron cartas con la rareza destacada o con precio en la base de datos actual.")

elif not is_initial_unfiltered_load: # Tabla visible solo al aplicar filtros
    st.header("Resultados de Cartas")
    logger.info(f"AGGRID_RENDERING: ID en session_state ANTES de AgGrid: {st.session_state.get('selected_card_id_from_grid')}")
    results_df_for_aggrid_display = results_df
    if len(results_df) > MAX_ROWS_NO_FILTER:
        st.info(f"Mostrando los primeros {MAX_ROWS_NO_FILTER} de {len(results_df)} resultados. Aplica filtros.")
        results_df_for_aggrid_display = results_df.head(MAX_ROWS_NO_FILTER)
    grid_response = None
    if not results_df_for_aggrid_display.empty:
        display_columns_mapping = {'id': 'ID', 'pokemon_name': 'Nombre Carta', 'supertype': 'Categoría', 'set_name': 'Set', 'rarity': 'Rareza', 'artist': 'Artista', 'price': 'Precio (Trend €)'}
        cols_in_df_for_display = [col for col in display_columns_mapping.keys() if col in results_df_for_aggrid_display.columns]
        final_display_df_aggrid = results_df_for_aggrid_display[cols_in_df_for_display].copy()
        final_display_df_aggrid.rename(columns=display_columns_mapping, inplace=True)
        price_col_aggrid = display_columns_mapping.get('price')
        if price_col_aggrid and price_col_aggrid in final_display_df_aggrid.columns:
             final_display_df_aggrid[price_col_aggrid] = final_display_df_aggrid[price_col_aggrid].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")
        gb = GridOptionsBuilder.from_dataframe(final_display_df_aggrid)
        gb.configure_selection(selection_mode='single', use_checkbox=False); gb.configure_grid_options(domLayout='normal')
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
        gridOptions = gb.build()
        st.write("Haz clic en una fila de la tabla para ver sus detalles y opciones de predicción:")
        grid_response = AgGrid( final_display_df_aggrid, gridOptions=gridOptions, height=500, width='100%', data_return_mode=DataReturnMode.AS_INPUT, update_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=False, allow_unsafe_jscode=True, key='pokemon_aggrid_main_display_vFINAL')
    else: logger.info("AGGRID_RENDERING: No hay datos para mostrar en AgGrid.")
    if grid_response:
        newly_selected_id = None; selected_rows = grid_response.get('selected_rows')
        if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
            try:
                if 'ID' in selected_rows.iloc[0]: newly_selected_id = selected_rows.iloc[0]['ID']
            except Exception as e: logger.error(f"AGGRID_HANDLER_DF_ERR: {e}", exc_info=True); newly_selected_id = None
        elif isinstance(selected_rows, list) and selected_rows:
            try:
                if isinstance(selected_rows[0], dict): newly_selected_id = selected_rows[0].get('ID')
            except Exception as e: logger.error(f"AGGRID_HANDLER_LIST_ERR: {e}", exc_info=True); newly_selected_id = None
        current_id = st.session_state.get('selected_card_id_from_grid')
        if newly_selected_id is not None and newly_selected_id != current_id:
            logger.info(f"AGGRID_SELECT_CHANGE: De '{current_id}' a '{newly_selected_id}'. Rerunning.")
            st.session_state.selected_card_id_from_grid = newly_selected_id
            st.rerun()

# --- Sección de Detalle de Carta Seleccionada y Predicción ---
if st.session_state.selected_card_id_from_grid is not None:
    st.divider(); st.header("Detalle de Carta Seleccionada")
    card_to_display_in_detail_section = None
    id_for_detail_view_from_session = st.session_state.get('selected_card_id_from_grid')
    source_df = results_df if (not results_df.empty and id_for_detail_view_from_session in results_df['id'].values) else all_card_metadata_df
    if not source_df.empty:
        matched_rows = source_df[source_df['id'] == id_for_detail_view_from_session]
        if not matched_rows.empty: card_to_display_in_detail_section = matched_rows.iloc[0]
        else: logger.warning(f"DETAIL_NOT_FOUND: ID '{id_for_detail_view_from_session}' no en fuente. Resetting."); st.session_state.selected_card_id_from_grid = None; st.rerun()
    else: logger.warning(f"DETAIL_NO_DATA: Fuente de detalles vacía para ID '{id_for_detail_view_from_session}'."); st.error("Error: Datos no cargados para detalles.")

    if card_to_display_in_detail_section is not None:
        card_id_render = card_to_display_in_detail_section.get('id', "N/A")
        card_name_render = card_to_display_in_detail_section.get('pokemon_name', card_to_display_in_detail_section.get('name', "N/A"))
        card_set_render = card_to_display_in_detail_section.get('set_name', "N/A")
        card_image_url_render = card_to_display_in_detail_section.get('image_url', None)
        card_supertype_render = card_to_display_in_detail_section.get('supertype', "N/A")
        card_rarity_render = card_to_display_in_detail_section.get('rarity', "N/A")
        card_artist_render = card_to_display_in_detail_section.get('artist_name', card_to_display_in_detail_section.get('artist', None))
        card_price_actual_render = card_to_display_in_detail_section.get('price', None)
        if pd.isna(card_price_actual_render) and id_for_detail_view_from_session and not results_df.empty and id_for_detail_view_from_session in results_df['id'].values:
             price_check = results_df[results_df['id'] == id_for_detail_view_from_session]['price'].iloc[0]
             if pd.notna(price_check): card_price_actual_render = price_check
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
            st.markdown(f"**ID:** `{card_id_render}`"); st.markdown(f"**Categoría:** {card_supertype_render}"); st.markdown(f"**Set:** {card_set_render}"); st.markdown(f"**Rareza:** {card_rarity_render}")
            if pd.notna(card_artist_render): st.markdown(f"**Artista:** {card_artist_render}")
            if pd.notna(card_price_actual_render): st.metric(label="Precio Actual (Trend €)", value=f"€{card_price_actual_render:.2f}")
            else: st.markdown("**Precio Actual (Trend €):** N/A")
            st.markdown("---"); st.subheader("Estimaciones de Precio")
            if mlp_model_layer and mlp_ohe and mlp_scaler:
                if pd.notna(card_price_actual_render):
                    if st.button("🔮 Estimar Precio Futuro (MLP)", key=f"predict_mlp_btn_{card_id_render}"):
                        with st.spinner("Calculando estimación futura (MLP)..."):
                            pred_price_mlp = predict_price_with_local_tf_layer(mlp_model_layer, mlp_ohe, mlp_scaler, card_to_display_in_detail_section)
                        if pred_price_mlp is not None:
                            delta_mlp = pred_price_mlp - card_price_actual_render
                            delta_color_mlp = "normal" if delta_mlp < -0.01 else ("inverse" if delta_mlp > 0.01 else "off")
                            st.metric(label="Estimado Futuro (MLP)", value=f"€{pred_price_mlp:.2f}", delta=f"{delta_mlp:+.2f}€ vs Actual", delta_color=delta_color_mlp)
                        else: st.warning("No se pudo obtener estimación futura (MLP).")
                else: st.info("Estimación futura (MLP) no posible sin precio actual.")
            else: st.caption("Modelo MLP no disponible.")
            if lgbm_pipeline_high or lgbm_pipeline_low:
                input_df_for_lgbm = pd.DataFrame([card_to_display_in_detail_section])
                can_predict_lgbm = all(col in input_df_for_lgbm.columns and pd.notna(input_df_for_lgbm[col].iloc[0]) for col in _LGBM_NUM_FEATURES_INPUT)
                if can_predict_lgbm:
                    if st.button("⚖️ Calcular Precio Justo (LGBM)", key=f"predict_lgbm_btn_{card_id_render}"):
                        with st.spinner("Calculando precio justo (LGBM)..."):
                            pred_price_lgbm, pipeline_lgbm = predict_price_with_lgbm_pipelines_app(input_df_for_lgbm, lgbm_threshold_config, lgbm_pipeline_high, lgbm_pipeline_low)
                        if pred_price_lgbm is not None:
                            st.metric(label=f"Precio Justo Estimado ({pipeline_lgbm})", value=f"€{pred_price_lgbm:.2f}")
                            if pd.notna(card_price_actual_render): delta_lgbm = pred_price_lgbm - card_price_actual_render; st.caption(f"Diferencia con precio actual: {delta_lgbm:+.2f}€")
                        else: st.warning("No se pudo obtener el precio justo (LGBM).")
                else: st.info("Cálculo de precio justo (LGBM) no posible sin datos de precios recientes (cm_avgX, trendPrice).")
            else: st.caption("Modelos LGBM no disponibles.")
else:
    if results_df.empty:
         if not is_initial_unfiltered_load: st.info("No se encontraron cartas con los filtros seleccionados.")
         else:
              if bq_client and LATEST_SNAPSHOT_TABLE:
                   if not all_card_metadata_df.empty: st.info("No se encontraron cartas con precio en la base de datos actual.")
                   else: st.error("Error interno: No se cargaron los datos de metadatos.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Pokémon TCG Explorer v1.8 | TF: {tf.__version__}")
