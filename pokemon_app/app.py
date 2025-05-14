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
MODEL_FILES_DIR = os.path.join(BASE_DIR, "model_files") # Carpeta principal de modelos

# MLP (Predicción Futura)
MLP_ARTIFACTS_SUBDIR = "mlp_v1" # Subcarpeta donde están los artefactos del MLP
MLP_SAVED_MODEL_PATH = os.path.join(MODEL_FILES_DIR, MLP_ARTIFACTS_SUBDIR)
MLP_OHE_PKL_FILENAME = "ohe_mlp_cat.pkl"
MLP_SCALER_PKL_FILENAME = "scaler_mlp_num.pkl"
MLP_OHE_PATH = os.path.join(MLP_SAVED_MODEL_PATH, MLP_OHE_PKL_FILENAME)
MLP_SCALER_PATH = os.path.join(MLP_SAVED_MODEL_PATH, MLP_SCALER_PKL_FILENAME)

# LightGBM Pipelines (Precio Justo)
LGBM_MODELS_SUBDIR = os.path.join(MODEL_FILES_DIR, "lgbm_models") # Subdirectorio para modelos LGBM
PIPE_LOW_PKL_FILENAME = "modelo_pipe_low.pkl"
PIPE_HIGH_PKL_FILENAME = "modelo_pipe_high.pkl"
THRESHOLD_JSON_FILENAME = "threshold.json"

LGBM_HIGH_PKL_PATH = os.path.join(LGBM_MODELS_SUBDIR, PIPE_HIGH_PKL_FILENAME) # DEFINICIÓN
LGBM_LOW_PKL_PATH = os.path.join(LGBM_MODELS_SUBDIR, PIPE_LOW_PKL_FILENAME)   # DEFINICIÓN
THRESHOLD_LGBM_PATH = os.path.join(LGBM_MODELS_SUBDIR, THRESHOLD_JSON_FILENAME) # DEFINICIÓN


# --- CONFIGURACIÓN DEL MODELO MLP ---
_MLP_NUM_COLS = ['price_t0_log', 'days_diff']
_MLP_CAT_COLS = ['artist_name', 'pokemon_name', 'rarity', 'set_name', 'types', 'supertype', 'subtypes']
_MLP_INPUT_KEY = 'inputs'
_MLP_OUTPUT_KEY = 'output_0'
_MLP_TARGET_LOG = True
_MLP_DAYS_DIFF = 29.0

# --- CONFIGURACIÓN DE PIPELINES LGBM ---
_LGBM_NUM_FEATURES_INPUT = ['prev_price', 'days_since_prev_snapshot', 'cm_avg1', 'cm_avg7', 'cm_avg30', 'cm_trendPrice']
_LGBM_CAT_FEATURES_INPUT = ['artist_name', 'pokemon_name', 'rarity', 'set_name', 'types', 'supertype', 'subtypes']
_LGBM_ALL_FEATURES_APP = _LGBM_NUM_FEATURES_INPUT + _LGBM_CAT_FEATURES_INPUT
_LGBM_THRESHOLD_COLUMN_APP = 'cm_avg7'
_LGBM_TARGET_IS_LOG_TRANSFORMED = True

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
    # ... (código sin cambios)
    try:
        creds_json = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(creds_json)
        client = bigquery.Client(credentials=creds, project=GCP_PROJECT_ID)
        logger.info("CONNECT_BQ: Conexión OK.")
        return client
    except Exception as e: logger.error(f"CONNECT_BQ: Error: {e}", exc_info=True); st.error(f"Error BQ: {e}."); return None

bq_client = connect_to_bigquery()
if bq_client is None: st.stop()


# --- FUNCIONES DE CARGA DE ARTEFACTOS ---
@st.cache_resource
def load_tf_model_as_layer(model_path):
    # ... (código sin cambios)
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
def load_sklearn_pipeline(file_path, pipeline_name="Sklearn Pipeline"):
    # ... (código sin cambios)
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
def load_joblib_preprocessor(file_path, preprocessor_name="Preprocessor"):
    # ... (código sin cambios)
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

@st.cache_data # Usar cache_data para JSON simple
def load_json_config(file_path, config_name="JSON Config"):
    # ... (código sin cambios)
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
lgbm_threshold_value = lgbm_threshold_config.get("threshold", 30.0) if lgbm_threshold_config else 30.0
if lgbm_threshold_config is None: logger.warning("LGBM_THRESHOLD: Usando umbral por defecto (30.0) porque el archivo de config no cargó.")


# --- Funciones Auxiliares de Datos (Definidas antes de su primer uso) ---
# ... (AQUÍ VAN get_latest_snapshot_info, get_true_base_name, get_card_metadata_with_base_names,
#      fetch_card_data_from_bq, predict_price_with_mlp, predict_price_with_lgbm_pipelines_app)
#     (Por brevedad, no las repito aquí, pero deben estar como en el código completo anterior)
