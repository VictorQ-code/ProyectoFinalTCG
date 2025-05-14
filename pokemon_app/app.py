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
import joblib
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
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_files") # Carpeta principal para modelos

# MLP (TensorFlow SavedModel) - Si decides usarlo en el futuro
TF_SAVED_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "mlp_v1") # Asume que mlp_v1 es una carpeta
OHE_MLP_PKL_FILENAME = "ohe_mlp_cat.pkl" # Para MLP
SCALER_MLP_PKL_FILENAME = "scaler_mlp_num.pkl" # Para MLP
OHE_MLP_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "mlp_v1", OHE_MLP_PKL_FILENAME) # Asumiendo que están dentro de mlp_v1
SCALER_MLP_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "mlp_v1", SCALER_MLP_PKL_FILENAME)

# LightGBM (Pipelines)
LGBM_MODEL_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "lgbm_models") # Subcarpeta para modelos LGBM
PIPE_LOW_PKL_FILENAME = "modelo_pipe_low.pkl"
PIPE_HIGH_PKL_FILENAME = "modelo_pipe_high.pkl" # O "modelo_best_pipe_high.pkl"
THRESHOLD_JSON_FILENAME = "threshold.json"
PIPE_LOW_LGBM_PATH = os.path.join(LGBM_MODEL_DIR, PIPE_LOW_PKL_FILENAME)
PIPE_HIGH_LGBM_PATH = os.path.join(LGBM_MODEL_DIR, PIPE_HIGH_PKL_FILENAME)
THRESHOLD_LGBM_PATH = os.path.join(LGBM_MODEL_DIR, THRESHOLD_JSON_FILENAME)


# --- CONFIGURACIÓN DE FEATURES PARA MODELOS ---
# MLP (Si se usa)
_MLP_NUMERICAL_FEATURES_APP = ['price_t0_log', 'days_diff']
_MLP_CATEGORICAL_FEATURES_APP = ['artist_name', 'pokemon_name', 'rarity', 'set_name', 'types', 'supertype', 'subtypes']
_MLP_MODEL_INPUT_TENSOR_KEY_NAME = 'inputs'
_MLP_MODEL_OUTPUT_TENSOR_KEY_NAME = 'output_0'
_MLP_TARGET_PREDICTED_IS_LOG_TRANSFORMED = True
_MLP_DEFAULT_DAYS_DIFF_FOR_PREDICTION = 29.0

# LightGBM (Actualmente el principal)
_LGBM_NUMERIC_FEATURES_APP = ['prev_price', 'days_since_prev_snapshot', 'cm_avg1', 'cm_avg7', 'cm_avg30', 'cm_trendPrice']
_LGBM_CATEGORICAL_FEATURES_APP = ['artist_name', 'pokemon_name', 'rarity', 'set_name', 'types', 'supertype', 'subtypes']
_LGBM_ALL_FEATURES_APP = _LGBM_NUMERIC_FEATURES_APP + _LGBM_CATEGORICAL_FEATURES_APP
_LGBM_THRESHOLD_COLUMN_APP = 'cm_avg7' # Columna usada para el umbral en predict_mixed
# _LGBM_TARGET_IS_LOG_TRANSFORMED se maneja dentro de la función de predicción por np.expm1

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

# --- FUNCIONES DE CARGA DE MODELO Y PREPROCESADORES ---
# Para MLP (TensorFlow SavedModel)
@st.cache_resource
def load_tf_model_as_layer(model_path):
    # ... (código de la función sin cambios) ...
    saved_model_pb_path = os.path.join(model_path, "saved_model.pb")
    if not os.path.exists(saved_model_pb_path):
        logger.error(f"LOAD_TF_LAYER: 'saved_model.pb' no encontrado en la ruta del modelo: {model_path}")
        st.error(f"Error Crítico: El archivo 'saved_model.pb' no se encuentra en '{model_path}'. "
                 f"Asegúrate de que la carpeta '{os.path.basename(model_path)}' (ej. 'model_files/mlp_v1') "
                 "exista en tu repositorio y contenga la estructura del SavedModel (saved_model.pb, variables/, etc.).")
        return None
    try:
        logger.info(f"LOAD_TF_LAYER: Intentando cargar SavedModel como TFSMLayer desde: {model_path}")
        model_as_layer_obj = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        logger.info(f"LOAD_TF_LAYER: SavedModel cargado exitosamente como TFSMLayer.")
        try: logger.info(f"LOAD_TF_LAYER: Call Signature (info interna): {model_as_layer_obj._call_signature}")
        except AttributeError: logger.warning("LOAD_TF_LAYER: No se pudo acceder a '_call_signature'.")
        logger.info("LOAD_TF_LAYER: La inspección directa de 'structured_outputs' no está disponible en esta versión de TFSMLayer. "
                    f"Se usará la clave de salida configurada: '{_MLP_MODEL_OUTPUT_TENSOR_KEY_NAME}'. " # Usar constante MLP
                    "Verifícala con `saved_model_cli` if there are prediction issues.")
        return model_as_layer_obj
    except Exception as e:
        logger.error(f"LOAD_TF_LAYER: Error crítico al cargar SavedModel como TFSMLayer desde {model_path}: {e}", exc_info=True)
        st.error(f"Error Crítico al Cargar Modelo Local (TF): {e}. Revisa los logs y la configuración del modelo.")
        st.info( "Si el error es sobre 'call_endpoint', verifica el nombre usado al guardar el modelo "
                "(puedes usar `saved_model_cli show --dir ruta/a/model_files --all` en tu terminal local).")
        return None

# Para Preprocesadores y Pipelines LGBM (Joblib)
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
        return 30.0 # Default si no se encuentra
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

# --- Carga de Modelos y Preprocesadores ---
logger.info("APP_INIT: Iniciando carga de artefactos de modelos.")
# MLP (si se usa en el futuro)
# local_tf_model_layer = load_tf_model_as_layer(TF_SAVED_MODEL_PATH)
# ohe_mlp_preprocessor = load_joblib_object(OHE_MLP_PATH, "OHE para MLP")
# scaler_mlp_preprocessor = load_joblib_object(SCALER_MLP_PATH, "Scaler para MLP")

# LightGBM (modelo activo)
pipe_low_lgbm_app = load_joblib_object(PIPE_LOW_LGBM_PATH, "Pipeline LGBM Precios Bajos")
pipe_high_lgbm_app = load_joblib_object(PIPE_HIGH_LGBM_PATH, "Pipeline LGBM Precios Altos")
threshold_lgbm_app = load_threshold_from_json(THRESHOLD_LGBM_PATH)


# --- FUNCIONES UTILITARIAS DE DATOS ---
@st.cache_data(ttl=3600)
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    # ... (código sin cambios) ...
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
    # ... (código sin cambios) ...
    if not isinstance(name_str, str) or supertype != 'Pokémon': return name_str
    for mw_base in multi_word_bases:
        if name_str.startswith(mw_base): return mw_base
    cleaned_name = name_str
    for suffix in suffixes:
        if cleaned_name.endswith(suffix): cleaned_name = cleaned_name[:-len(suffix)].strip()
    return cleaned_name if cleaned_name else name_str

@st.cache_data(ttl=3600)
def get_card_metadata_with_base_names(_client: bigquery.Client) -> pd.DataFrame:
    # Query para metadatos, asegurando que los nombres de columna coincidan con lo que espera la app
    query = f"""
    SELECT
        id,
        name         AS pokemon_name, -- Nombre para display y feature categórica
        supertype,
        subtypes,
        types,
        rarity,
        set_name,
        artist       AS artist_name,  -- Feature categórica
        images_large AS image_url,
        cardmarket_url,
        tcgplayer_url
    FROM `{CARD_METADATA_TABLE}`
    """
    logger.info(f"METADATA_BQ: Ejecutando query para metadatos: {query[:100]}...")
    try:
        df = _client.query(query).to_dataframe()
        if df.empty: logger.warning("METADATA_BQ: DataFrame de metadatos vacío."); st.warning("No se pudo cargar metadatos."); return pd.DataFrame()
        
        # Asegurar que todas las columnas esperadas por el OHE del LGBM (si se toman de aquí) estén
        for col_expected in ['artist_name', 'pokemon_name', 'rarity', 'set_name', 'types', 'supertype', 'subtypes']:
            if col_expected not in df.columns:
                df[col_expected] = 'Unknown_Placeholder' # O un valor por defecto más apropiado
                logger.warning(f"METADATA_BQ: Columna '{col_expected}' no en metadatos, añadida con placeholder.")
        
        df['base_pokemon_name_display'] = df.apply(lambda row: get_true_base_name(row['pokemon_name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES), axis=1)
        logger.info(f"METADATA_BQ: Metadatos cargados y procesados. Total filas: {len(df)}.")
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("METADATA_BQ: Error de 'db-dtypes'.", exc_info=True); st.error("Error de Dependencia: Falta 'db-dtypes'.")
        else: logger.error(f"METADATA_BQ: Error al cargar metadatos de BigQuery: {e}", exc_info=True); st.error(f"Error al cargar metadatos de cartas: {e}.")
        return pd.DataFrame()

# --- FUNCIÓN DE CONSULTA DE DATOS DE PRECIOS Y METADATOS COMBINADOS ---
@st.cache_data(ttl=600)
def fetch_card_data_from_bq(
    _client: bigquery.Client, latest_table_path: str, supertype_ui_filter: str | None,
    sets_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
    sort_direction: str, full_metadata_df_param: pd.DataFrame
) -> pd.DataFrame:
    # ... (Esta función fue movida aquí, asegurar que su contenido sea el correcto de v1.7 / v1.9)
    # ... (Query SQL dentro de esta función DEBE seleccionar TODAS las columnas necesarias para results_df)
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

    # Esta query debe traer TODAS las columnas que usa el modelo LGBM y para display
    query_sql_template = f"""
    SELECT
        meta.id, meta.pokemon_name, meta.supertype, meta.subtypes, meta.types,
        meta.set_name, meta.rarity, meta.artist_name, meta.images_large AS image_url,
        meta.cardmarket_url, meta.tcgplayer_url,
        prices.precio, prices.cm_trendPrice,
        prices.cm_avg1, prices.cm_avg7, prices.cm_avg30,
        prices.fecha AS fecha_snapshot 
    FROM `{CARD_METADATA_TABLE}` AS meta
    LEFT JOIN `{latest_table_path}` AS prices ON meta.id = prices.card_id 
    WHERE meta.id IN UNNEST(@card_ids_param)
    ORDER BY prices.precio {sort_direction} 
    """
    # Nota: Usamos LEFT JOIN desde metadatos a precios. Si una carta no tiene precio en el snapshot,
    # las columnas de precios serán NaN, lo cual es manejado.
    # La tabla de precios 'monthly_*' debe tener una columna 'card_id' que coincida con meta.id.
    # y las columnas: precio (cm_averageSellPrice), cm_trendPrice, cm_avg1, cm_avg7, cm_avg30, fecha.

    query_params_bq = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids_to_query)]
    job_config_bq = bigquery.QueryJobConfig(query_parameters=query_params_bq)
    logger.info(f"FETCH_BQ_DATA: SQL BQ para {len(list_of_card_ids_to_query)} IDs. Orden: {sort_direction}")
    try:
        results_from_bq_df = _client.query(query_sql_template, job_config=job_config_bq).to_dataframe()
        # Convertir columnas de precio a numérico, los errores se convierten en NaN
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
    # ... (código de la función sin cambios estructurales, solo asegurar que los nombres de columna de
    #      _LGBM_NUMERIC_FEATURES_APP y _LGBM_CATEGORICAL_FEATURES_APP coincidan con lo que
    #      card_data_for_prediction (una fila de results_df) realmente contiene)
    logger.info(f"LGBM_PRED_APP: Iniciando predicción para carta ID: {card_data_for_prediction.get('id', 'N/A')}")
    if not pipe_low_lgbm_loaded or not pipe_high_lgbm_loaded or threshold_lgbm_value is None:
        logger.error("LGBM_PRED_APP: Pipelines LGBM o umbral no cargados.")
        st.error("Error Interno: Modelos LGBM o umbral no disponibles.")
        return None
    try:
        input_dict = {}
        current_price_val = card_data_for_prediction.get('precio') # 'precio' es cm_averageSellPrice del snapshot actual
        input_dict['prev_price'] = float(current_price_val) if pd.notna(current_price_val) else 0.0
        input_dict['days_since_prev_snapshot'] = 30.0 # Horizonte de predicción

        for col_name in ['cm_avg1', 'cm_avg7', 'cm_avg30', 'cm_trendPrice']:
            val = card_data_for_prediction.get(col_name)
            input_dict[col_name] = float(val) if pd.notna(val) else 0.0
            if pd.isna(val): logger.warning(f"LGBM_PRED_APP: Feature numérica '{col_name}' es NaN. Usando 0.0.")
        
        # Categóricas (deben coincidir con _LGBM_CATEGORICAL_FEATURES_APP)
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
        
        # El ColumnTransformer dentro del pipeline se encargará de seleccionar y ordenar
        X_new_predict_for_pipe = X_new_predict_df # Pasar el DataFrame completo

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
all_card_metadata_df = get_card_metadata_with_base_names(bq_client) # Contiene 'pokemon_name', 'artist_name', etc.
if not LATEST_SNAPSHOT_TABLE or all_card_metadata_df.empty:
    logger.critical("APP_INIT_FAIL: Datos esenciales de BigQuery no cargados.")
    st.error("Error Crítico: No se pudieron cargar los datos esenciales de BigQuery.")
    st.stop()
logger.info("APP_INIT: Datos iniciales de BigQuery cargados OK.")


# --- Sidebar y Filtros ---
st.sidebar.header("Filtros y Opciones")
options_df_for_filters = all_card_metadata_df.copy() # Usar metadatos para los filtros
supertype_options_list = sorted(options_df_for_filters['supertype'].dropna().unique().tolist())
select_supertype_options = ["Todos"] + supertype_options_list if supertype_options_list else ["Todos"]
selected_supertype = st.sidebar.selectbox("Categoría:", select_supertype_options, index=0, key="sb_supertype_filter_v3")
if selected_supertype != "Todos": options_df_for_filters = options_df_for_filters[options_df_for_filters['supertype'] == selected_supertype]
set_options_list = sorted(options_df_for_filters['set_name'].dropna().unique().tolist())
selected_sets = st.sidebar.multiselect("Set(s):", set_options_list, key="ms_sets_filter_v3")
if selected_sets: options_df_for_filters = options_df_for_filters[options_df_for_filters['set_name'].isin(selected_sets)]
name_label = "Nombre de Carta:"; name_col_for_options = 'pokemon_name' # Usar 'pokemon_name'
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

# --- Carga de results_df (Datos para mostrar y predicción) ---
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
             display_cards_df = special_illustration_rares.sample(n=num_cards_to_show, random_state=1).reset_index(drop=True) # random_state para consistencia en dev
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
    if len(results_df) > MAX_ROWS_NO_FILTER and is_initial_unfiltered_load : # Solo limitar si es carga inicial y hay muchos
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
        st.write("Haz clic en una fila para ver sus detalles:")
        grid_response = AgGrid( final_display_df_aggrid, gridOptions=gridOptions, height=500, width='100%', data_return_mode=DataReturnMode.AS_INPUT, update_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=False, allow_unsafe_jscode=True, key='pokemon_aggrid_main_vFINAL_LGBM')
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
        # ... (resto de la asignación de variables de detalle)
        card_image_url_render = card_to_display_in_detail_section.get('image_url', None)
        card_supertype_render = card_to_display_in_detail_section.get('supertype', "N/A")
        card_rarity_render = card_to_display_in_detail_section.get('rarity', "N/A")
        card_artist_render = card_to_display_in_detail_section.get('artist_name', None)
        card_price_actual_render = card_to_display_in_detail_section.get('precio', None)
        cardmarket_url_render = card_to_display_in_detail_section.get('cardmarket_url', None)
        tcgplayer_url_render = card_to_display_in_detail_section.get('tcgplayer_url', None)

        col_img, col_info = st.columns([1, 2])
        with col_img: # ... (código de imagen y links sin cambios)
            if pd.notna(card_image_url_render): st.image(card_image_url_render, caption=f"{card_name_render} ({card_set_render})", width=300)
            else: st.warning("Imagen no disponible.")
            links_html = []
            if pd.notna(cardmarket_url_render) and cardmarket_url_render.startswith("http"): links_html.append(f"<a href='{cardmarket_url_render}' target='_blank' style='...'>Cardmarket</a>")
            if pd.notna(tcgplayer_url_render) and tcgplayer_url_render.startswith("http"): links_html.append(f"<a href='{tcgplayer_url_render}' target='_blank' style='...'>TCGplayer</a>")
            if links_html: st.markdown(" ".join(links_html), unsafe_allow_html=True)
            else: st.caption("Links no disponibles.")
        with col_info: # ... (código de información de carta sin cambios)
            st.subheader(f"{card_name_render}")
            st.markdown(f"**ID:** `{card_to_display_in_detail_section.get('id')}`"); st.markdown(f"**Categoría:** {card_supertype_render}"); st.markdown(f"**Set:** {card_set_render}"); st.markdown(f"**Rareza:** {card_rarity_render}")
            if pd.notna(card_artist_render): st.markdown(f"**Artista:** {card_artist_render}")
            if pd.notna(card_price_actual_render): st.metric(label="Precio Actual (Trend €)", value=f"€{card_price_actual_render:.2f}")
            else: st.markdown("**Precio Actual (Trend €):** N/A")
            st.markdown("---"); st.subheader("Predicción de Precio (Modelo LGBM Estimado)")
            
            if pipe_low_lgbm_app and pipe_high_lgbm_app and threshold_lgbm_app is not None:
                # Solo mostrar botón si tenemos precio actual Y cm_avg7 (necesario para el umbral)
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
        if not all_card_metadata_df[all_card_metadata_df['rarity'] == FEATURED_RARITY].empty: pass
        else: st.info("No se encontraron cartas destacadas ni otros resultados iniciales.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Pokémon TCG Explorer v1.10 | LGBM")
