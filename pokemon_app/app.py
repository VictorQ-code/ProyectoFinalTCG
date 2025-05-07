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

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Pokémon TCG Explorer")
logging.basicConfig(
    level=logging.INFO, # Puedes cambiar a logging.DEBUG para más detalle si depuras
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

# --- RUTAS Y NOMBRES DE ARCHIVOS DEL MODELO LOCAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_files")

TF_SAVED_MODEL_PATH = MODEL_ARTIFACTS_DIR
OHE_PKL_FILENAME = "ohe_mlp_cat.pkl"
SCALER_PKL_FILENAME = "scaler_mlp_num.pkl"
OHE_PATH = os.path.join(MODEL_ARTIFACTS_DIR, OHE_PKL_FILENAME)
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, SCALER_PKL_FILENAME)

# --- CONFIGURACIÓN DEL MODELO LOCAL ---
_NUMERICAL_COLS_FOR_MODEL_PREPROCESSING = ['price_t0_log', 'days_diff']

_CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING = [
    'artist_name', 'pokemon_name', 'rarity',
    'set_name', 'types', 'supertype', 'subtypes'
]

_MODEL_INPUT_TENSOR_KEY_NAME = 'inputs'
_MODEL_OUTPUT_TENSOR_KEY_NAME = 'output_0'

_TARGET_PREDICTED_IS_LOG_TRANSFORMED = True
DEFAULT_DAYS_DIFF_FOR_PREDICTION = 29.0

# --- CONFIGURACIÓN PARA CARTAS DESTACADAS ---
FEATURED_RARITY = 'Special Illustration Rare' # Raleza a destacar
NUM_FEATURED_CARDS_TO_DISPLAY = 5            # Cuántas mostrar en la sección

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

# --- FUNCIONES DE CARGA DE MODELO Y PREPROCESADORES ---
@st.cache_resource
def load_local_tf_model_as_layer(model_path):
    saved_model_pb_path = os.path.join(model_path, "saved_model.pb")
    if not os.path.exists(saved_model_pb_path):
        logger.error(f"LOAD_TF_LAYER: 'saved_model.pb' no encontrado en la ruta del modelo: {model_path}")
        st.error(f"Error Crítico: El archivo 'saved_model.pb' no se encuentra en '{model_path}'. "
                 f"Asegúrate de que la carpeta '{os.path.basename(model_path)}' (ej. 'model_files') "
                 "exista en tu repositorio y contenga la estructura del SavedModel (saved_model.pb, variables/, etc.).")
        return None
    try:
        logger.info(f"LOAD_TF_LAYER: Intentando cargar SavedModel como TFSMLayer desde: {model_path}")
        model_as_layer_obj = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        logger.info(f"LOAD_TF_LAYER: SavedModel cargado exitosamente como TFSMLayer.")
        try: logger.info(f"LOAD_TF_LAYER: Call Signature (info interna): {model_as_layer_obj._call_signature}")
        except AttributeError: logger.warning("LOAD_TF_LAYER: No se pudo acceder a '_call_signature'.")
        logger.info("LOAD_TF_LAYER: La inspección directa de 'structured_outputs' no está disponible en esta versión de TFSMLayer. "
                    f"Se usará la clave de salida configurada: '{_MODEL_OUTPUT_TENSOR_KEY_NAME}'. "
                    "Verifícala con `saved_model_cli` if there are prediction issues.")
        return model_as_layer_obj
    except Exception as e:
        logger.error(f"LOAD_TF_LAYER: Error crítico al cargar SavedModel como TFSMLayer desde {model_path}: {e}", exc_info=True)
        st.error(f"Error Crítico al Cargar Modelo Local: {e}. Revisa los logs y la configuración del modelo.")
        st.info( "Si el error es sobre 'call_endpoint', verifica el nombre usado al guardar el modelo "
                "(puedes usar `saved_model_cli show --dir ruta/a/model_files --all` en tu terminal local).")
        return None

@st.cache_resource
def load_local_preprocessor(file_path, preprocessor_name="Preprocessor"):
    if not os.path.exists(file_path):
        logger.error(f"LOAD_PREPROC: Archivo '{preprocessor_name}' no encontrado en: {file_path}")
        st.error(f"Error Crítico: El archivo preprocesador '{preprocessor_name}' no se encuentra en '{file_path}'. "
                 f"Asegúrate de que esté en la carpeta '{os.path.basename(os.path.dirname(file_path))}'.")
        return None
    try:
        preprocessor = joblib.load(file_path)
        logger.info(f"LOAD_PREPROC: {preprocessor_name} cargado exitosamente desde: {file_path}")
        return preprocessor
    except Exception as e:
        logger.error(f"LOAD_PREPROC: Error crítico al cargar {preprocessor_name} desde {file_path}: {e}", exc_info=True)
        st.error(f"Error Crítico al Cargar Preprocesador '{preprocesador_name}': {e}")
        return None

# --- Carga de Modelo y Preprocesadores al inicio de la app ---
logger.info("APP_INIT: Iniciando carga de artefactos del modelo local.")
local_tf_model_layer = load_local_tf_model_as_layer(TF_SAVED_MODEL_PATH)
ohe_local_preprocessor = load_local_preprocessor(OHE_PATH, "OneHotEncoder")
scaler_local_preprocessor = load_local_preprocessor(SCALER_PATH, "ScalerNumérico")


# --- Funciones Auxiliares (código de BigQuery, etc.) ---
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

# --- MOVER DEFINICIÓN DE fetch_card_data_from_bq AQUÍ ---
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
        actual_name_col_to_filter_on = 'base_pokemon_name' if supertype_ui_filter == 'Pokémon' else 'name'
        if actual_name_col_to_filter_on in ids_to_query_df.columns: ids_to_query_df = ids_to_query_df[ids_to_query_df[actual_name_col_to_filter_on].isin(names_ui_filter)]
    if ids_to_query_df.empty: logger.info("FETCH_BQ_DATA: No hay IDs que coincidan."); return pd.DataFrame()
    list_of_card_ids_to_query = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids_to_query: logger.info("FETCH_BQ_DATA: Lista IDs vacía."); return pd.DataFrame()
    query_sql_template = f"""
    SELECT
        meta.id, meta.name AS pokemon_name, meta.supertype,
        meta.subtypes, meta.types, -- ASEGURADAS
        meta.set_name, meta.rarity, meta.artist, meta.images_large AS image_url,
        meta.cardmarket_url, meta.tcgplayer_url, prices.cm_trendPrice AS price
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
        if 'price' in results_from_bq_df.columns: results_from_bq_df['price'] = pd.to_numeric(results_from_bq_df['price'], errors='coerce')
        logger.info(f"FETCH_BQ_DATA: Consulta a BQ OK. Filas: {len(results_from_bq_df)}.")
        return results_from_bq_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("FETCH_BQ_DATA_FAIL: Error de 'db-dtypes'.", exc_info=True); st.error("Error de Dependencia: Falta 'db-dtypes'.")
        else: logger.error(f"FETCH_BQ_DATA_FAIL: Error BQ: {e}", exc_info=True); st.error(f"Error al obtener datos de cartas: {e}.")
        return pd.DataFrame()


# --- FUNCIÓN DE PREDICCIÓN CON MODELO LOCAL (MLP/TFSMLayer) ---
def predict_price_with_local_tf_layer(
    model_layer: tf.keras.layers.TFSMLayer,
    ohe: typing.Any, # OneHotEncoder
    scaler: typing.Any, # StandardScaler
    card_data_series: pd.Series
) -> float | None:
    logger.info(f"PREDICT_LOCAL_ENTRY: Iniciando predicción para carta ID: {card_data_series.get('id', 'N/A')}")
    if not model_layer or not ohe or not scaler:
        logger.error("PREDICT_LOCAL_FAIL: Modelo TFSMLayer o preprocesadores no disponibles.")
        st.error("Error Interno: Componentes del modelo local no disponibles para predicción.")
        return None
    try:
        # --- PASO 1: Preparar DataFrame de entrada para preprocesamiento ---
        data_for_preprocessing_df_dict = {}
        current_price = card_data_series.get('price')
        if pd.notna(current_price) and current_price > 0: data_for_preprocessing_df_dict['price_t0_log'] = np.log1p(current_price)
        else: data_for_preprocessing_df_dict['price_t0_log'] = np.log1p(0); logger.warning(f"PREDICT_LOCAL_MAP: Precio actual no válido ('{current_price}') para 'price_t0_log', usando np.log1p(0).")
        data_for_preprocessing_df_dict['days_diff'] = float(DEFAULT_DAYS_DIFF_FOR_PREDICTION)

        data_for_preprocessing_df_dict['artist_name'] = str(card_data_series.get('artist', 'Unknown_Artist'))
        data_for_preprocessing_df_dict['pokemon_name'] = str(card_data_series.get('pokemon_name', 'Unknown_Pokemon'))
        data_for_preprocessing_df_dict['rarity'] = str(card_data_series.get('rarity', 'Unknown_Rarity'))
        data_for_preprocessing_df_dict['set_name'] = str(card_data_series.get('set_name', 'Unknown_Set'))
        data_for_preprocessing_df_dict['supertype'] = str(card_data_series.get('supertype', 'Unknown_Supertype'))

        types_val = card_data_series.get('types')
        if isinstance(types_val, list) and types_val: data_for_preprocessing_df_dict['types'] = str(types_val[0]) if pd.notna(types_val[0]) else 'Unknown_Type'
        elif pd.notna(types_val): data_for_preprocessing_df_dict['types'] = str(types_val)
        else: data_for_preprocessing_df_dict['types'] = 'Unknown_Type'

        subtypes_val = card_data_series.get('subtypes')
        if isinstance(subtypes_val, list) and subtypes_val:
            cleaned_subtypes = [str(s) for s in subtypes_val if pd.notna(s)]
            data_for_preprocessing_df_dict['subtypes'] = ', '.join(sorted(list(set(cleaned_subtypes)))) if cleaned_subtypes else 'None'
        elif pd.notna(subtypes_val): data_for_preprocessing_df_dict['subtypes'] = str(subtypes_val)
        else: data_for_preprocessing_df_dict['subtypes'] = 'None'

        current_input_df_for_preprocessing = pd.DataFrame([data_for_preprocessing_df_dict])
        ordered_cols_for_df = _NUMERICAL_COLS_FOR_MODEL_PREPROCESSING + _CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING
        try: current_input_df_for_preprocessing = current_input_df_for_preprocessing[ordered_cols_for_df]
        except KeyError as e_key:
            missing_keys_in_df = [col for col in ordered_cols_for_df if col not in current_input_df_for_preprocessing.columns]
            logger.error(f"PREDICT_LOCAL_ORDER_FAIL: Error al ordenar columnas. Faltan: {missing_keys_in_df}. Error: {e_key}")
            st.error(f"Error Interno: No se pudieron ordenar las características para el modelo ({', '.join(missing_keys_in_df)}).")
            return None
        logger.info(f"PREDICT_LOCAL_PREPROC_DF: DataFrame (1 fila): {current_input_df_for_preprocessing.shape}. Cols: {list(current_input_df_for_preprocessing.columns)}")
        logger.debug(f"PREDICT_LOCAL_PREPROC_DF_VALUES: {current_input_df_for_preprocessing.iloc[0].to_dict()}")

        # --- PASO 2: Aplicar preprocesamiento ---
        processed_feature_parts = []
        if _NUMERICAL_COLS_FOR_MODEL_PREPROCESSING:
            num_df_slice = current_input_df_for_preprocessing[_NUMERICAL_COLS_FOR_MODEL_PREPROCESSING]
            if num_df_slice.isnull().values.any():
                logger.warning(f"PREDICT_LOCAL_SCALE: NaNs ANTES de escalar: {num_df_slice.isnull().sum().to_dict()}. Imputando con 0.")
                num_df_slice = num_df_slice.fillna(0)
            numerical_features_scaled_array = scaler.transform(num_df_slice)
            processed_feature_parts.append(numerical_features_scaled_array)
            logger.info(f"PREDICT_LOCAL_SCALE: Numéricas escaladas (shape): {numerical_features_scaled_array.shape}")

        if _CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING:
            cat_df_slice = current_input_df_for_preprocessing[_CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING].astype(str)
            categorical_features_encoded_dense_array = ohe.transform(cat_df_slice)
            processed_feature_parts.append(categorical_features_encoded_dense_array)
            logger.info(f"PREDICT_LOCAL_OHE: Categóricas codificadas (shape): {categorical_features_encoded_dense_array.shape}")

        if not processed_feature_parts:
            logger.error("PREDICT_LOCAL_COMBINE_FAIL: No se procesaron características.")
            st.error("Error Interno: No se pudieron procesar las características para el modelo.")
            return None

        # --- PASO 3: Combinar ---
        final_input_array_for_model = np.concatenate(processed_feature_parts, axis=1)
        logger.info(f"PREDICT_LOCAL_COMBINE: Array final para modelo (shape): {final_input_array_for_model.shape}")

        EXPECTED_NUM_FEATURES = 4865 # Confirmado por saved_model_cli
        if final_input_array_for_model.shape[1] != EXPECTED_NUM_FEATURES:
            logger.error(f"¡¡¡DESAJUSTE DE SHAPE EN LA ENTRADA DEL MODELO!!!")
            logger.error(f"    Modelo espera: {EXPECTED_NUM_FEATURES} características.")
            logger.error(f"    Array preprocesado tiene: {final_input_array_for_model.shape[1]} características.")
            if 'numerical_features_scaled_array' in locals(): logger.debug(f"    Shape numéricas escaladas: {numerical_features_scaled_array.shape}")
            if 'categorical_features_encoded_dense_array' in locals(): logger.debug(f"    Shape categóricas OHE: {categorical_features_encoded_dense_array.shape}")
            st.error(f"Error Crítico de Preprocesamiento: Discrepancia en el número de características. Esperadas: {EXPECTED_NUM_FEATURES}, Generadas: {final_input_array_for_model.shape[1]}.")
            return None

        # --- PASO 4: Predicción ---
        final_input_tensor_for_model = tf.convert_to_tensor(final_input_array_for_model, dtype=tf.float32)
        logger.info(f"PREDICT_LOCAL_TENSOR: Tensor de entrada para TFSMLayer (shape): {final_input_tensor_for_model.shape}, dtype: {final_input_tensor_for_model.dtype}")
        if _MODEL_INPUT_TENSOR_KEY_NAME: # Que es 'inputs'
            model_input_feed_dict = {_MODEL_INPUT_TENSOR_KEY_NAME: final_input_tensor_for_model}
            logger.info(f"PREDICT_LOCAL_CALL: Llamando a TFSMLayer con diccionario desempaquetado: Clave='{_MODEL_INPUT_TENSOR_KEY_NAME}'")
            raw_prediction_output = model_layer(**model_input_feed_dict)
        else:
            logger.info("PREDICT_LOCAL_CALL: Llamando a TFSMLayer con tensor de entrada directo (no usado con firma 'inputs').")
            raw_prediction_output = model_layer(final_input_tensor_for_model)

        logger.info(f"PREDICT_LOCAL_RAW_OUT: Salida cruda de TFSMLayer (tipo {type(raw_prediction_output)}): {raw_prediction_output}")

        if not isinstance(raw_prediction_output, dict):
            if tf.is_tensor(raw_prediction_output): predicted_value_tensor = raw_prediction_output
            else:
                logger.error(f"PREDICT_LOCAL_EXTRACT_FAIL: Salida de TFSMLayer no es ni dict ni tensor, es {type(raw_prediction_output)}.")
                st.error("Error Interno: Formato de salida del modelo local inesperado.")
                return None
        elif not raw_prediction_output: logger.error("PREDICT_LOCAL_EXTRACT_FAIL: Dict de salida vacío."); st.error("Error Interno: El modelo local devolvió salida vacía.")
        elif _MODEL_OUTPUT_TENSOR_KEY_NAME not in raw_prediction_output:
            available_keys = list(raw_prediction_output.keys())
            logger.error(f"PREDICT_LOCAL_EXTRACT_FAIL: Clave '{_MODEL_OUTPUT_TENSOR_KEY_NAME}' NO en dict. Claves: {available_keys}")
            st.error(f"Error Interno: Clave de salida ('{_MODEL_OUTPUT_TENSOR_KEY_NAME}') no encontrada. Disponibles: {available_keys}")
            return None
        else: predicted_value_tensor = raw_prediction_output[_MODEL_OUTPUT_TENSOR_KEY_NAME]
        logger.info(f"PREDICT_LOCAL_EXTRACT: Tensor (clave '{_MODEL_OUTPUT_TENSOR_KEY_NAME}' si dict). Shape: {predicted_value_tensor.shape}")

        if predicted_value_tensor.shape == (1, 1) or predicted_value_tensor.shape == (1,):
            predicted_value_numeric = predicted_value_tensor.numpy()[0][0] if len(predicted_value_tensor.shape) == 2 else predicted_value_tensor.numpy()[0]
        else:
            logger.error(f"PREDICT_LOCAL_NUMERIC_FAIL: Shape tensor predicción: {predicted_value_tensor.shape}. Esperada (1,1) o (1,).")
            st.error("Error Interno: Formato del valor de predicción inesperado.")
            return None
        logger.info(f"PREDICT_LOCAL_NUMERIC: Valor numérico extraído: {predicted_value_numeric}")

        # --- PASO 5: Postprocesar ---
        if _TARGET_PREDICTED_IS_LOG_TRANSFORMED: final_predicted_price = np.expm1(predicted_value_numeric)
        else: final_predicted_price = predicted_value_numeric
        logger.info(f"PREDICT_LOCAL_POSTPROC: Predicción final: {final_predicted_price}")
        return float(final_predicted_price)

    except Exception as e:
        logger.error(f"PREDICT_LOCAL_EXCEPTION: Excepción en predicción: {e}", exc_info=True)
        st.error(f"Error Crítico Durante la Predicción Local: {e}")
        import traceback
        st.text_area("Stack Trace (Predicción Local):", traceback.format_exc(), height=200)
        return None


# --- Carga de Datos Inicial de BigQuery ---
logger.info("APP_INIT: Cargando datos iniciales de BigQuery.")
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
# all_card_metadata_df es necesario para los filtros y para la sección destacada
all_card_metadata_df = get_card_metadata_with_base_names(bq_client)

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

# --- Mover la carga de results_df aquí para que SIEMPRE esté definido DESPUÉS de los inputs de la sidebar ---
# results_df contiene los datos fetcheados basados en los filtros de la sidebar
logger.info("MAIN_APP: Fetcheando resultados principales de BigQuery (basado en filtros de sidebar).")
results_df = fetch_card_data_from_bq(
    bq_client, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets,
    selected_names_to_filter, selected_rarities, sort_sql, all_card_metadata_df # Pasamos metadatos para el pre-filtro
)
logger.info(f"MAIN_APP: 'results_df' cargado con {len(results_df)} filas (reflejando filtros).")


# --- Inicializar session_state key para la carta seleccionada ---
if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
    logger.info("SESSION_STATE_INIT: 'selected_card_id_from_grid' inicializado a None.")
logger.info(f"DETAIL_VIEW_ENTRY (pre-render): ID en session_state: {st.session_state.get('selected_card_id_from_grid')}")


# --- Lógica para establecer la carta seleccionada al inicio si no hay nada ---
# Esto ocurre en la PRIMERA carga o si session_state se resetea y no hay AgGrid seleccionado.
# Queremos seleccionar una carta *aleatoria con precio* de results_df si results_df no está vacío.
# Esta lógica DEBE estar DESPUÉS de que results_df se ha cargado basado en los filtros actuales.
if st.session_state.selected_card_id_from_grid is None and not results_df.empty:
    # Filtrar cartas con precio actual para seleccionar un fallback que permita predecir
    cards_with_price = results_df[pd.notna(results_df['price'])]
    if not cards_with_price.empty:
        # Seleccionar una carta aleatoria de las que tienen precio
        random_card_row = cards_with_price.sample(1).iloc[0]
        random_card_id = random_card_row.get('id')
        if random_card_id and pd.notna(random_card_id):
            st.session_state.selected_card_id_from_grid = random_card_id
            logger.info(f"FALLBACK_SELECT: Seleccionando carta aleatoria con precio como fallback: '{random_card_id}'.")
            # No st.rerun() aquí, la próxima ejecución ya usará este ID
    else:
        logger.warning("FALLBACK_SELECT: No se encontraron cartas con precio en los resultados filtrados para seleccionar un fallback.")


# --- Determinar si estamos en la carga inicial sin filtros ---
is_initial_unfiltered_load = (not selected_sets and not selected_names_to_filter and not selected_rarities and (selected_supertype == "Todos" or not selected_supertype))


# --- SECCIÓN PRINCIPAL DE CONTENIDO ---
st.title("Explorador de Cartas Pokémon TCG")

# --- Mostrar Cartas Destacadas O Tabla de Resultados ---
if is_initial_unfiltered_load and not all_card_metadata_df.empty:
    # --- SECCIÓN: Cartas Destacadas (solo imágenes y set) ---
    st.header("Cartas Destacadas")

    special_illustration_rares = all_card_metadata_df[
        all_card_metadata_df['rarity'] == FEATURED_RARITY
    ].copy()

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
                     card_id_featured = card.get('id')
                     card_name_featured = card.get('name', 'N/A')
                     card_set_featured = card.get('set_name', 'N/A')
                     image_url_featured = card.get('images_large')

                     # Mostrar la imagen con st.image (NO es clicable)
                     if pd.notna(image_url_featured) and isinstance(image_url_featured, str):
                         st.image(image_url_featured, width=150, caption=card_set_featured)
                     else:
                         st.warning("Imagen no disponible")
                         st.caption(f"{card_name_featured} ({card_set_featured})")

             # Si mostramos destacadas, ponemos un separador antes de los detalles
             st.markdown("---")

        elif special_illustration_rares.empty:
             logger.info(f"FEATURED_CARDS: No se encontraron cartas con rareza '{FEATURED_RARITY}'.")

    # Si es carga inicial SIN filtros PERO no se mostraron destacadas
    # Mostrar un mensaje antes del detalle si no hay destacadas
    # Si hay destacadas, ya se mostraron y no necesitamos mensaje extra aquí.
    # Este mensaje solo se muestra si special_illustration_rares está vacío Y results_df está vacío
    # y es la carga inicial sin filtros.
    if special_illustration_rares.empty and results_df.empty and is_initial_unfiltered_load and bq_client and LATEST_SNAPSHOT_TABLE:
         st.info("No se encontraron cartas con la rareza destacada o con precio en la base de datos actual.")
    # La tabla NO se muestra en este bloque (is_initial_unfiltered_load es True)


# Mostrar la tabla SOLO si NO es carga inicial sin filtros (es decir, se aplicaron filtros)
elif not is_initial_unfiltered_load: # Tabla visible solo al aplicar filtros
    # --- Área Principal: Visualización de Resultados (AgGrid) ---
    st.header("Resultados de Cartas")
    # st.session_state.selected_card_id_from_grid ya se inicializó y se estableció fallback arriba
    logger.info(f"AGGRID_RENDERING: ID en session_state ANTES de AgGrid: {st.session_state.get('selected_card_id_from_grid')}")
    results_df_for_aggrid_display = results_df # Usar el DF ya cargado
    if len(results_df) > MAX_ROWS_NO_FILTER: # Solo mostramos mensaje de limitación si hay muchos resultados
        logger.info(f"AGGRID_RENDERING: Limitando display a {MAX_ROWS_NO_FILTER} filas de {len(results_df)}.")
        st.info(f"Mostrando los primeros {MAX_ROWS_NO_FILTER} de {len(results_df)} resultados. Aplica filtros.")
        results_df_for_aggrid_display = results_df.head(MAX_ROWS_NO_FILTER)
    grid_response = None
    if not results_df_for_aggrid_display.empty:
        display_columns_mapping = {'id': 'ID', 'pokemon_name': 'Nombre Carta', 'supertype': 'Categoría', 'set_name': 'Set', 'rarity': 'Rareza', 'artist': 'Artista', 'price': 'Precio (Trend €)'}
        cols_in_df_for_display = [col for col in display_columns_mapping.keys() if col in results_df_for_aggrid_display.columns]
        final_display_df_aggrid = results_df_for_aggrid_display[cols_in_df_for_display].copy()
        final_display_df_aggrid.rename(columns=display_columns_mapping, inplace=True)
        price_display_col_name_in_aggrid = display_columns_mapping.get('price')
        if price_display_col_name_in_aggrid and price_display_col_name_in_aggrid in final_display_df_aggrid.columns:
             final_display_df_aggrid[price_display_col_name_in_aggrid] = final_display_df_aggrid[price_display_col_name_in_aggrid].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")
        gb = GridOptionsBuilder.from_dataframe(final_display_df_aggrid)
        gb.configure_selection(selection_mode='single', use_checkbox=False); gb.configure_grid_options(domLayout='normal')
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
        gridOptions = gb.build()
        st.write("Haz clic en una fila de la tabla para ver sus detalles y opciones de predicción:")
        grid_response = AgGrid( final_display_df_aggrid, gridOptions=gridOptions, height=500, width='100%', data_return_mode=DataReturnMode.AS_INPUT, update_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=False, allow_unsafe_jscode=True, key='pokemon_aggrid_main_display_vFINAL')
    else: logger.info("AGGRID_RENDERING: No hay datos para mostrar en AgGrid.")

    # Lógica de Manejo de Clic en AgGrid (si la tabla se mostró)
    if grid_response: # grid_response solo se define si AgGrid se mostró
        logger.debug(f"AGGRID_HANDLER: Procesando grid_response. Tipo de selected_rows: {type(grid_response.get('selected_rows'))}")
        newly_selected_id_from_grid_click = None; selected_rows_data_from_grid = grid_response.get('selected_rows')
        if isinstance(selected_rows_data_from_grid, pd.DataFrame) and not selected_rows_data_from_grid.empty:
            try: # <-- CORRECCIÓN DE SINTAXIS try:
                first_selected_row_as_series = selected_rows_data_from_grid.iloc[0]
                if 'ID' in first_selected_row_as_series: newly_selected_id_from_grid_click = selected_rows_data_from_grid.iloc[0]['ID']
            except Exception as e_aggrid_df: logger.error(f"AGGRID_HANDLER_DF: Error: {e_aggrid_df}", exc_info=True); newly_selected_id_from_grid_click = None # Asegurar None
        elif isinstance(selected_rows_data_from_grid, list) and selected_rows_data_from_grid:
            try: # <-- CORRECCIÓN DE SINTAXIS try:
                if isinstance(selected_rows_data_from_grid[0], dict): newly_selected_id_from_grid_click = selected_rows_data_from_grid[0].get('ID')
            except Exception as e_aggrid_list: logger.error(f"AGGRID_HANDLER_LIST: Error: {e_aggrid_list}", exc_info=True); newly_selected_id_from_grid_click = None # Asegurar None

        current_id_in_session = st.session_state.get('selected_card_id_from_grid')
        if newly_selected_id_from_grid_click is not None and newly_selected_id_from_grid_click != current_id_in_session:
            logger.info(f"AGGRID_HANDLER_STATE_CHANGE: CAMBIO DE SELECCIÓN. Anterior: '{current_id_in_session}', Nuevo: '{newly_selected_id_from_grid_click}'. RE-EJECUTANDO.")
            st.session_state.selected_card_id_from_grid = newly_selected_id_from_grid_click
            st.rerun()
    # else: logger.debug("AgGrid section was displayed but grid_response is None.")


# --- Sección de Detalle de Carta Seleccionada y Predicción ---
# Esta sección solo se intenta mostrar si hay una carta seleccionada en session_state.
if st.session_state.selected_card_id_from_grid is not None:
    st.divider(); st.header("Detalle de Carta Seleccionada")
    card_to_display_in_detail_section = None
    id_for_detail_view_from_session = st.session_state.get('selected_card_id_from_grid')

    # Buscar la carta seleccionada en results_df. Si no está en results_df, buscar en all_card_metadata_df.
    # Usamos all_card_metadata_df como fallback para poder mostrar detalles básicos incluso si no hay precio fetchado.
    # source_df_for_details = results_df if (not results_df.empty and id_for_detail_view_from_session in results_df['id'].values) else all_card_metadata_df
    # Simplificamos la fuente, primero buscar en results_df, si no está, intentar en metadatos
    source_df_for_details_primary = results_df
    source_df_for_details_secondary = all_card_metadata_df


    if not source_df_for_details_primary.empty and id_for_detail_view_from_session in source_df_for_details_primary['id'].values:
         card_to_display_in_detail_section = source_df_for_details_primary[source_df_for_details_primary['id'] == id_for_detail_view_from_session].iloc[0]
         logger.info(f"DETAIL_VIEW_FOUND: Carta '{id_for_detail_view_from_session}' encontrada en results_df.")
    elif not source_df_for_details_secondary.empty and id_for_detail_view_from_session in source_df_for_details_secondary['id'].values:
         # Si no está en results_df (ej. seleccionó destacada y aplicó filtro), buscamos en metadatos
         card_to_display_in_detail_section = source_df_for_details_secondary[source_df_for_details_secondary['id'] == id_for_detail_view_from_session].iloc[0]
         logger.info(f"DETAIL_VIEW_FOUND: Carta '{id_for_detail_view_from_session}' encontrada en all_card_metadata_df.")
    else:
         # Si la carta seleccionada no se encuentra en ninguna de las fuentes
         logger.warning(f"DETAIL_VIEW_NOT_FOUND: ID '{id_for_detail_view_from_session}' NO ENCONTRADO en ninguna fuente de detalles. Resetting selection.")
         st.session_state.selected_card_id_from_grid = None # Resetear selección si no se encuentra
         st.rerun() # Re-ejecutar para limpiar la sección de detalle si la carta no se encuentra


    # Renderizar detalles si tenemos una carta
    if card_to_display_in_detail_section is not None and isinstance(card_to_display_in_detail_section, pd.Series) and not card_to_display_in_detail_section.empty:
        card_id_render = card_to_display_in_detail_section.get('id', "N/A")
        card_name_render = card_to_display_in_detail_section.get('pokemon_name', "N/A")
        card_set_render = card_to_display_in_detail_section.get('set_name', "N/A")
        card_image_url_render = card_to_display_in_detail_section.get('image_url', None)
        card_supertype_render = card_to_display_in_detail_section.get('supertype', "N/A")
        card_rarity_render = card_to_display_in_detail_section.get('rarity', "N/A")
        card_artist_render = card_to_display_in_detail_section.get('artist', None)
        # El precio solo se mostrará si la carta encontrada proviene de results_df
        card_price_actual_render = card_to_display_in_detail_section.get('price', None) # Precio solo está en results_df si la carta vino de ahí
        if pd.isna(card_price_actual_render) and id_for_detail_view_from_session and not results_df.empty and id_for_detail_view_from_session in results_df['id'].values:
             # Si el precio es NaN, pero la carta SÍ está en results_df, quizás el precio es realmente NaN.
             # Opcional: re-buscar solo el precio si está en results_df pero vino de metadatos.
             price_check = results_df[results_df['id'] == id_for_detail_view_from_session]['price'].iloc[0]
             if pd.notna(price_check):
                  card_price_actual_render = price_check
                  logger.warning(f"DETAIL_VIEW: Precio NaN en fuente principal para {id_for_detail_view_from_session}, encontrado en results_df: {card_price_actual_render}")


        cardmarket_url_render = card_to_display_in_detail_section.get('cardmarket_url', None)
        tcgplayer_url_render = card_to_display_in_detail_section.get('tcgplayer_url', None)
        col_img, col_info = st.columns([1, 2])
        with col_img:
            if pd.notna(card_image_url_render): st.image(card_image_url_render, caption=f"{card_name_render} ({card_set_render})", width=300)
            else: st.warning("Imagen no disponible.")
            links_html = []
            if pd.notna(cardmarket_url_render) and cardmarket_url_render.startswith("http"): links_html.append(f"<a href='{cardmarket_url_render}' target='_blank' style='display: inline-block; margin-top: 10px; margin-right: 10px; padding: 8px 12px; background-color: #FFCB05; color: #2a75bb; text-align: center; border-radius: 5px; text-decoration: none; font-weight: bold;'>Cardmarket</a>")
            if pd.notna(tcgplayer_url_render) and tcgplayer_url_render.startswith("http"): links_html.append(f"<a href='{tcgplayer_url_render}' target='_blank' style='display: inline-block; margin-top: 10px; padding: 8px 12px; background-color: #007bff; color: white; text-align: center; border-radius: 5px; text-decoration: none; font-weight: bold;'>TCGplayer</a>")
            if links_html: st.markdown(" ".join(links_html), unsafe_allow_html=True)
            else: st.caption("Links no disponibles.")
        with col_info:
            st.subheader(f"{card_name_render}")
            st.markdown(f"**ID:** `{card_id_render}`"); st.markdown(f"**Categoría:** {card_supertype_render}"); st.markdown(f"**Set:** {card_set_render}"); st.markdown(f"**Rareza:** {card_rarity_render}")
            if pd.notna(card_artist_render): st.markdown(f"**Artista:** {card_artist_render}")
            if pd.notna(card_price_actual_render): st.metric(label="Precio Actual (Trend €)", value=f"€{card_price_actual_render:.2f}")
            else: st.markdown("**Precio Actual (Trend €):** N/A")
            st.markdown("---"); st.subheader("Predicción de Precio (Modelo Local Estimado)")
            # El botón de predicción aparece solo si el modelo y preprocesadores cargan Y la carta tiene precio actual
            if local_tf_model_layer and ohe_local_preprocessor and scaler_local_preprocessor: # MLP componentes cargados
                if pd.notna(card_price_actual_render): # Solo mostrar botón si hay precio actual para usar como feature
                     if st.button("🧠 Estimar Precio Futuro (MLP)", key=f"predict_mlp_btn_{card_id_render}"):
                         with st.spinner("Calculando estimación (MLP)..."):
                             # Pasamos la Series completa, que ahora puede venir de metadatos o results_df.
                             # Si vino de metadatos, card_price_actual_render será None,
                             # pero la validación de pd.notna(card_price_actual_render) ya maneja esto.
                             # La función predict_price_with_local_tf_layer también maneja precio 0/NaN.
                             pred_price = predict_price_with_local_tf_layer(local_tf_model_layer, ohe_local_preprocessor, scaler_local_preprocessor, card_to_display_in_detail_section)
                         if pred_price is not None:
                             # Solo calcular delta si el precio actual estaba disponible
                             if pd.notna(card_price_actual_render):
                                 delta = pred_price - card_price_actual_render
                                 delta_color = "normal" if delta < -0.01 else ("inverse" if delta > 0.01 else "off")
                                 st.metric(label="Precio Estimado (MLP)", value=f"€{pred_price:.2f}", delta=f"{delta:+.2f}€ vs Actual", delta_color=delta_color)
                             else:
                                 st.metric(label="Precio Estimado (MLP)", value=f"€{pred_price:.2f}") # Mostrar solo la predicción si no hay precio actual
                         else: st.warning("No se pudo obtener estimación (MLP).")
                else:
                     st.info("El precio actual no está disponible para esta carta, no se puede realizar la estimación con el modelo MLP.")
            else:
                 st.warning("El modelo MLP o sus preprocesadores no están cargados correctamente.")


else:
    # Si no hay ninguna carta seleccionada en session_state, mostramos un mensaje guía.
    # Esto ocurrirá si results_df está vacío Y no es la carga inicial (aplicó filtro y no encontró)
    # O si results_df está vacío en la carga inicial (no hay datos en BQ), O si se resetea el estado.
    # La lógica de fallback ya seleccionó una carta si results_df NO está vacío.
    # Por lo tanto, este 'else' solo se ejecuta si results_df está vacío Y session_state.selected_card_id_from_grid is None.
    if results_df.empty:
         if not is_initial_unfiltered_load: # Si se aplicaron filtros y no hubo resultados
              st.info("No se encontraron cartas con los filtros seleccionados.")
         else: # Carga inicial y results_df está vacío
              if bq_client and LATEST_SNAPSHOT_TABLE:
                   if not all_card_metadata_df.empty: # Si hay metadatos, pero no results_df (no hay precio?)
                        st.info("No se encontraron cartas con precio en la base de datos actual.")
                   else: # No hay metadatos en absoluto (BQ error?)
                         st.error("Error interno: No se cargaron los datos de metadatos.")
    else:
         # Esto NO DEBERÍA OCURRIR con la lógica de fallback implementada,
         # ya que el fallback seleccionará una carta aleatoria con precio si results_df no está vacío y no hay selección previa.
         logger.error("UNEXPECTED_STATE: results_df not empty, but session_state.selected_card_id_from_grid is None and detail section not shown.")
         # st.error("Error Interno: Estado inesperado de la aplicación.") # No mostrar error al usuario, solo en logs


st.sidebar.markdown("---")
st.sidebar.caption(f"Pokémon TCG Explorer v1.5 | TF: {tf.__version__}")
