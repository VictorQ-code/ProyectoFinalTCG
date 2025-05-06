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
import joblib # Para cargar .pkl de scikit-learn

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Pokémon TCG Explorer")
logging.basicConfig(
    level=logging.INFO, # Puedes cambiar a logging.DEBUG para más detalle
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
# BQML_MODEL_NAME = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.mlp_price_predictor" # Ya no se usa para la predicción principal
MAX_ROWS_NO_FILTER = 200

# --- RUTAS Y NOMBRES DE ARCHIVOS DEL MODELO LOCAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_files")

TF_SAVED_MODEL_PATH = MODEL_ARTIFACTS_DIR
OHE_PKL_FILENAME = "ohe_mlp_cat.pkl"
SCALER_PKL_FILENAME = "scaler_mlp_num.pkl"
OHE_PATH = os.path.join(MODEL_ARTIFACTS_DIR, OHE_PKL_FILENAME)
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, SCALER_PKL_FILENAME)

# --- CONFIGURACIÓN DEL MODELO LOCAL (¡¡¡MUY IMPORTANTE AJUSTAR!!!) ---
# Estas son las columnas que tu modelo espera DESPUÉS de que hayas extraído
# y posiblemente transformado datos de 'card_data_series', y ANTES de aplicar ohe/scaler.
# EJEMPLO:
_NUMERICAL_COLS_FOR_MODEL_PREPROCESSING = ['log_current_price', 'days_on_market'] # REEMPLAZA con tus nombres reales
_CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING = ['rarity_transformed', 'set_name_category'] # REEMPLAZA con tus nombres reales

# Si tu SavedModel espera un diccionario como entrada para TFSMLayer (ej. {'input_1': tensor}),
# define el nombre de la clave aquí. Si espera un solo tensor, déjalo como None.
_MODEL_INPUT_TENSOR_KEY_NAME = None # Ejemplo: 'input_1' o 'dense_input'
# _MODEL_INPUT_TENSOR_KEY_NAME = 'input_1' # Si tu modelo espera una clave específica

# Nombre de la clave del tensor de salida en el diccionario devuelto por TFSMLayer.
# Comunes son 'output_0', 'dense_N' (nombre de la última capa densa).
# Si no estás seguro, carga el modelo y revisa model_as_layer.structured_outputs
_MODEL_OUTPUT_TENSOR_KEY_NAME = 'output_0' # Ejemplo: 'dense_2' o el nombre de tu capa de salida

# Si tu modelo predice un valor transformado (ej. logaritmo del precio)
_TARGET_PREDICTED_IS_LOG_TRANSFORMED = False # Cambia a True si es necesario
# _TARGET_PREDICTED_IS_LOG_TRANSFORMED = True # Si tu modelo predice log(precio)


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
        # El 'call_endpoint' puede necesitar ser ajustado si usaste firmas nombradas al guardar.
        # 'serving_default' es el estándar para modelos exportados para TF Serving.
        model_as_layer_obj = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        logger.info(f"LOAD_TF_LAYER: SavedModel cargado exitosamente como TFSMLayer.")
        logger.info(f"LOAD_TF_LAYER: Input Signature (puede ser una lista): {model_as_layer_obj.input_signature}")
        logger.info(f"LOAD_TF_LAYER: Output Signature (structured_outputs): {model_as_layer_obj.structured_outputs}")

        # Validación básica de la salida esperada
        if _MODEL_OUTPUT_TENSOR_KEY_NAME not in model_as_layer_obj.structured_outputs:
            logger.warning(f"LOAD_TF_LAYER: La clave de salida configurada '{_MODEL_OUTPUT_TENSOR_KEY_NAME}' "
                           f"NO se encuentra en las salidas del modelo: {list(model_as_layer_obj.structured_outputs.keys())}. "
                           "Por favor, verifica _MODEL_OUTPUT_TENSOR_KEY_NAME.")
            st.warning(f"Advertencia: La clave de salida del modelo local '{_MODEL_OUTPUT_TENSOR_KEY_NAME}' podría no ser correcta. "
                       f"Salidas disponibles: {list(model_as_layer_obj.structured_outputs.keys())}")

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
        st.error(f"Error Crítico al Cargar Preprocesador '{preprocessor_name}': {e}")
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

@st.cache_data(ttl=3600)
def get_card_metadata_with_base_names(_client: bigquery.Client) -> pd.DataFrame:
    query = f"""
    SELECT
        id, name, supertype, subtypes, rarity, set_id, set_name,
        artist, images_large, cardmarket_url, tcgplayer_url
    FROM `{CARD_METADATA_TABLE}`
    """
    logger.info(f"METADATA_BQ: Ejecutando query para metadatos: {query[:100]}...")
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            logger.warning("METADATA_BQ: DataFrame de metadatos vacío devuelto por BigQuery.")
            st.warning("No se pudieron cargar los metadatos de las cartas desde BigQuery.")
            return pd.DataFrame()

        for col_to_check in ['cardmarket_url', 'tcgplayer_url']:
            if col_to_check not in df.columns:
                df[col_to_check] = None
                logger.warning(f"METADATA_BQ: Columna '{col_to_check}' no encontrada en metadatos, añadida como None.")

        df['base_pokemon_name'] = df.apply(lambda row: get_true_base_name(row['name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES), axis=1)
        logger.info(f"METADATA_BQ: Metadatos cargados y procesados. Total filas: {len(df)}.")
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower():
            logger.error("METADATA_BQ: Error de 'db-dtypes'. Asegúrate de que 'pip install google-cloud-bigquery[pandas]' o 'pip install db-dtypes' esté en tu requirements.txt.", exc_info=True)
            st.error("Error de Dependencia: Falta 'db-dtypes' para BigQuery. Revisa `requirements.txt`.")
        else:
            logger.error(f"METADATA_BQ: Error al cargar metadatos de BigQuery: {e}", exc_info=True)
            st.error(f"Error al cargar metadatos de cartas: {e}.")
        return pd.DataFrame()


# --- FUNCIÓN DE PREDICCIÓN CON MODELO LOCAL (TFSMLayer) ---
def predict_price_with_local_tf_layer(
    model_layer: tf.keras.layers.TFSMLayer,
    ohe: joblib.externals.loky.backend.ReductionMixin, # O el tipo exacto de tu OHE
    scaler: joblib.externals.loky.backend.ReductionMixin, # O el tipo exacto de tu Scaler
    card_data_series: pd.Series
) -> float | None:
    logger.info(f"PREDICT_LOCAL_ENTRY: Iniciando predicción para carta ID: {card_data_series.get('id', 'N/A')}")

    if not model_layer:
        logger.error("PREDICT_LOCAL_FAIL: El objeto TFSMLayer no está cargado o es None.")
        st.error("Error Interno: Modelo local (TFSMLayer) no disponible para predicción.")
        return None
    if not ohe:
        logger.error("PREDICT_LOCAL_FAIL: Objeto OneHotEncoder no cargado o es None.")
        st.error("Error Interno: Preprocesador OneHotEncoder no disponible.")
        return None
    if not scaler:
        logger.error("PREDICT_LOCAL_FAIL: Objeto Scaler no cargado o es None.")
        st.error("Error Interno: Preprocesador Scaler no disponible.")
        return None

    try:
        # --- PASO 1: Preparar DataFrame de entrada para preprocesamiento ---
        # Esta sección es CRÍTICA y depende de cómo entrenaste tu modelo.
        # Debes mapear los datos de `card_data_series` a las características
        # que tus preprocesadores (ohe, scaler) y luego tu modelo esperan.
        data_for_preprocessing_df_dict = {}

        # EJEMPLO DE MAPEO (DEBES AJUSTAR ESTO DETALLADAMENTE):
        # Asume que card_data_series tiene 'price', 'rarity', 'set_name'
        # y tus _NUMERICAL_COLS_FOR_MODEL_PREPROCESSING / _CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING
        # están definidos arriba.

        # Característica numérica de ejemplo: logaritmo del precio actual
        # Suponiendo que 'log_current_price' está en _NUMERICAL_COLS_FOR_MODEL_PREPROCESSING
        if 'log_current_price' in _NUMERICAL_COLS_FOR_MODEL_PREPROCESSING:
            current_price = card_data_series.get('price', 0)
            if pd.notna(current_price) and current_price > 0:
                data_for_preprocessing_df_dict['log_current_price'] = np.log(current_price)
            else:
                data_for_preprocessing_df_dict['log_current_price'] = 0 # O imputación, o error si es crítico
                logger.warning("PREDICT_LOCAL_MAP: Precio actual no válido para log, usando 0.")
        
        # Característica categórica de ejemplo: rareza
        # Suponiendo que 'rarity_transformed' está en _CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING
        if 'rarity_transformed' in _CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING:
             data_for_preprocessing_df_dict['rarity_transformed'] = card_data_series.get('rarity', 'Unknown') # Usar 'Unknown' o una categoría por defecto

        # ... Añade más mapeos para TODAS tus características en _NUMERICAL_COLS_... y _CATEGORICAL_COLS_...

        # Validar que todas las columnas esperadas para preprocesamiento estén presentes
        all_expected_cols_for_prep = _NUMERICAL_COLS_FOR_MODEL_PREPROCESSING + _CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING
        missing_cols_for_prep = [col for col in all_expected_cols_for_prep if col not in data_for_preprocessing_df_dict]
        if missing_cols_for_prep:
            logger.error(f"PREDICT_LOCAL_MAP_FAIL: Faltan columnas después del mapeo para preprocesamiento: {missing_cols_for_prep}. "
                         f"Columnas mapeadas: {list(data_for_preprocessing_df_dict.keys())}")
            st.error(f"Error Interno: No se pudieron preparar todas las características requeridas para el modelo local ({', '.join(missing_cols_for_prep)}).")
            return None

        current_input_df_for_preprocessing = pd.DataFrame([data_for_preprocessing_df_dict])
        logger.info(f"PREDICT_LOCAL_PREPROC_DF: DataFrame para preprocesamiento: {current_input_df_for_preprocessing.to_dict()}")


        # --- PASO 2: Aplicar preprocesamiento (Scaler y OneHotEncoder) ---
        processed_feature_parts = []

        # Procesar características numéricas
        if _NUMERICAL_COLS_FOR_MODEL_PREPROCESSING:
            # Asegurarse de que las columnas existan en el DataFrame y no estén todas NaN
            num_df_slice = current_input_df_for_preprocessing[_NUMERICAL_COLS_FOR_MODEL_PREPROCESSING]
            if num_df_slice.isnull().all().all(): # Si todas las celdas son NaN
                logger.error("PREDICT_LOCAL_SCALE_FAIL: Todas las características numéricas son NaN antes de escalar.")
                st.error("Error Interno: Datos numéricos no válidos para el modelo.")
                return None
            
            numerical_features_scaled_array = scaler.transform(num_df_slice)
            processed_feature_parts.append(numerical_features_scaled_array)
            logger.info(f"PREDICT_LOCAL_SCALE: Características numéricas escaladas (shape): {numerical_features_scaled_array.shape}")
            logger.debug(f"PREDICT_LOCAL_SCALE_VALS: Valores escalados (primeros 5): {numerical_features_scaled_array.flatten()[:5]}")


        # Procesar características categóricas
        if _CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING:
            cat_df_slice = current_input_df_for_preprocessing[_CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING]
            categorical_features_encoded_sparse = ohe.transform(cat_df_slice)
            categorical_features_encoded_dense_array = categorical_features_encoded_sparse.toarray() # Keras/TF suele necesitar array denso
            processed_feature_parts.append(categorical_features_encoded_dense_array)
            logger.info(f"PREDICT_LOCAL_OHE: Características categóricas codificadas (shape): {categorical_features_encoded_dense_array.shape}")
            logger.debug(f"PREDICT_LOCAL_OHE_VALS: Valores OHE (primeros 5 de la primera fila): {categorical_features_encoded_dense_array[0, :5]}")


        if not processed_feature_parts:
            logger.error("PREDICT_LOCAL_COMBINE_FAIL: No se procesaron características (ni numéricas ni categóricas).")
            st.error("Error Interno: No se pudieron procesar las características para el modelo.")
            return None

        # --- PASO 3: Combinar características preprocesadas en el orden correcto ---
        # El orden de concatenación DEBE ser el mismo que durante el entrenamiento del modelo TF.
        # Generalmente: [escaladas_numericas, codificadas_categoricas_ohe]
        final_input_array_for_model = np.concatenate(processed_feature_parts, axis=1)
        logger.info(f"PREDICT_LOCAL_COMBINE: Array final para modelo (shape): {final_input_array_for_model.shape}")
        logger.debug(f"PREDICT_LOCAL_COMBINE_VALS: Array final (primeros 10 valores): {final_input_array_for_model.flatten()[:10]}")


        # --- PASO 4: Realizar Predicción con TFSMLayer ---
        # Convertir a tensor con el dtype esperado (usualmente float32)
        final_input_tensor_for_model = tf.convert_to_tensor(final_input_array_for_model, dtype=tf.float32)
        logger.info(f"PREDICT_LOCAL_TENSOR: Tensor de entrada para TFSMLayer (shape): {final_input_tensor_for_model.shape}, dtype: {final_input_tensor_for_model.dtype}")

        # La llamada a la capa puede requerir un diccionario si el SavedModel tiene múltiples
        # tensores de entrada con nombre (definido por _MODEL_INPUT_TENSOR_KEY_NAME).
        if _MODEL_INPUT_TENSOR_KEY_NAME:
            model_input_feed_dict = {_MODEL_INPUT_TENSOR_KEY_NAME: final_input_tensor_for_model}
            logger.info(f"PREDICT_LOCAL_CALL: Llamando a TFSMLayer con diccionario de entrada: Clave='{_MODEL_INPUT_TENSOR_KEY_NAME}'")
            raw_prediction_output_dict = model_layer(model_input_feed_dict)
        else:
            logger.info("PREDICT_LOCAL_CALL: Llamando a TFSMLayer con tensor de entrada directo.")
            raw_prediction_output_dict = model_layer(final_input_tensor_for_model) # Puede devolver dict o tensor

        logger.info(f"PREDICT_LOCAL_RAW_OUT: Salida cruda de TFSMLayer (tipo {type(raw_prediction_output_dict)}): {raw_prediction_output_dict}")

        # Extraer el tensor de predicción del diccionario de salida (si es un diccionario)
        if not isinstance(raw_prediction_output_dict, dict):
            if tf.is_tensor(raw_prediction_output_dict): # Si devuelve un solo tensor directamente
                logger.info("PREDICT_LOCAL_EXTRACT: TFSMLayer devolvió un tensor directamente.")
                predicted_value_tensor = raw_prediction_output_dict
            else:
                logger.error(f"PREDICT_LOCAL_EXTRACT_FAIL: Salida de TFSMLayer no es ni dict ni tensor, es {type(raw_prediction_output_dict)}.")
                st.error("Error Interno: Formato de salida del modelo local inesperado.")
                return None
        elif not raw_prediction_output_dict:
            logger.error("PREDICT_LOCAL_EXTRACT_FAIL: El diccionario de salida de TFSMLayer está vacío.")
            st.error("Error Interno: El modelo local devolvió una salida vacía.")
            return None
        elif _MODEL_OUTPUT_TENSOR_KEY_NAME not in raw_prediction_output_dict:
            available_keys = list(raw_prediction_output_dict.keys())
            logger.error(f"PREDICT_LOCAL_EXTRACT_FAIL: La clave de salida configurada '{_MODEL_OUTPUT_TENSOR_KEY_NAME}' "
                           f"NO se encuentra en el diccionario de salida de TFSMLayer. Claves disponibles: {available_keys}")
            st.error(f"Error Interno: Clave de salida del modelo local ('{_MODEL_OUTPUT_TENSOR_KEY_NAME}') no encontrada. Disponibles: {available_keys}")
            return None
        else:
            predicted_value_tensor = raw_prediction_output_dict[_MODEL_OUTPUT_TENSOR_KEY_NAME]
            logger.info(f"PREDICT_LOCAL_EXTRACT: Tensor de predicción extraído usando la clave '{_MODEL_OUTPUT_TENSOR_KEY_NAME}'. Shape: {predicted_value_tensor.shape}")


        # Convertir tensor a número. Asume batch size de 1 y una sola predicción escalar.
        # .numpy() convierte el EagerTensor a un array NumPy.
        if predicted_value_tensor.shape == (1, 1) or predicted_value_tensor.shape == (1,): # (batch, output_dim)
            predicted_value_numeric = predicted_value_tensor.numpy()[0][0] if len(predicted_value_tensor.shape) == 2 else predicted_value_tensor.numpy()[0]
            logger.info(f"PREDICT_LOCAL_NUMERIC: Valor de predicción numérico extraído: {predicted_value_numeric}")
        else:
            logger.error(f"PREDICT_LOCAL_NUMERIC_FAIL: Shape del tensor de predicción inesperado: {predicted_value_tensor.shape}. Se esperaba (1,1) o (1,).")
            st.error("Error Interno: Formato del valor de predicción inesperado.")
            return None

        # --- PASO 5: Postprocesar predicción (si es necesario) ---
        if _TARGET_PREDICTED_IS_LOG_TRANSFORMED:
            final_predicted_price = np.exp(predicted_value_numeric)
            logger.info(f"PREDICT_LOCAL_POSTPROC: Predicción post-exp (deshaciendo log): {final_predicted_price}")
        else:
            final_predicted_price = predicted_value_numeric
            logger.info(f"PREDICT_LOCAL_POSTPROC: Predicción final (sin postprocesamiento de log): {final_predicted_price}")

        return float(final_predicted_price)

    except Exception as e:
        logger.error(f"PREDICT_LOCAL_EXCEPTION: Excepción durante preprocesamiento o predicción local: {e}", exc_info=True)
        st.error(f"Error Crítico Durante la Predicción Local: {e}")
        import traceback
        st.text_area("Stack Trace (Predicción Local):", traceback.format_exc(), height=200)
        return None


# --- Carga de Datos Inicial de BigQuery ---
logger.info("APP_INIT: Cargando datos iniciales de BigQuery.")
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
all_card_metadata_df = get_card_metadata_with_base_names(bq_client)

if not LATEST_SNAPSHOT_TABLE or all_card_metadata_df.empty:
    logger.critical("APP_INIT_FAIL: Datos esenciales de BigQuery (snapshot o metadatos) no cargados. La aplicación no puede continuar sin ellos.")
    st.error("Error Crítico: No se pudieron cargar los datos esenciales de BigQuery (precios o metadatos de cartas). La aplicación se detendrá.")
    st.stop()
logger.info("APP_INIT: Datos iniciales de BigQuery cargados OK.")


# --- Sidebar y Filtros (tu código existente) ---
st.title("Explorador de Cartas Pokémon TCG")
st.sidebar.header("Filtros y Opciones")
# ... (Tu código de sidebar y filtros aquí, sin cambios) ...
# --- Sidebar (igual que antes) ---
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


# --- Función para Obtener Datos de Cartas (tu código existente) ---
@st.cache_data(ttl=600) # Ajusta TTL según necesidad
def fetch_card_data_from_bq(
    _client: bigquery.Client,
    latest_table_path: str,
    supertype_ui_filter: str | None,
    sets_ui_filter: list,
    names_ui_filter: list,
    rarities_ui_filter: list,
    sort_direction: str,
    full_metadata_df_param: pd.DataFrame # Renombrado para evitar confusión con la global
) -> pd.DataFrame:
    logger.info(f"FETCH_BQ_DATA: Iniciando con filtros - Supertype:{supertype_ui_filter}, Sets:{len(sets_ui_filter)}, Names:{len(names_ui_filter)}, Rarities:{len(rarities_ui_filter)}")

    if not latest_table_path: # Chequeo añadido
        logger.error("FETCH_BQ_DATA_FAIL: 'latest_table_path' (tabla de precios snapshot) es None. No se puede consultar.")
        st.error("Error Interno: No se pudo determinar la tabla de precios. Intenta recargar.")
        return pd.DataFrame()

    ids_to_query_df = full_metadata_df_param.copy() # Usar el parámetro
    if supertype_ui_filter and supertype_ui_filter != "Todos":
        ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter:
        ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]
    if rarities_ui_filter:
        ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:
        actual_name_col_to_filter_on = 'base_pokemon_name' if supertype_ui_filter == 'Pokémon' else 'name'
        if actual_name_col_to_filter_on in ids_to_query_df.columns:
            ids_to_query_df = ids_to_query_df[ids_to_query_df[actual_name_col_to_filter_on].isin(names_ui_filter)]
        else:
            logger.warning(f"FETCH_BQ_DATA: Columna de nombre para filtrar '{actual_name_col_to_filter_on}' no encontrada en el DataFrame de metadatos después de filtros previos.")

    if ids_to_query_df.empty:
        logger.info("FETCH_BQ_DATA: No hay IDs de cartas que coincidan con los filtros de metadatos. Devolviendo DataFrame vacío.")
        return pd.DataFrame()

    list_of_card_ids_to_query = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids_to_query:
        logger.info("FETCH_BQ_DATA: Lista de IDs de cartas para consultar está vacía. Devolviendo DataFrame vacío.")
        return pd.DataFrame()

    query_sql_template = f"""
    SELECT
        meta.id, meta.name AS pokemon_name, meta.supertype, meta.set_name,
        meta.rarity, meta.artist, meta.images_large AS image_url,
        meta.cardmarket_url, meta.tcgplayer_url, prices.cm_trendPrice AS price
    FROM `{CARD_METADATA_TABLE}` AS meta
    JOIN `{latest_table_path}` AS prices ON meta.id = prices.id
    WHERE meta.id IN UNNEST(@card_ids_param)
    ORDER BY prices.cm_trendPrice {sort_direction}
    """
    query_params_bq = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids_to_query)]
    job_config_bq = bigquery.QueryJobConfig(query_parameters=query_params_bq)

    logger.info(f"FETCH_BQ_DATA: Ejecutando consulta a BigQuery para {len(list_of_card_ids_to_query)} IDs. Orden: {sort_direction}")
    try:
        results_from_bq_df = _client.query(query_sql_template, job_config=job_config_bq).to_dataframe()
        if 'price' in results_from_bq_df.columns:
            results_from_bq_df['price'] = pd.to_numeric(results_from_bq_df['price'], errors='coerce')
        logger.info(f"FETCH_BQ_DATA: Consulta a BigQuery OK. Filas devueltas: {len(results_from_bq_df)}.")
        return results_from_bq_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): # Re-chequeo por si acaso
            logger.error("FETCH_BQ_DATA_FAIL: Error de 'db-dtypes'.", exc_info=True)
            st.error("Error de Dependencia: Falta 'db-dtypes' para BigQuery. Revisa `requirements.txt`.")
        else:
            logger.error(f"FETCH_BQ_DATA_FAIL: Error en la consulta a BigQuery: {e}", exc_info=True)
            st.error(f"Error al obtener datos de cartas de BigQuery: {e}.")
        return pd.DataFrame()

# Obtener datos para la tabla principal
results_df = fetch_card_data_from_bq(
    bq_client, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets,
    selected_names_to_filter, selected_rarities, sort_sql, all_card_metadata_df
)
logger.info(f"MAIN_APP: DataFrame 'results_df' cargado con {len(results_df)} filas después de aplicar filtros y consultar BQ.")


# --- Área Principal: Visualización de Resultados (AgGrid - tu código existente) ---
st.header("Resultados de Cartas")
# ... (Tu código de AgGrid y manejo de selección aquí, sin cambios) ...
if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
    logger.info("SESSION_STATE_INIT: 'selected_card_id_from_grid' inicializado a None.")

logger.info(f"AGGRID_RENDERING: ID en session_state ANTES de AgGrid: {st.session_state.get('selected_card_id_from_grid')}")

results_df_for_aggrid_display = results_df
is_initial_unfiltered_load = (not selected_sets and not selected_names_to_filter and not selected_rarities and (selected_supertype == "Todos" or not selected_supertype))
if is_initial_unfiltered_load and len(results_df) > MAX_ROWS_NO_FILTER:
    logger.info(f"AGGRID_RENDERING: Limitando display a {MAX_ROWS_NO_FILTER} filas de {len(results_df)}.")
    st.info(f"Mostrando los primeros {MAX_ROWS_NO_FILTER} de {len(results_df)} resultados. Aplica filtros para refinar la búsqueda.")
    results_df_for_aggrid_display = results_df.head(MAX_ROWS_NO_FILTER)

grid_response = None
if not results_df_for_aggrid_display.empty:
    display_columns_mapping = {'id': 'ID', 'pokemon_name': 'Nombre Carta', 'supertype': 'Categoría',
                               'set_name': 'Set', 'rarity': 'Rareza', 'artist': 'Artista',
                               'price': 'Precio (Trend €)'}
    cols_in_df_for_display = [col for col in display_columns_mapping.keys() if col in results_df_for_aggrid_display.columns]
    final_display_df_aggrid = results_df_for_aggrid_display[cols_in_df_for_display].copy()
    final_display_df_aggrid.rename(columns=display_columns_mapping, inplace=True)

    price_display_col_name_in_aggrid = display_columns_mapping.get('price')
    if price_display_col_name_in_aggrid and price_display_col_name_in_aggrid in final_display_df_aggrid.columns:
         final_display_df_aggrid[price_display_col_name_in_aggrid] = final_display_df_aggrid[price_display_col_name_in_aggrid].apply(
             lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A"
         )

    gb = GridOptionsBuilder.from_dataframe(final_display_df_aggrid)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_grid_options(domLayout='normal') # 'autoHeight' puede ser útil a veces
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
    # Podrías añadir más configuraciones, como ordenación en cliente si es útil
    # gb.configure_default_column(sortable=True, filter=True)
    gridOptions = gb.build()

    st.write("Haz clic en una fila de la tabla para ver sus detalles y opciones de predicción:")
    grid_response = AgGrid(
        final_display_df_aggrid, gridOptions=gridOptions, height=500, width='100%',
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=False, # Puede ser True si tienes pocas columnas
        allow_unsafe_jscode=True,
        key='pokemon_aggrid_main_display_v4', # Key estática, cámbiala si cambias radicalmente AgGrid
        # enable_enterprise_modules=False # A menos que tengas licencia
    )
else:
    logger.info("AGGRID_RENDERING: No hay datos para mostrar en AgGrid (results_df_for_aggrid_display está vacío).")


# --- Lógica de Manejo de Clic en AgGrid (tu código existente, revisado para robustez) ---
if grid_response:
    logger.debug(f"AGGRID_HANDLER: Procesando grid_response. Tipo de selected_rows: {type(grid_response.get('selected_rows'))}")
    newly_selected_id_from_grid_click = None # Renombrado para claridad
    selected_rows_data_from_grid = grid_response.get('selected_rows')

    if isinstance(selected_rows_data_from_grid, pd.DataFrame) and not selected_rows_data_from_grid.empty:
        try:
            first_selected_row_as_series = selected_rows_data_from_grid.iloc[0]
            if 'ID' in first_selected_row_as_series: # 'ID' es el nombre de columna en final_display_df_aggrid
                newly_selected_id_from_grid_click = first_selected_row_as_series['ID']
            else:
                logger.warning("AGGRID_HANDLER_DF: Columna 'ID' no encontrada en la fila seleccionada del DataFrame de AgGrid.")
            logger.info(f"AGGRID_HANDLER_DF: Fila seleccionada vía DataFrame. ID: {newly_selected_id_from_grid_click if newly_selected_id_from_grid_click else 'No ID'}")
        except IndexError:
            logger.warning("AGGRID_HANDLER_DF: Error de índice al acceder a selected_rows.iloc[0] (DataFrame de AgGrid podría estar vacío inesperadamente).")
        except Exception as e_aggrid_df:
            logger.error(f"AGGRID_HANDLER_DF: Error procesando fila seleccionada (DataFrame de AgGrid): {e_aggrid_df}", exc_info=True)
    elif isinstance(selected_rows_data_from_grid, list) and selected_rows_data_from_grid:
        try:
            row_data_dict = selected_rows_data_from_grid[0] # Asume que es una lista de diccionarios
            if isinstance(row_data_dict, dict):
                newly_selected_id_from_grid_click = row_data_dict.get('ID') # 'ID' es la clave del dict
                logger.info(f"AGGRID_HANDLER_LIST: Fila seleccionada vía Lista de Dicts. ID: {newly_selected_id_from_grid_click if newly_selected_id_from_grid_click else 'No ID'}")
            else:
                logger.warning(f"AGGRID_HANDLER_LIST: Elemento en selected_rows no es un dict, sino {type(row_data_dict)}.")
        except IndexError:
            logger.warning("AGGRID_HANDLER_LIST: selected_rows (lista) está vacía o no tiene elementos.")
        except Exception as e_aggrid_list:
            logger.error(f"AGGRID_HANDLER_LIST: Error procesando fila seleccionada (Lista de AgGrid): {e_aggrid_list}", exc_info=True)
    else:
        logger.debug(f"AGGRID_HANDLER: No hay filas seleccionadas válidas en grid_response o está vacío (selected_rows: {selected_rows_data_from_grid}).")

    current_id_in_session = st.session_state.get('selected_card_id_from_grid')
    logger.debug(f"AGGRID_HANDLER_COMPARE: ID actual en session_state: '{current_id_in_session}', ID recién seleccionado de AgGrid: '{newly_selected_id_from_grid_click}'")

    if newly_selected_id_from_grid_click is not None and newly_selected_id_from_grid_click != current_id_in_session:
        logger.info(f"AGGRID_HANDLER_STATE_CHANGE: CAMBIO DE SELECCIÓN DETECTADO en AgGrid. "
                    f"Anterior: '{current_id_in_session}', Nuevo: '{newly_selected_id_from_grid_click}'. RE-EJECUTANDO Streamlit.")
        st.session_state.selected_card_id_from_grid = newly_selected_id_from_grid_click
        st.rerun()
    elif newly_selected_id_from_grid_click is None and selected_rows_data_from_grid is not None: # Si hubo un clic pero no se pudo obtener ID
        logger.debug(f"AGGRID_HANDLER_NO_CHANGE: Nueva selección de AgGrid es None (o no se pudo extraer ID), no se cambia estado de sesión.")
    else: # Sin cambio o nuevo ID es el mismo
        logger.debug(f"AGGRID_HANDLER_NO_CHANGE: Sin cambio de selección (o nueva selección es igual a la actual), no se re-ejecuta.")


# --- Sección de Detalle de Carta y Predicción ---
st.divider()
st.header("Detalle de Carta Seleccionada")

card_to_display_in_detail_section = None
id_for_detail_view_from_session = st.session_state.get('selected_card_id_from_grid')
logger.info(f"DETAIL_VIEW_ENTRY: Intentando mostrar detalles para ID (de session_state): '{id_for_detail_view_from_session}'")

if id_for_detail_view_from_session:
    if not results_df.empty: # 'results_df' contiene todos los datos, incluyendo columnas no mostradas en AgGrid
        matched_rows_in_results_df_for_detail = results_df[results_df['id'] == id_for_detail_view_from_session]
        if not matched_rows_in_results_df_for_detail.empty:
            card_to_display_in_detail_section = matched_rows_in_results_df_for_detail.iloc[0] # Obtiene la pd.Series
            logger.info(f"DETAIL_VIEW_FOUND: Carta encontrada en 'results_df' para ID '{id_for_detail_view_from_session}'. Nombre: {card_to_display_in_detail_section.get('pokemon_name', 'N/A')}")
        else:
            logger.warning(f"DETAIL_VIEW_NOT_FOUND: ID '{id_for_detail_view_from_session}' NO ENCONTRADO en 'results_df' actual (total {len(results_df)} filas). Esto puede ocurrir si los filtros cambiaron y la carta ya no está.")
    else: # results_df está vacío
        logger.warning(f"DETAIL_VIEW_NO_DATA: 'results_df' está vacío, no se puede buscar ID '{id_for_detail_view_from_session}'.")

# Fallback a la primera carta si no hay selección o la selección no se encuentra (y hay datos)
if card_to_display_in_detail_section is None and not results_df.empty:
    card_to_display_in_detail_section = results_df.iloc[0]
    fallback_card_id = card_to_display_in_detail_section.get('id')
    logger.info(f"DETAIL_VIEW_FALLBACK: Usando FALLBACK a la primera carta de 'results_df'. ID: '{fallback_card_id}', Nombre: {card_to_display_in_detail_section.get('pokemon_name', 'N/A')}")
    # Actualizar session_state si el fallback es diferente o no había nada, para consistencia
    if id_for_detail_view_from_session is None or (fallback_card_id and id_for_detail_view_from_session != fallback_card_id):
        if fallback_card_id and pd.notna(fallback_card_id): # Asegurarse de que el ID de fallback es válido
            if st.session_state.get('selected_card_id_from_grid') != fallback_card_id:
                logger.info(f"DETAIL_VIEW_FALLBACK_STATE_UPDATE: Actualizando session_state con ID de fallback '{fallback_card_id}' (era '{st.session_state.get('selected_card_id_from_grid')}').")
                st.session_state.selected_card_id_from_grid = fallback_card_id
                # No se hace st.rerun() aquí para evitar bucles si el fallback ocurre en cada carga.
                # El próximo ciclo de Streamlit ya usará el ID actualizado.

# Renderizar detalles si tenemos una carta
if card_to_display_in_detail_section is not None and isinstance(card_to_display_in_detail_section, pd.Series) and not card_to_display_in_detail_section.empty:
    card_id_render = card_to_display_in_detail_section.get('id', "N/A")
    logger.info(f"DETAIL_VIEW_RENDERING: Renderizando detalles para carta ID: '{card_id_render}', Nombre: {card_to_display_in_detail_section.get('pokemon_name', 'N/A')}")

    # Extraer datos de la Series para mostrar
    card_name_render = card_to_display_in_detail_section.get('pokemon_name', "Nombre no disponible")
    card_set_render = card_to_display_in_detail_section.get('set_name', "Set no disponible")
    card_image_url_render = card_to_display_in_detail_section.get('image_url', None)
    card_supertype_render = card_to_display_in_detail_section.get('supertype', "Categoría no disponible")
    card_rarity_render = card_to_display_in_detail_section.get('rarity', "Rareza no disponible")
    card_artist_render = card_to_display_in_detail_section.get('artist', None) # Puede ser None
    card_price_actual_render = card_to_display_in_detail_section.get('price', None) # Puede ser None o NaN
    cardmarket_url_render = card_to_display_in_detail_section.get('cardmarket_url', None)
    tcgplayer_url_render = card_to_display_in_detail_section.get('tcgplayer_url', None)

    col_img, col_info = st.columns([1, 2]) # Ajusta proporciones según preferencia
    with col_img:
        if pd.notna(card_image_url_render) and isinstance(card_image_url_render, str):
            st.image(card_image_url_render, caption=f"{card_name_render} ({card_set_render})", width=300) # Ajusta width
        else:
            st.warning("Imagen no disponible para esta carta.")

        # Enlaces de compra
        links_html_parts_render = []
        if pd.notna(cardmarket_url_render) and isinstance(cardmarket_url_render, str) and cardmarket_url_render.startswith("http"):
            links_html_parts_render.append(f"<a href='{cardmarket_url_render}' target='_blank' style='display: inline-block; margin-top: 10px; margin-right: 10px; padding: 8px 12px; background-color: #FFCB05; color: #2a75bb; text-align: center; border-radius: 5px; text-decoration: none; font-weight: bold;'>Ver en Cardmarket</a>")
        if pd.notna(tcgplayer_url_render) and isinstance(tcgplayer_url_render, str) and tcgplayer_url_render.startswith("http"):
            links_html_parts_render.append(f"<a href='{tcgplayer_url_render}' target='_blank' style='display: inline-block; margin-top: 10px; padding: 8px 12px; background-color: #007bff; color: white; text-align: center; border-radius: 5px; text-decoration: none; font-weight: bold;'>Ver en TCGplayer</a>")

        if links_html_parts_render:
            st.markdown(" ".join(links_html_parts_render), unsafe_allow_html=True)
        else:
            st.caption("Links de compra no disponibles.")

    with col_info:
        st.subheader(f"{card_name_render}")
        st.markdown(f"**ID de Carta:** `{card_id_render}`")
        st.markdown(f"**Categoría Principal:** {card_supertype_render}")
        st.markdown(f"**Expansión (Set):** {card_set_render}")
        st.markdown(f"**Rareza:** {card_rarity_render}")
        if pd.notna(card_artist_render) and card_artist_render:
             st.markdown(f"**Artista Ilustrador:** {card_artist_render}")

        if pd.notna(card_price_actual_render):
             st.metric(label="Precio Actual de Mercado (Trend €)", value=f"€{card_price_actual_render:.2f}")
        else:
             st.markdown("**Precio Actual de Mercado (Trend €):** No disponible")

        st.markdown("---") # Separador visual
        st.subheader("Predicción de Precio (Modelo Local Estimado)")

        # Botón para predicción con modelo local TFSMLayer
        if local_tf_model_layer and ohe_local_preprocessor and scaler_local_preprocessor:
            # Usar una key única para el botón basada en el ID de la carta para evitar problemas de estado
            predict_button_key = f"predict_local_model_btn_{card_id_render}"
            if st.button("🧠 Estimar Precio Futuro (Modelo Local)", key=predict_button_key):
                if pd.notna(card_price_actual_render): # El precio actual es crucial si se usa como feature
                    with st.spinner("Calculando estimación con modelo local... Por favor espera."):
                        predicted_price_from_local_model = predict_price_with_local_tf_layer(
                            local_tf_model_layer,
                            ohe_local_preprocessor,
                            scaler_local_preprocessor,
                            card_to_display_in_detail_section # Pasamos la Series completa
                        )

                    if predicted_price_from_local_model is not None:
                        delta_vs_actual = predicted_price_from_local_model - card_price_actual_render
                        delta_color_display = "normal" # Por defecto
                        if delta_vs_actual > 0.01: delta_color_display = "inverse" # Sube
                        elif delta_vs_actual < -0.01: delta_color_display = "normal" # Baja (st.metric normal es rojo)
                        # else: delta_color_display = "off" # Si es muy similar

                        st.metric(label="Precio Estimado (Modelo Local)",
                                  value=f"€{predicted_price_from_local_model:.2f}",
                                  delta=f"{delta_vs_actual:+.2f}€ vs Actual", # Muestra la diferencia
                                  delta_color=delta_color_display)
                        logger.info(f"DETAIL_VIEW_PRED_LOCAL_OK: Predicción local para '{card_id_render}': {predicted_price_from_local_model:.2f}€")
                    else:
                        st.warning("No se pudo obtener la estimación de precio desde el modelo local. Revisa los logs para más detalles.")
                        logger.warning(f"DETAIL_VIEW_PRED_LOCAL_FAIL: Predicción local para '{card_id_render}' devolvió None.")
                else: # Precio actual no disponible
                    st.warning("No se puede realizar la estimación con el modelo local porque el precio actual de la carta no está disponible (y podría ser una característica necesaria).")
                    logger.warning(f"DETAIL_VIEW_PRED_LOCAL_SKIP: No se puede predecir para '{card_id_render}' sin precio actual.")
        else: # Modelo o preprocesadores locales no cargados
            st.warning("El modelo local de estimación de precios o sus preprocesadores no están cargados correctamente. La función de predicción no está disponible.")
            logger.warning("DETAIL_VIEW_PRED_LOCAL_UNAVAILABLE: Modelo local o preprocesadores no disponibles para predicción.")

else: # card_to_display_in_detail_section es None
    logger.info("DETAIL_VIEW_NO_CARD: No hay carta seleccionada o encontrada para mostrar en la sección de detalles.")
    if not results_df.empty: # Hay resultados pero ninguno seleccionado/encontrado
        st.info("Selecciona una carta de la tabla de resultados para ver sus detalles y opciones de estimación de precio.")
    # Si results_df también está vacío, el mensaje de AgGrid o el de abajo se mostrará.

# Mensajes finales basados en si hay resultados
if not results_df_for_aggrid_display.empty:
    pass # AgGrid se mostró
elif not results_df.empty and results_df_for_aggrid_display.empty :
    # Esto ocurre si results_df tiene datos pero se limitó por MAX_ROWS_NO_FILTER y el head() resultó vacío (improbable si MAX_ROWS > 0)
    # O si results_df tiene datos pero algún filtro posterior a la carga de AgGrid lo vació (no debería ser el caso aquí)
    logger.info(f"DISPLAY_MSG_FINAL: 'results_df' tiene {len(results_df)} filas, pero 'results_df_for_aggrid_display' está vacío. Esto es inusual.")
    st.info(f"Se encontraron {len(results_df)} resultados que coinciden con los filtros, pero no se pueden mostrar. Intenta refinar más los filtros.")
else: # results_df está vacío (ninguna carta coincidió con los filtros iniciales)
    logger.info("DISPLAY_MSG_FINAL: 'results_df' está vacío (no se encontraron cartas).")
    if bq_client and LATEST_SNAPSHOT_TABLE: # Si la conexión BQ y la tabla de precios están OK
        st.info("No se encontraron cartas que coincidan con los filtros de búsqueda seleccionados. Prueba con criterios más amplios.")
    # Si bq_client o LATEST_SNAPSHOT_TABLE fallaron antes, ya se mostró un error.

st.sidebar.markdown("---")
st.sidebar.caption(f"Pokémon TCG Explorer v0.2 | TF: {tf.__version__}")
