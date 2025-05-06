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

_TARGET_PREDICTED_IS_LOG_TRANSFORMED = True # Modelo predice log1p(precio)
DEFAULT_DAYS_DIFF_FOR_PREDICTION = 29.0    # Basado en la media del scaler


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
        try:
            logger.info(f"LOAD_TF_LAYER: Call Signature (info interna): {model_as_layer_obj._call_signature}")
        except AttributeError:
            logger.warning("LOAD_TF_LAYER: No se pudo acceder a '_call_signature'. Esto es solo informativo.")
        logger.info("LOAD_TF_LAYER: La inspección directa de 'structured_outputs' no está disponible en esta versión de TFSMLayer. "
                    f"Se usará la clave de salida configurada: '{_MODEL_OUTPUT_TENSOR_KEY_NAME}'. "
                    "Verifícala con `saved_model_cli` si hay problemas de predicción.")
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
    # Esta query ya selecciona types y subtypes, así que no necesita cambios aquí
    # si la query en fetch_card_data_from_bq también los selecciona.
    query = f"""
    SELECT
        id, name, supertype, subtypes, types, -- Asegurando que types y subtypes están aquí
        rarity, set_id, set_name,
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

        for col_to_check in ['cardmarket_url', 'tcgplayer_url', 'types', 'subtypes']: # Añadido types y subtypes al check
            if col_to_check not in df.columns:
                df[col_to_check] = None # O un placeholder apropiado como 'Unknown' para types/subtypes
                logger.warning(f"METADATA_BQ: Columna '{col_to_check}' no encontrada en metadatos, añadida como None/placeholder.")

        df['base_pokemon_name'] = df.apply(lambda row: get_true_base_name(row['name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES), axis=1)
        logger.info(f"METADATA_BQ: Metadatos cargados y procesados. Total filas: {len(df)}.")
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower():
            logger.error("METADATA_BQ: Error de 'db-dtypes'.", exc_info=True)
            st.error("Error de Dependencia: Falta 'db-dtypes' para BigQuery. Revisa `requirements.txt`.")
        else:
            logger.error(f"METADATA_BQ: Error al cargar metadatos de BigQuery: {e}", exc_info=True)
            st.error(f"Error al cargar metadatos de cartas: {e}.")
        return pd.DataFrame()


# --- FUNCIÓN DE PREDICCIÓN CON MODELO LOCAL (TFSMLayer) ---
def predict_price_with_local_tf_layer(
    model_layer: tf.keras.layers.TFSMLayer,
    ohe: typing.Any,
    scaler: typing.Any,
    card_data_series: pd.Series # Esta Series viene de tu results_df
) -> float | None:
    logger.info(f"PREDICT_LOCAL_ENTRY: Iniciando predicción para carta ID: {card_data_series.get('id', 'N/A')}")

    if not model_layer or not ohe or not scaler:
        logger.error("PREDICT_LOCAL_FAIL: Modelo TFSMLayer o preprocesadores no disponibles.")
        st.error("Error Interno: Componentes del modelo local no disponibles para predicción.")
        return None

    try:
        # --- PASO 1: Preparar DataFrame de entrada para preprocesamiento ---
        data_for_preprocessing_df_dict = {}

        # Mapeo para Columnas Numéricas
        current_price = card_data_series.get('price')
        if pd.notna(current_price) and current_price > 0:
            data_for_preprocessing_df_dict['price_t0_log'] = np.log1p(current_price)
        else:
            data_for_preprocessing_df_dict['price_t0_log'] = np.log1p(0)
            logger.warning(f"PREDICT_LOCAL_MAP: Precio actual no válido ('{current_price}') para 'price_t0_log', usando np.log1p(0).")

        data_for_preprocessing_df_dict['days_diff'] = float(DEFAULT_DAYS_DIFF_FOR_PREDICTION)

        # Mapeo para Columnas Categóricas
        # Los nombres de clave deben coincidir con _CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING
        # Los valores se toman de card_data_series (que es una fila de results_df)
        data_for_preprocessing_df_dict['artist_name'] = str(card_data_series.get('artist', 'Unknown_Artist'))
        data_for_preprocessing_df_dict['pokemon_name'] = str(card_data_series.get('pokemon_name', 'Unknown_Pokemon'))
        data_for_preprocessing_df_dict['rarity'] = str(card_data_series.get('rarity', 'Unknown_Rarity'))
        data_for_preprocessing_df_dict['set_name'] = str(card_data_series.get('set_name', 'Unknown_Set'))
        data_for_preprocessing_df_dict['supertype'] = str(card_data_series.get('supertype', 'Unknown_Supertype'))

        # Manejo de 'types' (asumiendo que es un string o puede ser una lista)
        types_val = card_data_series.get('types')
        if isinstance(types_val, list) and types_val:
            data_for_preprocessing_df_dict['types'] = str(types_val[0]) if pd.notna(types_val[0]) else 'Unknown_Type'
        elif pd.notna(types_val):
            data_for_preprocessing_df_dict['types'] = str(types_val)
        else:
            data_for_preprocessing_df_dict['types'] = 'Unknown_Type'

        # Manejo de 'subtypes' (asumiendo que es un string o puede ser una lista)
        subtypes_val = card_data_series.get('subtypes')
        if isinstance(subtypes_val, list) and subtypes_val:
            cleaned_subtypes = [str(s) for s in subtypes_val if pd.notna(s)]
            data_for_preprocessing_df_dict['subtypes'] = ', '.join(sorted(list(set(cleaned_subtypes)))) if cleaned_subtypes else 'None'
        elif pd.notna(subtypes_val):
            data_for_preprocessing_df_dict['subtypes'] = str(subtypes_val)
        else:
            data_for_preprocessing_df_dict['subtypes'] = 'None' # O 'Unknown_Subtype'

        current_input_df_for_preprocessing = pd.DataFrame([data_for_preprocessing_df_dict])
        # Asegurar el orden de las columnas exactamente como lo esperan los preprocesadores
        ordered_cols_for_df = _NUMERICAL_COLS_FOR_MODEL_PREPROCESSING + _CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING
        try:
            current_input_df_for_preprocessing = current_input_df_for_preprocessing[ordered_cols_for_df]
        except KeyError as e_key:
            missing_keys_in_df = [col for col in ordered_cols_for_df if col not in current_input_df_for_preprocessing.columns]
            logger.error(f"PREDICT_LOCAL_ORDER_FAIL: Error al ordenar columnas para preprocesamiento. Faltan: {missing_keys_in_df}. Error: {e_key}")
            st.error(f"Error Interno: No se pudieron ordenar las características para el modelo ({', '.join(missing_keys_in_df)}).")
            return None

        logger.info(f"PREDICT_LOCAL_PREPROC_DF: DataFrame para preprocesamiento (1 fila): {current_input_df_for_preprocessing.shape}. Columnas: {list(current_input_df_for_preprocessing.columns)}")
        logger.debug(f"PREDICT_LOCAL_PREPROC_DF_VALUES: Valores: {current_input_df_for_preprocessing.iloc[0].to_dict()}")

        # --- PASO 2: Aplicar preprocesamiento (Scaler y OneHotEncoder) ---
        processed_feature_parts = []
        if _NUMERICAL_COLS_FOR_MODEL_PREPROCESSING:
            num_df_slice = current_input_df_for_preprocessing[_NUMERICAL_COLS_FOR_MODEL_PREPROCESSING]
            if num_df_slice.isnull().values.any():
                logger.warning(f"PREDICT_LOCAL_SCALE: NaNs encontrados en características numéricas ANTES de escalar: {num_df_slice.isnull().sum().to_dict()}. Imputando con 0 para el scaler.")
                num_df_slice = num_df_slice.fillna(0) # Imputación simple, ajusta si es necesario
            numerical_features_scaled_array = scaler.transform(num_df_slice)
            processed_feature_parts.append(numerical_features_scaled_array)
            logger.info(f"PREDICT_LOCAL_SCALE: Características numéricas escaladas (shape): {numerical_features_scaled_array.shape}")

        if _CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING:
            cat_df_slice = current_input_df_for_preprocessing[_CATEGORICAL_COLS_FOR_MODEL_PREPROCESSING].astype(str) # Asegurar strings para OHE
            categorical_features_encoded_sparse = ohe.transform(cat_df_slice)
            categorical_features_encoded_dense_array = categorical_features_encoded_sparse.toarray()
            processed_feature_parts.append(categorical_features_encoded_dense_array)
            logger.info(f"PREDICT_LOCAL_OHE: Características categóricas codificadas (shape): {categorical_features_encoded_dense_array.shape}")

        if not processed_feature_parts:
            logger.error("PREDICT_LOCAL_COMBINE_FAIL: No se procesaron características.")
            st.error("Error Interno: No se pudieron procesar las características para el modelo.")
            return None

        # --- PASO 3: Combinar características preprocesadas ---
        final_input_array_for_model = np.concatenate(processed_feature_parts, axis=1)
        logger.info(f"PREDICT_LOCAL_COMBINE: Array final para modelo (shape): {final_input_array_for_model.shape}")

        EXPECTED_NUM_FEATURES = 4865
        if final_input_array_for_model.shape[1] != EXPECTED_NUM_FEATURES:
            logger.error(f"¡¡¡DESAJUSTE DE SHAPE EN LA ENTRADA DEL MODELO!!!")
            logger.error(f"    Modelo espera: {EXPECTED_NUM_FEATURES} características.")
            logger.error(f"    Array preprocesado tiene: {final_input_array_for_model.shape[1]} características.")
            if 'numerical_features_scaled_array' in locals(): logger.debug(f"    Shape numéricas escaladas: {numerical_features_scaled_array.shape}")
            if 'categorical_features_encoded_dense_array' in locals(): logger.debug(f"    Shape categóricas OHE: {categorical_features_encoded_dense_array.shape}")
            st.error(f"Error Crítico de Preprocesamiento: Discrepancia en el número de características. Esperadas: {EXPECTED_NUM_FEATURES}, Generadas: {final_input_array_for_model.shape[1]}.")
            return None

        # --- PASO 4: Realizar Predicción con TFSMLayer ---
        final_input_tensor_for_model = tf.convert_to_tensor(final_input_array_for_model, dtype=tf.float32)
        logger.info(f"PREDICT_LOCAL_TENSOR: Tensor de entrada para TFSMLayer (shape): {final_input_tensor_for_model.shape}, dtype: {final_input_tensor_for_model.dtype}")
        if _MODEL_INPUT_TENSOR_KEY_NAME:
            model_input_feed_dict = {_MODEL_INPUT_TENSOR_KEY_NAME: final_input_tensor_for_model}
            raw_prediction_output = model_layer(model_input_feed_dict)
        else:
            raw_prediction_output = model_layer(final_input_tensor_for_model)
        logger.info(f"PREDICT_LOCAL_RAW_OUT: Salida cruda de TFSMLayer (tipo {type(raw_prediction_output)}): {raw_prediction_output}")

        if not isinstance(raw_prediction_output, dict):
            if tf.is_tensor(raw_prediction_output): predicted_value_tensor = raw_prediction_output
            else:
                logger.error(f"PREDICT_LOCAL_EXTRACT_FAIL: Salida de TFSMLayer no es ni dict ni tensor, es {type(raw_prediction_output)}.")
                st.error("Error Interno: Formato de salida del modelo local inesperado.")
                return None
        elif not raw_prediction_output:
            logger.error("PREDICT_LOCAL_EXTRACT_FAIL: El diccionario de salida de TFSMLayer está vacío.")
            st.error("Error Interno: El modelo local devolvió una salida vacía.")
            return None
        elif _MODEL_OUTPUT_TENSOR_KEY_NAME not in raw_prediction_output:
            available_keys = list(raw_prediction_output.keys())
            logger.error(f"PREDICT_LOCAL_EXTRACT_FAIL: La clave de salida configurada '{_MODEL_OUTPUT_TENSOR_KEY_NAME}' NO se encuentra en dict. Claves: {available_keys}")
            st.error(f"Error Interno: Clave de salida del modelo ('{_MODEL_OUTPUT_TENSOR_KEY_NAME}') no encontrada. Disponibles: {available_keys}")
            return None
        else:
            predicted_value_tensor = raw_prediction_output[_MODEL_OUTPUT_TENSOR_KEY_NAME]
        logger.info(f"PREDICT_LOCAL_EXTRACT: Tensor de predicción extraído (clave '{_MODEL_OUTPUT_TENSOR_KEY_NAME}' si dict). Shape: {predicted_value_tensor.shape}")
        
        if predicted_value_tensor.shape == (1, 1) or predicted_value_tensor.shape == (1,):
            predicted_value_numeric = predicted_value_tensor.numpy()[0][0] if len(predicted_value_tensor.shape) == 2 else predicted_value_tensor.numpy()[0]
        else:
            logger.error(f"PREDICT_LOCAL_NUMERIC_FAIL: Shape del tensor de predicción inesperado: {predicted_value_tensor.shape}. Esperada (1,1) o (1,).")
            st.error("Error Interno: Formato del valor de predicción inesperado.")
            return None
        logger.info(f"PREDICT_LOCAL_NUMERIC: Valor de predicción numérico extraído: {predicted_value_numeric}")
                
        # --- PASO 5: Postprocesar predicción ---
        if _TARGET_PREDICTED_IS_LOG_TRANSFORMED:
            final_predicted_price = np.expm1(predicted_value_numeric) # Usar expm1 para invertir log1p
        else:
            final_predicted_price = predicted_value_numeric
        logger.info(f"PREDICT_LOCAL_POSTPROC: Predicción final: {final_predicted_price}")
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
all_card_metadata_df = get_card_metadata_with_base_names(bq_client) # Contiene types y subtypes
if not LATEST_SNAPSHOT_TABLE or all_card_metadata_df.empty:
    logger.critical("APP_INIT_FAIL: Datos esenciales de BigQuery no cargados.")
    st.error("Error Crítico: No se pudieron cargar los datos esenciales de BigQuery. La aplicación se detendrá.")
    st.stop()
logger.info("APP_INIT: Datos iniciales de BigQuery cargados OK.")


# --- Sidebar y Filtros ---
st.title("Explorador de Cartas Pokémon TCG")
st.sidebar.header("Filtros y Opciones")
# ... (código de sidebar sin cambios) ...
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


# --- Función para Obtener Datos de Cartas ---
@st.cache_data(ttl=600)
def fetch_card_data_from_bq(
    _client: bigquery.Client,
    latest_table_path: str,
    supertype_ui_filter: str | None,
    sets_ui_filter: list,
    names_ui_filter: list,
    rarities_ui_filter: list,
    sort_direction: str,
    full_metadata_df_param: pd.DataFrame # Usado para pre-filtrar IDs
) -> pd.DataFrame:
    logger.info(f"FETCH_BQ_DATA: Iniciando con filtros - Supertype:{supertype_ui_filter}, Sets:{len(sets_ui_filter)}, Names:{len(names_ui_filter)}, Rarities:{len(rarities_ui_filter)}")
    if not latest_table_path:
        logger.error("FETCH_BQ_DATA_FAIL: 'latest_table_path' es None.")
        st.error("Error Interno: No se pudo determinar la tabla de precios.")
        return pd.DataFrame()

    ids_to_query_df = full_metadata_df_param.copy()
    if supertype_ui_filter and supertype_ui_filter != "Todos": ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]
    if rarities_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:
        actual_name_col_to_filter_on = 'base_pokemon_name' if supertype_ui_filter == 'Pokémon' else 'name'
        if actual_name_col_to_filter_on in ids_to_query_df.columns: ids_to_query_df = ids_to_query_df[ids_to_query_df[actual_name_col_to_filter_on].isin(names_ui_filter)]

    if ids_to_query_df.empty:
        logger.info("FETCH_BQ_DATA: No hay IDs de cartas que coincidan con los filtros de metadatos.")
        return pd.DataFrame()
    list_of_card_ids_to_query = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids_to_query:
        logger.info("FETCH_BQ_DATA: Lista de IDs de cartas para consultar está vacía.")
        return pd.DataFrame()

    # Query actualizada para incluir types y subtypes
    query_sql_template = f"""
    SELECT
        meta.id, meta.name AS pokemon_name, meta.supertype,
        meta.subtypes, meta.types, -- AÑADIDAS
        meta.set_name, meta.rarity, meta.artist, meta.images_large AS image_url,
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
        # ... (manejo de error sin cambios) ...
        if "db-dtypes" in str(e).lower():
            logger.error("FETCH_BQ_DATA_FAIL: Error de 'db-dtypes'.", exc_info=True)
            st.error("Error de Dependencia: Falta 'db-dtypes' para BigQuery. Revisa `requirements.txt`.")
        else:
            logger.error(f"FETCH_BQ_DATA_FAIL: Error en la consulta a BigQuery: {e}", exc_info=True)
            st.error(f"Error al obtener datos de cartas de BigQuery: {e}.")
        return pd.DataFrame()

# --- Obtener datos para la tabla principal ---
results_df = fetch_card_data_from_bq(
    bq_client, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets,
    selected_names_to_filter, selected_rarities, sort_sql, all_card_metadata_df
)
logger.info(f"MAIN_APP: DataFrame 'results_df' cargado con {len(results_df)} filas después de aplicar filtros y consultar BQ.")


# --- Área Principal: Visualización de Resultados (AgGrid) ---
# ... (sin cambios) ...
st.header("Resultados de Cartas")
if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
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
    gb.configure_grid_options(domLayout='normal')
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
    gridOptions = gb.build()
    st.write("Haz clic en una fila de la tabla para ver sus detalles y opciones de predicción:")
    grid_response = AgGrid(
        final_display_df_aggrid, gridOptions=gridOptions, height=500, width='100%',
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        key='pokemon_aggrid_main_display_vFINAL', # Nueva key por si acaso
    )
else:
    logger.info("AGGRID_RENDERING: No hay datos para mostrar en AgGrid.")

# --- Lógica de Manejo de Clic en AgGrid ---
# ... (sin cambios) ...
if grid_response:
    logger.debug(f"AGGRID_HANDLER: Procesando grid_response. Tipo de selected_rows: {type(grid_response.get('selected_rows'))}")
    newly_selected_id_from_grid_click = None
    selected_rows_data_from_grid = grid_response.get('selected_rows')
    if isinstance(selected_rows_data_from_grid, pd.DataFrame) and not selected_rows_data_from_grid.empty:
        try:
            first_selected_row_as_series = selected_rows_data_from_grid.iloc[0]
            if 'ID' in first_selected_row_as_series: newly_selected_id_from_grid_click = first_selected_row_as_series['ID']
            logger.info(f"AGGRID_HANDLER_DF: Fila seleccionada vía DataFrame. ID: {newly_selected_id_from_grid_click if newly_selected_id_from_grid_click else 'No ID'}")
        except Exception as e_aggrid_df: logger.error(f"AGGRID_HANDLER_DF: Error procesando fila seleccionada (DataFrame de AgGrid): {e_aggrid_df}", exc_info=True)
    elif isinstance(selected_rows_data_from_grid, list) and selected_rows_data_from_grid:
        try:
            row_data_dict = selected_rows_data_from_grid[0]
            if isinstance(row_data_dict, dict): newly_selected_id_from_grid_click = row_data_dict.get('ID')
            logger.info(f"AGGRID_HANDLER_LIST: Fila seleccionada vía Lista de Dicts. ID: {newly_selected_id_from_grid_click if newly_selected_id_from_grid_click else 'No ID'}")
        except Exception as e_aggrid_list: logger.error(f"AGGRID_HANDLER_LIST: Error procesando fila seleccionada (Lista de AgGrid): {e_aggrid_list}", exc_info=True)
    current_id_in_session = st.session_state.get('selected_card_id_from_grid')
    if newly_selected_id_from_grid_click is not None and newly_selected_id_from_grid_click != current_id_in_session:
        logger.info(f"AGGRID_HANDLER_STATE_CHANGE: CAMBIO DE SELECCIÓN. Anterior: '{current_id_in_session}', Nuevo: '{newly_selected_id_from_grid_click}'. RE-EJECUTANDO.")
        st.session_state.selected_card_id_from_grid = newly_selected_id_from_grid_click
        st.rerun()

# --- Sección de Detalle de Carta y Predicción ---
# ... (sin cambios en la estructura, solo en la llamada a la predicción si es necesario) ...
st.divider()
st.header("Detalle de Carta Seleccionada")
card_to_display_in_detail_section = None
id_for_detail_view_from_session = st.session_state.get('selected_card_id_from_grid')
logger.info(f"DETAIL_VIEW_ENTRY: Intentando mostrar detalles para ID (de session_state): '{id_for_detail_view_from_session}'")
if id_for_detail_view_from_session:
    if not results_df.empty:
        matched_rows_in_results_df_for_detail = results_df[results_df['id'] == id_for_detail_view_from_session]
        if not matched_rows_in_results_df_for_detail.empty:
            card_to_display_in_detail_section = matched_rows_in_results_df_for_detail.iloc[0]
            logger.info(f"DETAIL_VIEW_FOUND: Carta encontrada en 'results_df' para ID '{id_for_detail_view_from_session}'.")
        else: logger.warning(f"DETAIL_VIEW_NOT_FOUND: ID '{id_for_detail_view_from_session}' NO ENCONTRADO en 'results_df'.")
    else: logger.warning(f"DETAIL_VIEW_NO_DATA: 'results_df' está vacío, no se puede buscar ID '{id_for_detail_view_from_session}'.")
if card_to_display_in_detail_section is None and not results_df.empty:
    card_to_display_in_detail_section = results_df.iloc[0]
    fallback_card_id = card_to_display_in_detail_section.get('id')
    logger.info(f"DETAIL_VIEW_FALLBACK: Usando FALLBACK a la primera carta de 'results_df'. ID: '{fallback_card_id}'.")
    if id_for_detail_view_from_session is None or (fallback_card_id and id_for_detail_view_from_session != fallback_card_id):
        if fallback_card_id and pd.notna(fallback_card_id) and st.session_state.get('selected_card_id_from_grid') != fallback_card_id:
            logger.info(f"DETAIL_VIEW_FALLBACK_STATE_UPDATE: Actualizando session_state con ID de fallback '{fallback_card_id}'.")
            st.session_state.selected_card_id_from_grid = fallback_card_id # No re-run aquí para evitar bucles
if card_to_display_in_detail_section is not None and isinstance(card_to_display_in_detail_section, pd.Series) and not card_to_display_in_detail_section.empty:
    card_id_render = card_to_display_in_detail_section.get('id', "N/A")
    logger.info(f"DETAIL_VIEW_RENDERING: Renderizando detalles para carta ID: '{card_id_render}'.")
    card_name_render = card_to_display_in_detail_section.get('pokemon_name', "N/A") # Usar los nombres de results_df
    card_set_render = card_to_display_in_detail_section.get('set_name', "N/A")
    card_image_url_render = card_to_display_in_detail_section.get('image_url', None)
    card_supertype_render = card_to_display_in_detail_section.get('supertype', "N/A")
    card_rarity_render = card_to_display_in_detail_section.get('rarity', "N/A")
    card_artist_render = card_to_display_in_detail_section.get('artist', None)
    card_price_actual_render = card_to_display_in_detail_section.get('price', None)
    cardmarket_url_render = card_to_display_in_detail_section.get('cardmarket_url', None)
    tcgplayer_url_render = card_to_display_in_detail_section.get('tcgplayer_url', None)
    col_img, col_info = st.columns([1, 2])
    with col_img: # ... (código de imagen y links sin cambios)
        if pd.notna(card_image_url_render) and isinstance(card_image_url_render, str):
            st.image(card_image_url_render, caption=f"{card_name_render} ({card_set_render})", width=300)
        else: st.warning("Imagen no disponible.")
        links_html_parts_render = []
        if pd.notna(cardmarket_url_render) and isinstance(cardmarket_url_render, str) and cardmarket_url_render.startswith("http"): links_html_parts_render.append(f"<a href='{cardmarket_url_render}' target='_blank' style='...'>Ver en Cardmarket</a>") # Estilos omitidos por brevedad
        if pd.notna(tcgplayer_url_render) and isinstance(tcgplayer_url_render, str) and tcgplayer_url_render.startswith("http"): links_html_parts_render.append(f"<a href='{tcgplayer_url_render}' target='_blank' style='...'>Ver en TCGplayer</a>") # Estilos omitidos
        if links_html_parts_render: st.markdown(" ".join(links_html_parts_render), unsafe_allow_html=True)
        else: st.caption("Links de compra no disponibles.")
    with col_info: # ... (código de información de carta sin cambios)
        st.subheader(f"{card_name_render}")
        st.markdown(f"**ID:** `{card_id_render}`")
        st.markdown(f"**Categoría:** {card_supertype_render}")
        st.markdown(f"**Set:** {card_set_render}")
        st.markdown(f"**Rareza:** {card_rarity_render}")
        if pd.notna(card_artist_render) and card_artist_render: st.markdown(f"**Artista:** {card_artist_render}")
        if pd.notna(card_price_actual_render): st.metric(label="Precio Actual (Trend €)", value=f"€{card_price_actual_render:.2f}")
        else: st.markdown("**Precio Actual (Trend €):** N/A")
        st.markdown("---")
        st.subheader("Predicción de Precio (Modelo Local Estimado)")
        if local_tf_model_layer and ohe_local_preprocessor and scaler_local_preprocessor: # MLP
            predict_button_key = f"predict_local_model_btn_{card_id_render}"
            if st.button("🧠 Estimar Precio Futuro (MLP)", key=predict_button_key):
                if pd.notna(card_price_actual_render):
                    with st.spinner("Calculando estimación con modelo MLP..."):
                        predicted_price_from_local_model = predict_price_with_local_tf_layer(
                            local_tf_model_layer, ohe_local_preprocessor, scaler_local_preprocessor,
                            card_to_display_in_detail_section # Pasamos la Series completa
                        )
                    if predicted_price_from_local_model is not None:
                        delta_vs_actual = predicted_price_from_local_model - card_price_actual_render
                        delta_color_display = "normal" if delta_vs_actual < -0.01 else ("inverse" if delta_vs_actual > 0.01 else "off")
                        st.metric(label="Precio Estimado (MLP)", value=f"€{predicted_price_from_local_model:.2f}",
                                  delta=f"{delta_vs_actual:+.2f}€ vs Actual", delta_color=delta_color_display)
                        logger.info(f"DETAIL_VIEW_PRED_MLP_OK: Predicción MLP para '{card_id_render}': {predicted_price_from_local_model:.2f}€")
                    else:
                        st.warning("No se pudo obtener la estimación (MLP). Revisa logs.")
                        logger.warning(f"DETAIL_VIEW_PRED_MLP_FAIL: Predicción MLP para '{card_id_render}' devolvió None.")
                else: st.warning("Estimación MLP no posible sin precio actual.")
        else: st.warning("Modelo MLP o preprocesadores no cargados.")
else:
    logger.info("DETAIL_VIEW_NO_CARD: No hay carta seleccionada o encontrada para detalles.")
    if not results_df.empty: st.info("Selecciona una carta para ver detalles.")

# --- Mensajes finales ---
# ... (sin cambios) ...
if not results_df_for_aggrid_display.empty: pass
elif not results_df.empty and results_df_for_aggrid_display.empty :
    logger.info(f"DISPLAY_MSG_FINAL: 'results_df' tiene {len(results_df)} filas, pero 'results_df_for_aggrid_display' está vacío.")
    st.info(f"Se encontraron {len(results_df)} resultados que coinciden con los filtros, pero no se pueden mostrar. Intenta refinar más los filtros.")
else:
    logger.info("DISPLAY_MSG_FINAL: 'results_df' está vacío (no se encontraron cartas).")
    if bq_client and LATEST_SNAPSHOT_TABLE:
        st.info("No se encontraron cartas que coincidan con los filtros de búsqueda seleccionados. Prueba con criterios más amplios.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Pokémon TCG Explorer v0.6 | TF: {tf.__version__}")
