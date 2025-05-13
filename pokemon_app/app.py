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
MODEL_FILES_DIR = os.path.join(BASE_DIR, "model_files") # Carpeta única para todos los modelos

# MLP (Predicción Futura) - Asumiendo que están en model_files/mlp_v1/
MLP_SAVED_MODEL_PATH = os.path.join(MODEL_FILES_DIR, "mlp_v1")
MLP_OHE_PKL_FILENAME = "ohe_mlp_cat.pkl"
MLP_SCALER_PKL_FILENAME = "scaler_mlp_num.pkl"
MLP_OHE_PATH = os.path.join(MODEL_FILES_DIR, "mlp_v1", MLP_OHE_PKL_FILENAME)
MLP_SCALER_PATH = os.path.join(MODEL_FILES_DIR, "mlp_v1", MLP_SCALER_PKL_FILENAME)

# LightGBM Pipelines (Precio Justo) - Asumiendo que están en model_files/ y subcarpetas High/Low
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
_LGBM_TARGET_LOG = True # Asumiendo que ambos pipelines LGBM predicen log

# --- CONFIGURACIÓN PARA CARTAS DESTACADAS ---
FEATURED_RARITY = 'Special Illustration Rare'
NUM_FEATURED_CARDS_TO_DISPLAY = 5

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
        st.error(f"Error Crítico: El archivo 'saved_model.pb' no se encuentra en '{model_path}'. ")
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
        st.error(f"Error Crítico: El archivo preprocesador '{preprocessor_name}' no se encuentra en '{file_path}'.")
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
    # Query para metadatos (asegurando types y subtypes)
    query = f"""
    SELECT
        id, name, name as pokemon_name, -- Añadir pokemon_name para consistencia si es necesario
        supertype, subtypes, types,
        rarity, set_id, set_name,
        artist, artist as artist_name, -- Añadir artist_name para consistencia
        images_large, cardmarket_url, tcgplayer_url
        -- Columnas necesarias para LGBM (cm_avg1, etc.) NO ESTÁN AQUÍ. Se obtienen de la tabla de precios.
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

    # Query actualizada para incluir TODAS las columnas necesarias para AMBOS modelos
    query_sql_template = f"""
    SELECT
        meta.id, meta.name, meta.name AS pokemon_name, meta.supertype,
        meta.subtypes, meta.types,
        meta.set_name, meta.rarity, meta.artist, meta.artist as artist_name,
        meta.images_large AS image_url,
        meta.cardmarket_url, meta.tcgplayer_url,
        prices.cm_averageSellPrice AS price, -- Para MLP y LGBM
        prices.cm_avg1, prices.cm_avg7, prices.cm_avg30, prices.cm_trendPrice -- Para LGBM
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
        # Convertir columnas de precio a numérico, manejando errores
        price_cols_to_convert = ['price', 'cm_avg1', 'cm_avg7', 'cm_avg30', 'cm_trendPrice']
        for p_col in price_cols_to_convert:
            if p_col in results_from_bq_df.columns:
                results_from_bq_df[p_col] = pd.to_numeric(results_from_bq_df[p_col], errors='coerce')
        logger.info(f"FETCH_BQ_DATA: Consulta a BQ OK. Filas: {len(results_from_bq_df)}.")
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
    # ... (Pega aquí la función predict_price_with_local_tf_layer COMPLETA que ya funcionaba)
    # Asegúrate que use _MLP_NUM_COLS, _MLP_CAT_COLS, _MLP_INPUT_KEY, _MLP_OUTPUT_KEY, _MLP_TARGET_LOG, _MLP_DAYS_DIFF
    # ... y los preprocesadores mlp_ohe, mlp_scaler
    logger.info(f"PREDICT_MLP: Iniciando para ID: {card_data_series.get('id', 'N/A')}")
    if not model_layer or not ohe_mlp or not scaler_mlp: return None
    try:
        data_dict = {}
        price_val = card_data_series.get('price')
        data_dict['price_t0_log'] = np.log1p(price_val) if pd.notna(price_val) and price_val > 0 else np.log1p(0)
        data_dict['days_diff'] = float(_MLP_DAYS_DIFF)
        for col in _MLP_CAT_COLS: # Usa las columnas definidas para MLP
            val = card_data_series.get(col) # Asume que card_data_series tiene estas columnas (ej. 'artist_name')
            if col == 'types':
                if isinstance(val, list) and val: data_dict[col] = str(val[0]) if pd.notna(val[0]) else 'Unknown_Type'
                elif pd.notna(val): data_dict[col] = str(val)
                else: data_dict[col] = 'Unknown_Type'
            elif col == 'subtypes':
                if isinstance(val, list) and val: cleaned = [str(s) for s in val if pd.notna(s)]; data_dict[col] = ', '.join(sorted(list(set(cleaned)))) if cleaned else 'None'
                elif pd.notna(val): data_dict[col] = str(val)
                else: data_dict[col] = 'None'
            else: data_dict[col] = str(val) if pd.notna(val) else f'Unknown_{col}' # Genérico para otras

        df_prep = pd.DataFrame([data_dict])
        df_prep = df_prep[_MLP_NUM_COLS + _MLP_CAT_COLS] # Ordenar

        num_feat = scaler_mlp.transform(df_prep[_MLP_NUM_COLS].fillna(0))
        cat_feat = mlp_ohe.transform(df_prep[_MLP_CAT_COLS].astype(str))
        X_final = np.concatenate([num_feat, cat_feat], axis=1)

        if X_final.shape[1] != 4865: logger.error(f"MLP SHAPE MISMATCH! Expected 4865, got {X_final.shape[1]}"); return None
        
        pred_raw = model_layer(**{_MLP_INPUT_KEY: tf.convert_to_tensor(X_final, dtype=tf.float32)})
        pred_tensor = pred_raw[_MLP_OUTPUT_KEY]
        pred_numeric = pred_tensor.numpy()[0][0]
        return np.expm1(pred_numeric) if _MLP_TARGET_LOG else pred_numeric
    except Exception as e: logger.error(f"PREDICT_MLP_EXC: {e}", exc_info=True); return None


# LGBM
def predict_price_with_lgbm_pipelines_app(
    input_df_row: pd.DataFrame, config: dict, pipe_high, pipe_low
) -> typing.Tuple[typing.Optional[float], str]:
    # ... (Pega aquí la función get_prediction_with_lgbm_pipelines de Colab,
    #      asegurándote que use _LGBM_ALL_INPUT_COLS y _LGBM_TARGET_LOG)
    chosen_pipeline_name = "None"; final_prediction = None
    try: X_new_predict = input_df_row[_LGBM_ALL_INPUT_COLS]
    except KeyError as e: logger.error(f"LGBM_PRED_FAIL: Faltan cols en input: {e}"); return None, "Error_Input_Cols"
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
            pred_log = active_pipe.predict(X_new_predict)
            pred_numeric = pred_log[0]
            final_prediction = np.expm1(pred_numeric) if _LGBM_TARGET_LOG else pred_numeric
        except Exception as e_pipe: logger.error(f"LGBM_PRED_PIPE_EXC: {e_pipe}", exc_info=True); final_prediction = None
    else: logger.error("LGBM_PRED_FAIL: Ningún pipeline activo.")
    return final_prediction, chosen_pipeline_name


# --- Carga de Datos Inicial de BigQuery ---
# ... (sin cambios: logger.info, LATEST_SNAPSHOT_TABLE, all_card_metadata_df) ...

# --- Carga de results_df (después de filtros sidebar) ---
# ... (sin cambios: logger.info, results_df = fetch_card_data_from_bq(...)) ...

# --- Lógica de Selección Inicial / Fallback ---
# ... (sin cambios) ...

# --- Determinar si es carga inicial ---
# ... (is_initial_unfiltered_load sin cambios) ...


# --- SECCIÓN PRINCIPAL DE CONTENIDO ---
st.title("Explorador de Cartas Pokémon TCG")

# --- Mostrar Cartas Destacadas O Tabla de Resultados ---
# ... (Lógica de if is_initial_unfiltered_load ... elif not is_initial_unfiltered_load ... sin cambios) ...
# ... (Incluyendo la sección de AgGrid y su manejo de clics)

# --- Sección de Detalle de Carta Seleccionada y Predicción ---
if st.session_state.selected_card_id_from_grid is not None:
    st.divider(); st.header("Detalle de Carta Seleccionada")
    # ... (Lógica para obtener card_to_display_in_detail_section sin cambios) ...

    if card_to_display_in_detail_section is not None:
        # ... (Lógica para extraer card_id_render, card_name_render, etc. sin cambios) ...
        # ... (Columnas col_img y su contenido sin cambios) ...

        with col_info: # Reemplazar esta sección en tu código existente
            # ... (mostrar ID, Categoría, Set, Rareza, Artista, Precio Actual sin cambios) ...
            st.subheader("Estimaciones de Precio")

            # Botón para MLP (Predicción Futura)
            if mlp_model_layer and mlp_ohe and mlp_scaler:
                if pd.notna(card_price_actual_render):
                    if st.button("🔮 Estimar Precio Futuro (MLP)", key=f"predict_mlp_btn_{card_id_render}"):
                        with st.spinner("Calculando estimación futura (MLP)..."):
                            # Asegurar que card_to_display_in_detail_section tiene las columnas esperadas por MLP
                            # (artist_name, pokemon_name, etc.)
                            pred_price_mlp = predict_price_with_local_tf_layer(
                                mlp_model_layer, mlp_ohe, mlp_scaler,
                                card_to_display_in_detail_section # Esta Series debe tener las columnas _MLP_CAT_COLS
                            )
                        if pred_price_mlp is not None:
                            delta_mlp = pred_price_mlp - card_price_actual_render
                            delta_color_mlp = "normal" if delta_mlp < -0.01 else ("inverse" if delta_mlp > 0.01 else "off")
                            st.metric(label="Estimado Futuro (MLP)", value=f"€{pred_price_mlp:.2f}",
                                      delta=f"{delta_mlp:+.2f}€ vs Actual", delta_color=delta_color_mlp)
                        else: st.warning("No se pudo obtener estimación futura (MLP).")
                else:
                     st.info("Estimación futura (MLP) no posible sin precio actual para la carta.")
            else:
                 st.caption("Modelo MLP de predicción futura no disponible.")

            # Botón para LGBM (Precio Justo)
            if lgbm_pipeline_high or lgbm_pipeline_low: # Al menos uno debe estar cargado
                # Para LGBM, necesitamos las columnas de input correctas en card_to_display_in_detail_section
                # results_df (fuente de card_to_display_in_detail_section) ya debería tenerlas
                # gracias a la query SQL actualizada.
                input_df_for_lgbm = pd.DataFrame([card_to_display_in_detail_section])

                # Verificar si tenemos las columnas numéricas para la decisión y el modelo LGBM
                # Si no, el botón no debería aparecer o debería estar deshabilitado.
                can_predict_lgbm = all(col in input_df_for_lgbm.columns and pd.notna(input_df_for_lgbm[col].iloc[0]) for col in _LGBM_NUM_FEATURES_INPUT)

                if can_predict_lgbm:
                    if st.button("⚖️ Calcular Precio Justo (LGBM)", key=f"predict_lgbm_btn_{card_id_render}"):
                        with st.spinner("Calculando precio justo (LGBM)..."):
                            pred_price_lgbm, pipeline_lgbm = predict_price_with_lgbm_pipelines_app(
                                input_df_for_lgbm,
                                lgbm_threshold_config,
                                lgbm_pipeline_high,
                                lgbm_pipeline_low
                            )
                        if pred_price_lgbm is not None:
                            st.metric(label=f"Precio Justo Estimado ({pipeline_lgbm})", value=f"€{pred_price_lgbm:.2f}")
                            if pd.notna(card_price_actual_render): # Solo mostrar delta si hay precio actual
                                delta_lgbm = pred_price_lgbm - card_price_actual_render
                                st.caption(f"Diferencia con precio actual: {delta_lgbm:+.2f}€")
                        else:
                            st.warning("No se pudo obtener el precio justo (LGBM).")
                else:
                    st.info("Cálculo de precio justo (LGBM) no posible sin datos de precios recientes (cm_avgX, trendPrice).")
            else:
                st.caption("Modelos LGBM de precio justo no disponibles.")
# ... (resto del código: sección else para if st.session_state..., y pie de página de sidebar sin cambios) ...
# (Pega aquí el resto del código desde la v1.6, desde la sección de detalles,
#  asegurándote de que la lógica de los botones de predicción se ajuste como arriba)

else:
    # Si no hay ninguna carta seleccionada en session_state, mostramos un mensaje guía.
    # Esto ocurrirá si results_df está vacío Y no es la carga inicial (aplicó filtro y no encontró)
    # O si results_df está vacío en la carga inicial (no hay datos en BQ), O si se resetea el estado.
    # La lógica de fallback ya seleccionó una carta si results_df NO está vacío.
    # Por lo tanto, este 'else' solo se ejecuta si session_state.selected_card_id_from_grid is None.
    # Si results_df está vacío, la sección de detalles no se muestra, y el usuario ve un mensaje abajo.
    # Si results_df NO está vacío, la lógica de fallback ya puso un ID en session_state,
    # y la sección de detalles (if st.session_state.selected_card_id_from_grid is not None:) se ejecutará.
    # No necesitamos un mensaje aquí, ya que el flujo principal o el fallback manejan la visualización.
    pass # No mostrar nada si no hay selección, la sección de detalles ya maneja si mostrarse o no.


st.sidebar.markdown("---")
st.sidebar.caption(f"Pokémon TCG Explorer v1.7 | TF: {tf.__version__}")
