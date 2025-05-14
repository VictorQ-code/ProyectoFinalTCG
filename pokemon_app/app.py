import streamlit as st
import pandas as pd
# import tensorflow as tf # Ya no se usa para los modelos LGBM
from google.cloud import bigquery
from google.oauth2 import service_account
import re
import logging
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import numpy as np
import os
import joblib # Para cargar modelos LGBM/Pipelines
import typing
import random
import json # Para cargar el umbral

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Pokémon TCG Explorer")
logging.basicConfig(
    level=logging.INFO, # Puedes cambiar a logging.DEBUG para más detalle si depuras
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# Si no usas TF/Keras en ninguna parte, puedes comentar o eliminar estas líneas:
# logger.info(f"TensorFlow Version: {tf.__version__}")
# logger.info(f"Keras Version (via TF): {tf.keras.__version__}")


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

# --- RUTAS Y NOMBRES DE ARCHIVOS DEL MODELO (LGBM) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_files")

LGBM_MODELS_SUBDIR = os.path.join(MODEL_ARTIFACTS_DIR, "lgbm_models") # Subdirectorio para modelos LGBM
PIPE_LOW_PKL_FILENAME = "modelo_pipe_low.pkl"
PIPE_HIGH_PKL_FILENAME = "modelo_pipe_high.pkl"
THRESHOLD_JSON_FILENAME = "threshold.json"

PIPE_LOW_LGBM_PATH = os.path.join(LGBM_MODELS_SUBDIR, PIPE_LOW_PKL_FILENAME)
PIPE_HIGH_LGBM_PATH = os.path.join(LGBM_MODELS_SUBDIR, PIPE_HIGH_PKL_FILENAME)
THRESHOLD_LGBM_PATH = os.path.join(LGBM_MODELS_SUBDIR, THRESHOLD_JSON_FILENAME)


# --- CONFIGURACIÓN DE FEATURES PARA LGBM (Debe coincidir con el entrenamiento LGBM) ---
# Estas son las columnas que tu pipeline LGBM espera en su método .transform() o .predict()
# Asegúrate que los nombres aquí coincidan exactamente con los nombres de las features que usaste en el entrenamiento.
_LGBM_NUMERIC_FEATURES_APP = ['prev_price', 'days_since_prev_snapshot', 'cm_avg1', 'cm_avg7', 'cm_avg30', 'cm_trendPrice']
_LGBM_CATEGORICAL_FEATURES_APP = ['artist_name', 'pokemon_name', 'rarity', 'set_name', 'types', 'supertype', 'subtypes']
# Lista combinada de todas las features esperadas por el pipeline LGBM
_LGBM_ALL_FEATURES_APP = _LGBM_NUMERIC_FEATURES_APP + _LGBM_CATEGORICAL_FEATURES_APP

# Columna usada para decidir qué pipeline LGBM usar (ej. cm_avg7)
_LGBM_THRESHOLD_COLUMN_APP = 'cm_avg7'

# Si el target (predicción) de tus modelos LGBM fue transformado (ej. log)
_LGBM_TARGET_IS_LOG_TRANSFORMED = True # Cambia a False si tu LGBM predice el precio directamente


# --- CONFIGURACIÓN PARA CARTAS DESTACADAS ---
FEATURED_RARITY = 'Special Illustration Rare'
NUM_FEATURED_CARDS_TO_DISPLAY = 5


# --- Conexión Segura a BigQuery ---
@st.cache_resource
def connect_to_bigquery():
    try:
        # ... (código de conexión sin cambios) ...
        creds_json = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(creds_json)
        client = bigquery.Client(credentials=creds, project=GCP_PROJECT_ID)
        logger.info("CONNECT_BQ: Conexión OK.")
        return client
    except Exception as e: logger.error(f"CONNECT_BQ: Error: {e}", exc_info=True); st.error(f"Error BQ: {e}."); return None

bq_client = connect_to_bigquery()
if bq_client is None: st.stop()


# --- FUNCIONES DE CARGA DE MODELOS Y PREPROCESADORES (LGBM) ---
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

@st.cache_data # Usar cache_data para JSON simple
def load_threshold_from_json(file_path):
    if not os.path.exists(file_path):
        logger.error(f"LOAD_THRESHOLD: Archivo de umbral no en: {file_path}")
        st.error(f"Error Crítico: Archivo de umbral no en '{file_path}'.")
        return 30.0 # Valor por defecto si el archivo no existe
    try:
        with open(file_path, "r") as f: data = json.load(f)
        threshold_value = data.get("threshold")
        if threshold_value is None:
            logger.error(f"LOAD_THRESHOLD: Clave 'threshold' no en {file_path}.")
            st.error("Error Crítico: Formato incorrecto de archivo de umbral.")
            return 30.0 # Valor por defecto si el formato es incorrecto
        logger.info(f"LOAD_THRESHOLD: Umbral {threshold_value} cargado desde: {file_path}")
        return float(threshold_value)
    except Exception as e: logger.error(f"LOAD_THRESHOLD: Error: {e}", exc_info=True); st.error(f"Error Cargar Umbral: {e}"); return 30.0 # Valor por defecto en caso de excepción


# --- Carga de Modelos LGBM y Umbral ---
logger.info("APP_INIT: Iniciando carga de modelos LGBM y umbral.")
pipe_low_lgbm_app = load_joblib_object(PIPE_LOW_LGBM_PATH, "Pipeline LGBM Precios Bajos")
pipe_high_lgbm_app = load_joblib_object(PIPE_HIGH_LGBM_PATH, "Pipeline LGBM Precios Altos")
threshold_lgbm_app = load_threshold_from_json(THRESHOLD_LGBM_PATH)


# --- FUNCIONES UTILITARIAS DE DATOS ---
@st.cache_data(ttl=3600)
def get_latest_snapshot_info(_client: bigquery.Client) -> tuple[str | None, pd.Timestamp | None]:
    query = f"SELECT table_id FROM `{_client.project}.{BIGQUERY_DATASET}`.__TABLES__ WHERE STARTS_WITH(table_id, 'monthly_') ORDER BY table_id DESC LIMIT 1"
    try:
        results = _client.query(query).result()
        if results.total_rows > 0:
            latest_table_id_str = list(results)[0].table_id
            full_table_path = f"{_client.project}.{BIGQUERY_DATASET}.{latest_table_id_str}"
            date_str_from_suffix = latest_table_id_str.replace("monthly_", "")
            snapshot_date = pd.to_datetime(date_str_from_suffix, format='%Y_%m_%d')
            logger.info(f"SNAPSHOT_INFO: Usando tabla: {full_table_path}, Fecha Snapshot: {snapshot_date.date()}")
            return full_table_path, snapshot_date
        logger.warning("SNAPSHOT_TABLE: No se encontraron tablas snapshot 'monthly_...'.")
        return None, None
    except Exception as e: logger.error(f"SNAPSHOT_TABLE: Error: {e}", exc_info=True); return None, None

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
    # Query para metadatos, usando nombres de columna reales de BQ y aplicando alias
    query = f"""
    SELECT
        id,
        name AS pokemon_name, -- Nombre real 'name', alias 'pokemon_name'
        supertype,
        subtypes,
        types,
        rarity,
        set_name,
        artist AS artist_name, -- Nombre real 'artist', alias 'artist_name'
        images_large AS image_url,
        cardmarket_url,
        tcgplayer_url
    FROM `{CARD_METADATA_TABLE}`
    """
    logger.info(f"METADATA_BQ: Ejecutando query para metadatos: {query[:100]}...")
    try:
        df = _client.query(query).to_dataframe()
        if df.empty: logger.warning("METADATA_BQ: DataFrame de metadatos vacío."); st.warning("No se pudo cargar metadatos."); return pd.DataFrame()
        # Verificar las columnas con ALIAS
        expected_cols_meta = ['artist_name', 'pokemon_name', 'rarity', 'set_name', 'types', 'supertype', 'subtypes', 'cardmarket_url', 'tcgplayer_url', 'image_url']
        for col in expected_cols_meta:
            if col not in df.columns:
                df[col] = 'Unknown_Placeholder' if col not in ['cardmarket_url', 'tcgplayer_url', 'image_url'] else None
                logger.error(f"METADATA_BQ: Columna con alias '{col}' no se generó correctamente en el DataFrame de metadatos.")
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
    _client: bigquery.Client, latest_table_path_param: str, snapshot_date_param: pd.Timestamp,
    supertype_ui_filter: str | None, sets_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
    sort_direction: str, full_metadata_df_param: pd.DataFrame # Este es all_card_metadata_df
) -> pd.DataFrame:
    logger.info(f"FETCH_BQ_DATA: Ini. SType:{supertype_ui_filter}, Sets:{len(sets_ui_filter)}, Names:{len(names_ui_filter)}, Rars:{len(rarities_ui_filter)}")

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

    snapshot_date_str_for_query = snapshot_date_param.strftime('%Y-%m-%d')
    query_sql_template = f"""
    SELECT
        meta.id,
        meta.name AS pokemon_name,
        meta.supertype,
        meta.subtypes,
        meta.types,
        meta.set_name,
        meta.rarity,
        meta.artist AS artist_name, -- Usar el alias aquí para coincidir con el nombre de feature esperado por LGBM
        meta.images_large AS image_url,
        meta.cardmarket_url,
        meta.tcgplayer_url,
        prices.cm_averageSellPrice AS precio, -- Nombre de columna real en tabla prices
        prices.cm_trendPrice,
        prices.cm_avg1,
        prices.cm_avg7,
        prices.cm_avg30,
        DATE('{snapshot_date_str_for_query}') AS fecha_snapshot -- Añadir fecha snapshot si es necesaria para features como days_since_prev
    FROM `{CARD_METADATA_TABLE}` AS meta
    LEFT JOIN `{latest_table_path_param}` AS prices ON meta.id = prices.id
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
        
        # Añadir features adicionales necesarias para LGBM si no vienen directas de la query
        # Ejemplo: days_since_prev_snapshot si se basa en fecha_snapshot
        if 'fecha_snapshot' in results_from_bq_df.columns:
             # Asumiendo que 'days_since_prev_snapshot' es una constante (ej. 30) para la predicción actual
             results_from_bq_df['days_since_prev_snapshot'] = 30.0 # O calculada si tienes fecha previa en BQ
             logger.info("FETCH_BQ_DATA: Añadida columna 'days_since_prev_snapshot'.")
        else:
             logger.warning("FETCH_BQ_DATA: Columna 'fecha_snapshot' no en resultados. No se puede añadir 'days_since_prev_snapshot' automáticamente.")
             # Si days_since_prev_snapshot es una feature requerida por LGBM y no se puede calcular,
             # DEBES ASEGURARTE DE QUE ESTÉ PRESENTE CON UN VALOR POR DEFECTO O IMPUTADO
             # en la función predict_price_with_lgbm_pipelines_app antes de la predicción.


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
) -> tuple[float | None, str | None]:
    logger.info(f"LGBM_PRED_APP: Iniciando predicción para carta ID: {card_data_for_prediction.get('id', 'N/A')}")
    model_type_used = None
    if not pipe_low_lgbm_loaded or not pipe_high_lgbm_loaded or threshold_lgbm_value is None:
        logger.error("LGBM_PRED_APP: Pipelines LGBM o umbral no cargados.")
        st.error("Error Interno: Modelos LGBM o umbral no disponibles.")
        return None, model_type_used
    try:
        input_dict = {}
        # --- Mapeo de features para LGBM (DEBE COINCIDIR CON _LGBM_ALL_FEATURES_APP) ---
        # Asegúrate que los nombres de clave en input_dict coincidan con los de _LGBM_ALL_FEATURES_APP
        # y que los valores se extraigan correctamente de card_data_for_prediction (la Series).

        # Features Numéricas
        for col_name in _LGBM_NUMERIC_FEATURES_APP:
            val = card_data_for_prediction.get(col_name)
            input_dict[col_name] = float(val) if pd.notna(val) else 0.0 # Imputar NaNs numéricos con 0.0
            if pd.isna(val): logger.warning(f"LGBM_PRED_APP: Feature numérica '{col_name}' es NaN para {card_data_for_prediction.get('id')}. Usando 0.0.")

        # Features Categóricas
        for col_name in _LGBM_CATEGORICAL_FEATURES_APP:
            val = card_data_for_prediction.get(col_name)
            # Conversión a string y manejo básico de NaNs/Listas para categóricas
            if col_name == 'types':
                if isinstance(val, list) and val: input_dict[col_name] = str(val[0]) if pd.notna(val[0]) else 'Unknown_Type'
                elif pd.notna(val): input_dict[col_name] = str(val)
                else: input_dict[col_name] = 'Unknown_Type'
            elif col_name == 'subtypes':
                if isinstance(val, list) and val:
                    cleaned_subtypes = [str(s) for s in val if pd.notna(s)]
                    input_dict[col_name] = ', '.join(sorted(list(set(cleaned_subtypes)))) if cleaned_subtypes else 'None'
                elif pd.notna(val): input_dict[col_name] = str(val)
                else: input_dict[col_name] = 'None'
            elif col_name in ['artist_name', 'pokemon_name', 'rarity', 'set_name', 'supertype']:
                 input_dict[col_name] = str(val) if pd.notna(val) else f'Unknown_{col_name}' # Placeholder específico
            else: # Otras categóricas no mapeadas específicamente
                 input_dict[col_name] = str(val) if pd.notna(val) else 'Unknown_Category'


        X_new_predict_df = pd.DataFrame([input_dict])

        # Asegurar que el DataFrame de entrada tenga TODAS las columnas que el pipeline espera,
        # en el orden correcto. Esto es CRUCIAL para pipelines de sklearn.
        missing_cols_for_lgbm_pipe = [col for col in _LGBM_ALL_FEATURES_APP if col not in X_new_predict_df.columns]
        if missing_cols_for_lgbm_pipe:
             logger.error(f"LGBM_PRED_APP: Faltan columnas en el DataFrame final para el pipeline LGBM: {missing_cols_for_lgbm_pipe}")
             st.error(f"Error Interno: Faltan datos para predicción LGBM ({', '.join(missing_cols_for_lgbm_pipe)}).")
             return None, None
        # Reordenar columnas para que coincidan con el entrenamiento del pipeline
        X_new_predict_for_pipe = X_new_predict_df[_LGBM_ALL_FEATURES_APP]


        # Decidir qué pipeline usar basado en el umbral
        threshold_feature_value = X_new_predict_for_for_pipe.loc[0, _LGBM_THRESHOLD_COLUMN_APP]
        if pd.isna(threshold_feature_value):
            logger.warning(f"LGBM_PRED_APP: Valor para '{_LGBM_THRESHOLD_COLUMN_APP}' es NaN. Usando pipeline bajo por defecto.")
            threshold_feature_value = threshold_lgbm_value - 1 # Forzar uso del pipeline bajo
        
        active_pipe = pipe_low_lgbm_loaded if threshold_feature_value < threshold_lgbm_value else pipe_high_lgbm_loaded
        model_type_used = "Low-Price Pipe" if threshold_feature_value < threshold_lgbm_value else "High-Price Pipe"
        logger.info(f"LGBM_PRED_APP: Usando {model_type_used} basado en {_LGBM_THRESHOLD_COLUMN_APP}={threshold_feature_value}")

        # Realizar la predicción
        pred_log = active_pipe.predict(X_new_predict_for_pipe)
        prediction_final = np.expm1(pred_log[0]) if _LGBM_TARGET_IS_LOG_TRANSFORMED else pred_log[0]
        logger.info(f"LGBM_PRED_APP: Predicción (escala original): {prediction_final:.2f}€")
        return float(prediction_final), model_type_used

    except KeyError as e_key: logger.error(f"LGBM_PRED_APP: KeyError en mapeo o acceso a datos: {e_key}", exc_info=True); st.error(f"Error Datos: Falta la clave '{e_key}' en los datos de la carta. Revisa la configuración de features y la consulta a BQ."); return None, None
    except Exception as e: logger.error(f"LGBM_PRED_APP_EXC: Excepción durante preprocesamiento o predicción LGBM: {e}", exc_info=True); st.error(f"Error Predicción LGBM: {e}"); return None, None


# --- Carga de Datos Inicial de BigQuery ---
logger.info("APP_INIT: Cargando datos iniciales de BigQuery.")
LATEST_SNAPSHOT_TABLE_PATH, LATEST_SNAPSHOT_DATE = get_latest_snapshot_info(bq_client)
# all_card_metadata_df es necesario para los filtros y para la sección destacada
all_card_metadata_df = get_card_metadata_with_base_names(bq_client)

if not LATEST_SNAPSHOT_TABLE_PATH or LATEST_SNAPSHOT_DATE is None or all_card_metadata_df.empty:
    logger.critical("APP_INIT_FAIL: Datos esenciales de BigQuery no cargados.")
    st.error("Error Crítico: No se pudieron cargar los datos esenciales de BigQuery.")
    st.stop()
logger.info("APP_INIT: Datos iniciales de BigQuery cargados OK.")


# --- Sidebar y Filtros ---
st.sidebar.header("Filtros y Opciones")
# selected_... variables ya están definidas por los widgets en la sidebar
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
    bq_client, LATEST_SNAPSHOT_TABLE_PATH, LATEST_SNAPSHOT_DATE,
    selected_supertype, selected_sets, selected_names_to_filter, selected_rarities,
    sort_sql, all_card_metadata_df
)
logger.info(f"MAIN_APP: 'results_df' cargado con {len(results_df)} filas (reflejando filtros).")

# --- Inicializar session_state y Lógica de Fallback ---
if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
    logger.info("SESSION_STATE_INIT: 'selected_card_id_from_grid' inicializado a None.")
# logger.info(f"DETAIL_VIEW_ENTRY (pre-render): ID en session_state: {st.session_state.get('selected_card_id_from_grid')}") # Moved log below

# --- Lógica para establecer la carta seleccionada al inicio si no hay nada ---
# Esto ocurre en la PRIMERA carga o si session_state se resetea y no hay AgGrid seleccionado.
# Queremos seleccionar una carta *aleatoria con precio* de results_df si results_df no está vacío.
# Esta lógica DEBE estar DESPUÉS de que results_df se ha cargado basado en los filtros actuales.
if st.session_state.selected_card_id_from_grid is None and not results_df.empty:
    # Filtrar cartas con precio actual para seleccionar un fallback que permita predecir
    # Nota: La columna de precio en results_df ahora se llama 'precio'
    cards_with_price = results_df[pd.notna(results_df['precio'])]
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
             # Usamos random_state=None para que cambie en cada rerun
             sample_indices = random.sample(special_illustration_rares.index.tolist(), num_cards_to_show)
             display_cards_df = special_illustration_rares.loc[sample_indices].reset_index(drop=True)
        else: display_cards_df = pd.DataFrame()

        if not display_cards_df.empty:
             cols = st.columns(num_cards_to_show)

             for i, card in display_cards_df.iterrows():
                 if i < len(cols): # Asegurarse de no exceder el número de columnas
                     with cols[i]:
                         card_id_featured = card.get('id')
                         card_name_featured = card.get('pokemon_name', 'N/A') # Usar pokemon_name del DF
                         card_set_featured = card.get('set_name', 'N/A')
                         image_url_featured = card.get('image_url')

                         # Mostrar la imagen con st.image (NO es clicable)
                         if pd.notna(image_url_featured) and isinstance(image_url_featured, str):
                             st.image(image_url_featured, width=150, caption=card_set_featured)
                         else:
                             st.warning("Imagen no disponible")
                             st.caption(f"{card_name_featured} ({card_set_featured})")

             st.markdown("---") # Separador visual antes de los detalles

        elif special_illustration_rares.empty:
             logger.info(f"FEATURED_CARDS: No se encontraron cartas con rareza '{FEATURED_RARITY}'.")

    # Este mensaje se muestra solo si special_illustration_rares está vacío Y results_df está vacío
    # y es la carga inicial sin filtros.
    if special_illustration_rares.empty and results_df.empty and is_initial_unfiltered_load and bq_client and LATEST_SNAPSHOT_TABLE_PATH:
         st.info("No se encontraron cartas con la rareza destacada o con precio en la base de datos actual.")
    # La tabla NO se muestra en este bloque (is_initial_unfiltered_load es True)


# Mostrar la tabla SOLO si NO es carga inicial sin filtros (es decir, se aplicaron filtros)
# O si results_df está vacío (esto cubre el caso donde no hay resultados para los filtros aplicados)
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
        # Nota: Usar 'precio' y 'artist_name' si son los nombres en results_df
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
        grid_response = AgGrid( final_display_df_aggrid, gridOptions=gridOptions, height=500, width='100%', data_return_mode=DataReturnMode.AS_INPUT, update_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=False, allow_unsafe_jscode=True, key='pokemon_aggrid_main_display_v1_16') # Nueva key por si acaso
    else: st.info("No hay datos para mostrar en AgGrid que coincidan con los filtros.") # Mensaje si no hay resultados en AgGrid


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
    id_for_detail_view = st.session_state.selected_card_id_from_grid

    # Buscar la carta seleccionada: intentar en results_df primero (tiene precio), si no, en all_card_metadata_df
    source_df_for_details_primary = results_df
    source_df_for_details_secondary = all_card_metadata_df

    if not source_df_for_details_primary.empty and id_for_detail_view in source_df_for_details_primary['id'].values:
         card_to_display_in_detail_section = source_df_for_details_primary[source_df_for_details_primary['id'] == id_for_detail_view].iloc[0]
         logger.info(f"DETAIL_VIEW_FOUND: Carta '{id_for_detail_view}' encontrada en results_df.")
    elif not source_df_for_details_secondary.empty and id_for_detail_view in source_df_for_details_secondary['id'].values:
         card_to_display_in_detail_section = source_df_for_details_secondary[source_df_for_details_secondary['id'] == id_for_detail_view].iloc[0]
         logger.info(f"DETAIL_VIEW_FOUND: Carta '{id_for_detail_view}' encontrada en all_card_metadata_df (precio no garantizado).")
    else:
         logger.warning(f"DETAIL_VIEW_NOT_FOUND: ID '{id_for_detail_view}' NO ENCONTRADO en ninguna fuente de detalles. Resetting selection.")
         st.session_state.selected_card_id_from_grid = None # Resetear selección si no se encuentra
         st.rerun() # Re-ejecutar para limpiar la sección de detalle si la carta no se encuentra


    # Renderizar detalles si tenemos una carta
    if card_to_display_in_detail_section is not None and isinstance(card_to_display_in_detail_section, pd.Series) and not card_to_display_in_detail_section.empty:
        # Extraer datos - usar .get() con valor por defecto por seguridad
        card_id_render = card_to_display_in_detail_section.get('id', "N/A")
        card_name_render = card_to_display_in_detail_section.get('pokemon_name', "N/A")
        card_set_render = card_to_display_in_detail_section.get('set_name', "N/A")
        card_image_url_render = card_to_display_in_detail_section.get('image_url', None)
        card_supertype_render = card_to_display_in_detail_section.get('supertype', "N/A")
        card_rarity_render = card_to_display_in_detail_section.get('rarity', "N/A")
        card_artist_render = card_to_display_in_detail_section.get('artist_name', None) # Usar artist_name
        card_price_actual_render = card_to_display_in_detail_section.get('precio', None) # Usar precio
        # Asegurar que si la carta fue encontrada en metadatos, el precio sea None si no está
        if source_df_for_details_primary is all_card_metadata_df: card_price_actual_render = None
        # Si precio es NaN pero está en results_df, cogerlo de results_df (redundante con la fuente primaria, pero seguro)
        if pd.isna(card_price_actual_render) and card_id_render != "N/A" and not results_df.empty and card_id_render in results_df['id'].values:
             price_check = results_df[results_df['id'] == card_id_render]['precio'].iloc[0]
             if pd.notna(price_check): card_price_actual_render = price_check; logger.warning(f"DETAIL_VIEW: Precio NaN en fuente principal para {card_id_render}, encontrado en results_df: {card_price_actual_render}")

        cardmarket_url_render = card_to_display_in_detail_section.get('cardmarket_url', None)
        tcgplayer_url_render = card_to_display_in_detail_section.get('tcgplayer_url', None)
        col_img, col_info = st.columns([1, 2])
        with col_img:
            if pd.notna(card_image_url_render): st.image(card_image_url_render, caption=f"{card_name_render} ({card_set_render})", width=300)
            else: st.warning("Imagen no disponible.")
            links_html = []
            if pd.notna(cardmarket_url_render) and isinstance(cardmarket_url_render, str) and cardmarket_url_render.startswith("http"): links_html.append(f"<a href='{cardmarket_url_render}' target='_blank' style='display: inline-block; margin-top: 10px; margin-right: 10px; padding: 8px 12px; background-color: #FFCB05; color: #2a75bb; text-align: center; border-radius: 5px; text-decoration: none; font-weight: bold;'>Cardmarket</a>")
            if pd.notna(tcgplayer_url_render) and isinstance(tcgplayer_url_render, str) and tcgplayer_url_render.startswith("http"): links_html.append(f"<a href='{tcgplayer_url_render}' target='_blank' style='display: inline-block; margin-top: 10px; padding: 8px 12px; background-color: #007bff; color: white; text-align: center; border-radius: 5px; text-decoration: none; font-weight: bold;'>TCGplayer</a>")
            if links_html: st.markdown(" ".join(links_html), unsafe_allow_html=True)
            else: st.caption("Links no disponibles.")
        with col_info:
            st.subheader(f"{card_name_render}")
            st.markdown(f"**ID:** `{card_id_render}`"); st.markdown(f"**Categoría:** {card_supertype_render}"); st.markdown(f"**Set:** {card_set_render}"); st.markdown(f"**Rareza:** {card_rarity_render}")
            if pd.notna(card_artist_render): st.markdown(f"**Artista:** {card_artist_render}")
            if pd.notna(card_price_actual_render): st.metric(label="Precio Actual (€)", value=f"€{card_price_actual_render:.2f}")
            else: st.markdown("**Precio Actual (€):** N/A")
            st.markdown("---"); st.subheader("Estimaciones de Precio") # Título más general
            
            # Botón de Predicción LGBM
            if pipe_low_lgbm_app and pipe_high_lgbm_app and threshold_lgbm_app is not None:
                # Columnas que DEBEN estar en card_to_display_in_detail_section y NO ser NaN para que el botón aparezca
                required_cols_for_lgbm_button = ['precio', _LGBM_THRESHOLD_COLUMN_APP] + _LGBM_ALL_FEATURES_APP
                required_cols_for_lgbm_button = list(set(required_cols_for_lgbm_button))
                
                can_predict_lgbm = all(
                    col in card_to_display_in_detail_section.index and \
                    pd.notna(card_to_display_in_detail_section.get(col)) for col in required_cols_for_lgbm_button
                )
                
                if can_predict_lgbm:
                     # Botón de predicción LGBM
                     if st.button("⚡ Estimar Precio Actual (LGBM)", key=f"predict_lgbm_btn_{card_id_render}"):
                         with st.spinner("Calculando estimación (LGBM)..."):
                             # Llamar a la función de predicción LGBM
                             pred_price, pipeline_lgbm_used = predict_price_with_lgbm_pipelines_app(
                                 pipe_low_lgbm_app, pipe_high_lgbm_app, threshold_lgbm_app,
                                 card_to_display_in_detail_section # Pasar la Series completa
                             )
                         if pred_price is not None and card_price_actual_render is not None:
                             delta_lgbm = pred_price - card_price_actual_render
                             delta_color_lgbm = "normal" if delta_lgbm < -0.01 else ("inverse" if delta_lgbm > 0.01 else "off")
                             st.metric(label=f"Precio Justo Estimado ({pipeline_lgbm_used})", value=f"€{pred_price:.2f}", delta=f"{delta_lgbm:+.2f}€ vs Actual", delta_color=delta_color_lgbm)
                         elif pred_price is not None:
                              st.metric(label=f"Precio Justo Estimado ({pipeline_lgbm_used})", value=f"€{pred_price:.2f}")
                         else: st.warning("No se pudo obtener estimación (LGBM).")
                else:
                     # Mensaje si faltan datos para la predicción LGBM
                     missing_pred_cols = [col for col in required_cols_for_lgbm_button if col not in card_to_display_in_detail_section.index or pd.isna(card_to_display_in_detail_section.get(col))]
                     st.info(f"Datos insuficientes para estimación LGBM (faltan o son NaN: {', '.join(missing_pred_cols)}).")
            else:
                 # Mensaje si los modelos LGBM no cargaron
                 st.caption("Modelos LGBM o umbral no disponibles.")


else:
    # Si no hay ninguna carta seleccionada en session_state, y por lo tanto la sección de detalles no se muestra.
    # Esto solo debería ocurrir si results_df está vacío (ya que el fallback selecciona una carta si no está vacío).
    if results_df.empty:
         if not is_initial_unfiltered_load: # Si se aplicaron filtros y no hubo resultados
              st.info("No se encontraron cartas que coincidan con los filtros seleccionados.")
         else: # Carga inicial y results_df está vacío
              if bq_client and LATEST_SNAPSHOT_TABLE_PATH:
                   if not all_card_metadata_df.empty:
                        st.info("No se encontraron cartas con precio en la base de datos actual.")
                   else:
                         st.error("Error interno: No se cargaron los datos de metadatos.")
    # Si results_df no está vacío pero no hay selección (muy raro), la lógica de fallback DEBERÍA haber puesto un ID.
    # Si por alguna razón no lo hizo, no se muestra nada aquí.


st.sidebar.markdown("---")
st.sidebar.caption(f"Pokémon TCG Explorer v1.16 | LGBM")
