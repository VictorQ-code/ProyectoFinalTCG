import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re
import logging
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import numpy as np
import os
# import tensorflow as tf # No necesario si solo usamos LightGBM pipelines
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

# --- Constantes y Configuración de GCP ---
try:
    GCP_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
    logger.info(f"CONFIG: GCP Project ID '{GCP_PROJECT_ID}' cargado.")
except KeyError:
    logger.critical("CRITICAL_CONFIG: 'project_id' no encontrado en secrets.")
    st.error("Error Crítico: Configuración de 'project_id' no encontrada.")
    st.stop()

BIGQUERY_DATASET = "pokemon_dataset"
CARD_METADATA_TABLE = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.card_metadata"
MAX_ROWS_NO_FILTER = 200

# --- RUTAS Y NOMBRES DE ARCHIVOS DEL MODELO LGBM ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Asumimos que los modelos LGBM están en una subcarpeta 'lgbm_models' dentro de 'model_files'
# o ajusta las rutas según dónde los coloques en tu repo de GitHub.
LGBM_MODEL_DIR = os.path.join(BASE_DIR, "model_files", "lgbm_models") # Ejemplo de subcarpeta

PIPE_LOW_PKL_FILENAME = "modelo_pipe_low.pkl"
PIPE_HIGH_PKL_FILENAME = "modelo_pipe_high.pkl" # O "modelo_best_pipe_high.pkl"
THRESHOLD_JSON_FILENAME = "threshold.json"

PIPE_LOW_PATH = os.path.join(LGBM_MODEL_DIR, PIPE_LOW_PKL_FILENAME)
PIPE_HIGH_PATH = os.path.join(LGBM_MODEL_DIR, PIPE_HIGH_PKL_FILENAME)
THRESHOLD_PATH = os.path.join(LGBM_MODEL_DIR, THRESHOLD_JSON_FILENAME)


# --- CONFIGURACIÓN DE FEATURES PARA LGBM (DEBE COINCIDIR CON EL ENTRENAMIENTO) ---
# Estas son las features que tus pipelines LGBM esperan
_LGBM_NUMERIC_FEATURES_APP = [
    'prev_price', 'days_since_prev_snapshot',
    'cm_avg1', 'cm_avg7', 'cm_avg30', 'cm_trendPrice'
]
_LGBM_CATEGORICAL_FEATURES_APP = [
    'artist_name', 'pokemon_name', 'rarity',
    'set_name', 'types', 'supertype', 'subtypes'
]
_LGBM_ALL_FEATURES_APP = _LGBM_NUMERIC_FEATURES_APP + _LGBM_CATEGORICAL_FEATURES_APP
_LGBM_THRESHOLD_COLUMN_APP = 'cm_avg7' # Columna usada para el umbral en predict_mixed

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

# --- FUNCIONES DE CARGA DE MODELOS LGBM Y UMBRAL ---
@st.cache_resource
def load_lgbm_pipeline(file_path, pipeline_name="Pipeline LGBM"):
    if not os.path.exists(file_path):
        logger.error(f"LOAD_LGBM: Archivo '{pipeline_name}' no en: {file_path}")
        st.error(f"Error Crítico: Modelo '{pipeline_name}' no en '{file_path}'.")
        return None
    try:
        pipeline = joblib.load(file_path)
        logger.info(f"LOAD_LGBM: {pipeline_name} cargado desde: {file_path}")
        return pipeline
    except Exception as e:
        logger.error(f"LOAD_LGBM: Error cargando {pipeline_name} desde {file_path}: {e}", exc_info=True)
        st.error(f"Error Crítico al Cargar Modelo {pipeline_name}: {e}")
        return None

@st.cache_data # El umbral es un dato pequeño
def load_threshold(file_path):
    if not os.path.exists(file_path):
        logger.error(f"LOAD_THRESHOLD: Archivo de umbral no en: {file_path}")
        st.error(f"Error Crítico: Archivo de umbral no en '{file_path}'.")
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        threshold_value = data.get("threshold")
        if threshold_value is None:
            logger.error(f"LOAD_THRESHOLD: Clave 'threshold' no en {file_path}.")
            st.error("Error Crítico: Formato incorrecto de archivo de umbral.")
            return 30.0 # Un default por si acaso, pero debería fallar antes
        logger.info(f"LOAD_THRESHOLD: Umbral {threshold_value} cargado desde: {file_path}")
        return float(threshold_value)
    except Exception as e:
        logger.error(f"LOAD_THRESHOLD: Error cargando umbral: {e}", exc_info=True)
        st.error(f"Error Crítico al Cargar Umbral: {e}")
        return 30.0 # Default

# --- Carga de Modelos LGBM y Umbral ---
logger.info("APP_INIT: Iniciando carga de modelos LGBM y umbral.")
pipe_low_lgbm_app = load_lgbm_pipeline(PIPE_LOW_PATH, "Pipeline LGBM Precios Bajos")
pipe_high_lgbm_app = load_lgbm_pipeline(PIPE_HIGH_PATH, "Pipeline LGBM Precios Altos")
threshold_lgbm_app = load_threshold(THRESHOLD_PATH)


# --- Funciones Auxiliares de Datos ---
@st.cache_data(ttl=3600)
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    # ... (sin cambios) ...
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
    # ... (sin cambios) ...
    if not isinstance(name_str, str) or supertype != 'Pokémon': return name_str
    for mw_base in multi_word_bases:
        if name_str.startswith(mw_base): return mw_base
    cleaned_name = name_str
    for suffix in suffixes:
        if cleaned_name.endswith(suffix): cleaned_name = cleaned_name[:-len(suffix)].strip()
    return cleaned_name if cleaned_name else name_str

@st.cache_data(ttl=3600)
def get_card_metadata_with_base_names(_client: bigquery.Client) -> pd.DataFrame:
    # ... (sin cambios, ya selecciona types, subtypes, artist_name, pokemon_name) ...
    query = f"""
    SELECT
        id, name AS pokemon_name, supertype, subtypes, types,
        rarity, set_id, set_name,
        artist AS artist_name, images_large, cardmarket_url, tcgplayer_url
    FROM `{CARD_METADATA_TABLE}`
    """ # 'name' es pokemon_name, 'artist' es artist_name
    logger.info(f"METADATA_BQ: Ejecutando query para metadatos: {query[:100]}...")
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            logger.warning("METADATA_BQ: DataFrame de metadatos vacío devuelto por BigQuery.")
            st.warning("No se pudieron cargar los metadatos de las cartas desde BigQuery.")
            return pd.DataFrame()
        for col_to_check in ['cardmarket_url', 'tcgplayer_url', 'types', 'subtypes', 'artist_name']:
            if col_to_check not in df.columns:
                df[col_to_check] = None
                logger.warning(f"METADATA_BQ: Columna '{col_to_check}' no encontrada en metadatos, añadida como None/placeholder.")
        df['base_pokemon_name_display'] = df.apply(lambda row: get_true_base_name(row['pokemon_name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES), axis=1)
        logger.info(f"METADATA_BQ: Metadatos cargados y procesados. Total filas: {len(df)}.")
        return df


@st.cache_data(ttl=600)
def fetch_card_data_from_bq(
    _client: bigquery.Client, latest_table_path: str, supertype_ui_filter: str | None,
    sets_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
    sort_direction: str, full_metadata_df_param: pd.DataFrame
) -> pd.DataFrame:
    # ... (Definición de fetch_card_data_from_bq, asegurando que la query SQL traiga
    #      TODAS las columnas necesarias: 'precio', 'cm_trendPrice', 'cm_avg1', 'cm_avg7', 'cm_avg30'
    #      y los metadatos como 'artist_name', 'pokemon_name', 'rarity', 'set_name', 'types', 'supertype', 'subtypes')
    logger.info(f"FETCH_BQ_DATA: Ini. SType:{supertype_ui_filter}, Sets:{len(sets_ui_filter)}, Names:{len(names_ui_filter)}, Rars:{len(rarities_ui_filter)}")
    if not latest_table_path: logger.error("FETCH_BQ_DATA_FAIL: 'latest_table_path' es None."); st.error("Error Interno: No se pudo determinar la tabla de precios."); return pd.DataFrame()
    
    ids_to_query_df = full_metadata_df_param.copy()
    if supertype_ui_filter and supertype_ui_filter != "Todos": ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]
    if rarities_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:
        actual_name_col_to_filter_on = 'base_pokemon_name_display' if supertype_ui_filter == 'Pokémon' else 'pokemon_name' # Usar pokemon_name para no-Pokémon
        if actual_name_col_to_filter_on in ids_to_query_df.columns: ids_to_query_df = ids_to_query_df[ids_to_query_df[actual_name_col_to_filter_on].isin(names_ui_filter)]
    
    if ids_to_query_df.empty: logger.info("FETCH_BQ_DATA: No hay IDs que coincidan."); return pd.DataFrame()
    list_of_card_ids_to_query = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids_to_query: logger.info("FETCH_BQ_DATA: Lista IDs vacía."); return pd.DataFrame()

    query_sql_template = f"""
    SELECT
        meta.id, meta.pokemon_name, meta.supertype, meta.subtypes, meta.types,
        meta.set_name, meta.rarity, meta.artist_name, meta.images_large AS image_url,
        meta.cardmarket_url, meta.tcgplayer_url,
        prices.precio, prices.cm_trendPrice,
        prices.cm_avg1, prices.cm_avg7, prices.cm_avg30,
        prices.fecha AS fecha_snapshot -- Añadido para posible uso de days_since_prev_snapshot
    FROM `{CARD_METADATA_TABLE}` AS meta
    JOIN `{latest_table_path}` AS prices ON meta.id = prices.card_id -- ASUMIENDO que la tabla de precios usa card_id
    WHERE meta.id IN UNNEST(@card_ids_param)
    -- No podemos ordenar por prices.cm_trendPrice si no está en la tabla de precios,
    -- o si queremos ordenar por el 'precio' (cm_averageSellPrice) actual.
    -- Por ahora, ordenaremos por el precio principal.
    ORDER BY prices.precio {sort_direction}
    """
    # Nota: La tabla de precios 'monthly_*' debe tener las columnas:
    # id (renombrado a card_id), cm_averageSellPrice (renombrado a precio),
    # cm_trendPrice, cm_avg1, cm_avg7, cm_avg30, y fecha (PARSE_DATE).
    # Si los nombres son diferentes en la tabla de precios, ajústalos en la query.

    query_params_bq = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids_to_query)]
    job_config_bq = bigquery.QueryJobConfig(query_parameters=query_params_bq)
    logger.info(f"FETCH_BQ_DATA: SQL BQ para {len(list_of_card_ids_to_query)} IDs. Orden: {sort_direction}")
    try:
        results_from_bq_df = _client.query(query_sql_template, job_config=job_config_bq).to_dataframe()
        if 'precio' in results_from_bq_df.columns: results_from_bq_df['precio'] = pd.to_numeric(results_from_bq_df['precio'], errors='coerce')
        logger.info(f"FETCH_BQ_DATA: Consulta a BQ OK. Filas: {len(results_from_bq_df)}.")
        # Renombrar para consistencia con el resto de la app si es necesario
        # ej. si 'name' se espera como 'pokemon_name' en results_df
        # if 'name' in results_from_bq_df.columns and 'pokemon_name' not in results_from_bq_df.columns:
        #    results_from_bq_df.rename(columns={'name': 'pokemon_name'}, inplace=True)
        return results_from_bq_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("FETCH_BQ_DATA_FAIL: Error de 'db-dtypes'.", exc_info=True); st.error("Error de Dependencia: Falta 'db-dtypes'.")
        else: logger.error(f"FETCH_BQ_DATA_FAIL: Error BQ: {e}", exc_info=True); st.error(f"Error al obtener datos de cartas: {e}.")
        return pd.DataFrame()


# --- FUNCIÓN DE PREDICCIÓN CON MODELOS LGBM (Adaptada de test_set_future) ---
def predict_price_with_lgbm_pipelines_app(
    pipe_low_lgbm_loaded,
    pipe_high_lgbm_loaded,
    threshold_lgbm_value: float,
    card_data_for_prediction: pd.Series, # Fila de results_df
    # No necesitamos latest_snapshot_date si 'days_since_prev_snapshot' es un horizonte fijo
) -> float | None:
    logger.info(f"LGBM_PRED_APP: Iniciando predicción para carta ID: {card_data_for_prediction.get('id', 'N/A')}")

    if not pipe_low_lgbm_loaded or not pipe_high_lgbm_loaded or threshold_lgbm_value is None:
        logger.error("LGBM_PRED_APP: Pipelines LGBM o umbral no cargados.")
        st.error("Error Interno: Modelos LGBM o umbral no disponibles.")
        return None

    try:
        # --- 1. Crear el DataFrame de entrada para el modelo (X_new_predict) ---
        input_dict = {}

        # a) Características Numéricas
        current_price_val = card_data_for_prediction.get('precio')
        input_dict['prev_price'] = float(current_price_val) if pd.notna(current_price_val) else 0.0
        if pd.isna(current_price_val): logger.warning(f"LGBM_PRED_APP: 'prev_price' (de 'precio' actual) es NaN. Usando 0.0.")

        # 'days_since_prev_snapshot' - para la app, es el horizonte de predicción
        # Este valor debe ser con el que el modelo fue entrenado para esta feature.
        # Si en entrenamiento esto era la diff real entre snapshots (ej. ~30), y quieres predecir
        # el siguiente snapshot, usa ese valor.
        input_dict['days_since_prev_snapshot'] = 30.0 # ¡EJEMPLO! Ajusta al horizonte que tu modelo espera

        # Features que deben venir de card_data_for_prediction (results_df)
        for col_name in ['cm_avg1', 'cm_avg7', 'cm_avg30', 'cm_trendPrice']:
            val = card_data_for_prediction.get(col_name)
            input_dict[col_name] = float(val) if pd.notna(val) else 0.0
            if pd.isna(val): logger.warning(f"LGBM_PRED_APP: Feature numérica '{col_name}' es NaN. Usando 0.0.")

        # b) Características Categóricas
        # Asegúrate que los nombres aquí coincidan con _LGBM_CATEGORICAL_FEATURES_APP
        # y que card_data_for_prediction tenga estas columnas (artist_name, pokemon_name, etc.)
        input_dict['artist_name'] = str(card_data_for_prediction.get('artist_name', 'Unknown_Artist'))
        input_dict['pokemon_name'] = str(card_data_for_prediction.get('pokemon_name', 'Unknown_Pokemon'))
        input_dict['rarity'] = str(card_data_for_prediction.get('rarity', 'Unknown_Rarity'))
        input_dict['set_name'] = str(card_data_for_prediction.get('set_name', 'Unknown_Set'))
        input_dict['supertype'] = str(card_data_for_prediction.get('supertype', 'Unknown_Supertype'))
        
        types_val = card_data_for_prediction.get('types')
        if isinstance(types_val, list) and types_val: input_dict['types'] = str(types_val[0]) if pd.notna(types_val[0]) else 'Unknown_Type'
        elif pd.notna(types_val): input_dict['types'] = str(types_val)
        else: input_dict['types'] = 'Unknown_Type'

        subtypes_val = card_data_for_prediction.get('subtypes')
        if isinstance(subtypes_val, list) and subtypes_val:
            cleaned_subtypes = [str(s) for s in subtypes_val if pd.notna(s)]
            input_dict['subtypes'] = ', '.join(sorted(list(set(cleaned_subtypes)))) if cleaned_subtypes else 'None'
        elif pd.notna(subtypes_val): input_dict['subtypes'] = str(subtypes_val)
        else: input_dict['subtypes'] = 'None'

        # Crear el DataFrame de una fila
        X_new_predict_df = pd.DataFrame([input_dict])
        logger.info(f"LGBM_PRED_APP: DataFrame de entrada creado: {X_new_predict_df.to_dict(orient='records')}")

        # Verificar que todas las columnas que los pipelines esperan estén presentes
        missing_cols = [col for col in _LGBM_ALL_FEATURES_APP if col not in X_new_predict_df.columns]
        if missing_cols:
            logger.error(f"LGBM_PRED_APP: Faltan columnas en X_new_predict_df: {missing_cols}")
            st.error(f"Error Interno: Faltan datos para la predicción LGBM ({', '.join(missing_cols)}).")
            return None
        
        # Seleccionar y ordenar columnas (opcional si ColumnTransformer maneja nombres)
        # X_new_predict_for_pipe = X_new_predict_df[_LGBM_ALL_FEATURES_APP]


        # --- 2. Aplicar el umbral y predecir ---
        threshold_feature_value = X_new_predict_df.loc[0, _LGBM_THRESHOLD_COLUMN_APP]
        
        active_pipe = pipe_low_lgbm_loaded if threshold_feature_value < threshold_lgbm_value else pipe_high_lgbm_loaded
        model_type_used = "Low-Price Pipe" if threshold_feature_value < threshold_lgbm_value else "High-Price Pipe"
        logger.info(f"LGBM_PRED_APP: Usando {model_type_used} basado en {_LGBM_THRESHOLD_COLUMN_APP}={threshold_feature_value}")

        pred_log = active_pipe.predict(X_new_predict_df) # Los pipelines ya tienen preprocesador
        
        # --- 3. Postprocesar (asumiendo que el modelo predice log1p) ---
        # En tu entrenamiento, y_tr_low_log = np.log1p(y_tr_low), y_tr_high_log = np.log1p(y_tr_high)
        prediction_final = np.expm1(pred_log[0])
        logger.info(f"LGBM_PRED_APP: Predicción (escala original): {prediction_final:.2f}€")
        
        return float(prediction_final)

    except KeyError as e_key:
        logger.error(f"LGBM_PRED_APP: KeyError al preparar datos o predecir: {e_key}", exc_info=True)
        st.error(f"Error de Datos: Falta la columna '{e_key}' para la predicción LGBM.")
        return None
    except Exception as e:
        logger.error(f"LGBM_PRED_APP_EXC: Excepción: {e}", exc_info=True)
        st.error(f"Error Crítico Durante la Predicción LGBM: {e}")
        return None


# --- Carga de Datos Inicial de BigQuery ---
logger.info("APP_INIT: Cargando datos iniciales de BigQuery.")
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
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
name_label = "Nombre de Carta:"; name_col_for_options = 'pokemon_name' # Usar 'pokemon_name' de metadatos
if selected_supertype == 'Pokémon': name_col_for_options = 'base_pokemon_name_display'; name_label = "Pokémon (Nombre Base):"
elif selected_supertype != "Todos": name_label = f"Nombre ({selected_supertype}):"
if name_col_for_options in options_df_for_filters.columns: name_options_list = sorted(options_df_for_filters[name_col_for_options].dropna().unique().tolist())
else: name_options_list = []; logger.warning(f"SIDEBAR: Columna '{name_col_for_options}' no encontrada para filtro de nombre.")
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

# --- Inicializar session_state ---
if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
logger.info(f"DETAIL_VIEW_ENTRY (pre-render): ID en session_state: {st.session_state.get('selected_card_id_from_grid')}")

if st.session_state.selected_card_id_from_grid is None and not results_df.empty:
    cards_with_price = results_df[pd.notna(results_df['precio'])] # Usar 'precio'
    if not cards_with_price.empty:
        random_card_row = cards_with_price.sample(1).iloc[0]
        random_card_id = random_card_row.get('id')
        if random_card_id and pd.notna(random_card_id):
            st.session_state.selected_card_id_from_grid = random_card_id
            logger.info(f"FALLBACK_SELECT: Seleccionando carta aleatoria con precio como fallback: '{random_card_id}'.")
    else: logger.warning("FALLBACK_SELECT: No se encontraron cartas con precio en los resultados filtrados para seleccionar un fallback.")

is_initial_unfiltered_load = (not selected_sets and not selected_names_to_filter and not selected_rarities and (selected_supertype == "Todos" or not selected_supertype))

# --- SECCIÓN PRINCIPAL DE CONTENIDO ---
st.title("Explorador de Cartas Pokémon TCG")

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
                     card_name_featured = card.get('pokemon_name', 'N/A') # Usar pokemon_name
                     card_set_featured = card.get('set_name', 'N/A')
                     image_url_featured = card.get('images_large')
                     if pd.notna(image_url_featured): st.image(image_url_featured, width=150, caption=card_set_featured)
                     else: st.warning("Imagen no disponible"); st.caption(f"{card_name_featured} ({card_set_featured})")
             st.markdown("---")
        elif special_illustration_rares.empty: logger.info(f"FEATURED_CARDS: No se encontraron cartas con rareza '{FEATURED_RARITY}'.")
    if special_illustration_rares.empty and results_df.empty and is_initial_unfiltered_load and bq_client and LATEST_SNAPSHOT_TABLE:
         st.info("No se encontraron cartas con la rareza destacada o con precio en la base de datos actual.")

elif not is_initial_unfiltered_load:
    st.header("Resultados de Cartas")
    logger.info(f"AGGRID_RENDERING: ID en session_state ANTES de AgGrid: {st.session_state.get('selected_card_id_from_grid')}")
    results_df_for_aggrid_display = results_df
    if len(results_df) > MAX_ROWS_NO_FILTER:
        st.info(f"Mostrando los primeros {MAX_ROWS_NO_FILTER} de {len(results_df)} resultados. Aplica filtros.")
        results_df_for_aggrid_display = results_df.head(MAX_ROWS_NO_FILTER)
    grid_response = None
    if not results_df_for_aggrid_display.empty:
        display_columns_mapping = {'id': 'ID', 'pokemon_name': 'Nombre Carta', 'supertype': 'Categoría', 'set_name': 'Set', 'rarity': 'Rareza', 'artist_name': 'Artista', 'precio': 'Precio (Trend €)'} # Usar artist_name
        cols_in_df_for_display = [col for col in display_columns_mapping.keys() if col in results_df_for_aggrid_display.columns]
        final_display_df_aggrid = results_df_for_aggrid_display[cols_in_df_for_display].copy()
        final_display_df_aggrid.rename(columns=display_columns_mapping, inplace=True)
        price_display_col_name_in_aggrid = display_columns_mapping.get('Precio (Trend €)') # Ajustado
        if price_display_col_name_in_aggrid and price_display_col_name_in_aggrid in final_display_df_aggrid.columns:
             final_display_df_aggrid[price_display_col_name_in_aggrid] = final_display_df_aggrid[price_display_col_name_in_aggrid].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")
        gb = GridOptionsBuilder.from_dataframe(final_display_df_aggrid)
        gb.configure_selection(selection_mode='single', use_checkbox=False); gb.configure_grid_options(domLayout='normal')
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
        gridOptions = gb.build()
        st.write("Haz clic en una fila de la tabla para ver sus detalles y opciones de predicción:")
        grid_response = AgGrid( final_display_df_aggrid, gridOptions=gridOptions, height=500, width='100%', data_return_mode=DataReturnMode.AS_INPUT, update_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=False, allow_unsafe_jscode=True, key='pokemon_aggrid_main_display_vFINAL_LGBM')
    else: logger.info("AGGRID_RENDERING: No hay datos para mostrar en AgGrid.")
    if grid_response:
        newly_selected_id_from_grid_click = None; selected_rows_data_from_grid = grid_response.get('selected_rows')
        if isinstance(selected_rows_data_from_grid, pd.DataFrame) and not selected_rows_data_from_grid.empty:
            try:
                if 'ID' in selected_rows_data_from_grid.iloc[0]: newly_selected_id_from_grid_click = selected_rows_data_from_grid.iloc[0]['ID']
            except Exception as e_aggrid_df: logger.error(f"AGGRID_HANDLER_DF: Error: {e_aggrid_df}", exc_info=True); newly_selected_id_from_grid_click = None
        elif isinstance(selected_rows_data_from_grid, list) and selected_rows_data_from_grid:
            try:
                if isinstance(selected_rows_data_from_grid[0], dict): newly_selected_id_from_grid_click = selected_rows_data_from_grid[0].get('ID')
            except Exception as e_aggrid_list: logger.error(f"AGGRID_HANDLER_LIST: Error: {e_aggrid_list}", exc_info=True); newly_selected_id_from_grid_click = None
        current_id_in_session = st.session_state.get('selected_card_id_from_grid')
        if newly_selected_id_from_grid_click is not None and newly_selected_id_from_grid_click != current_id_in_session:
            st.session_state.selected_card_id_from_grid = newly_selected_id_from_grid_click; st.rerun()

# --- Sección de Detalle de Carta Seleccionada y Predicción ---
if st.session_state.selected_card_id_from_grid is not None:
    st.divider(); st.header("Detalle de Carta Seleccionada")
    card_to_display_in_detail_section = None
    id_for_detail_view_from_session = st.session_state.get('selected_card_id_from_grid')
    source_df_for_details_primary = results_df if not results_df.empty and id_for_detail_view_from_session in results_df['id'].values else all_card_metadata_df
    if not source_df_for_details_primary.empty:
        matched_rows = source_df_for_details_primary[source_df_for_details_primary['id'] == id_for_detail_view_from_session]
        if not matched_rows.empty: card_to_display_in_detail_section = matched_rows.iloc[0]
        else: st.session_state.selected_card_id_from_grid = None; st.rerun()
    else: st.error("Error interno: No se cargaron los datos de metadatos."); st.stop() # Detener si no hay metadatos

    if card_to_display_in_detail_section is not None:
        card_name_render = card_to_display_in_detail_section.get('pokemon_name', "N/A")
        card_set_render = card_to_display_in_detail_section.get('set_name', "N/A")
        card_image_url_render = card_to_display_in_detail_section.get('image_url', None)
        card_supertype_render = card_to_display_in_detail_section.get('supertype', "N/A")
        card_rarity_render = card_to_display_in_detail_section.get('rarity', "N/A")
        card_artist_render = card_to_display_in_detail_section.get('artist_name', None) # Usar artist_name
        card_price_actual_render = card_to_display_in_detail_section.get('precio', None) # Usar precio
        # ... (resto de la sección de detalle, pero ahora con el botón de predicción LGBM)
        col_img, col_info = st.columns([1, 2])
        with col_img:
            if pd.notna(card_image_url_render): st.image(card_image_url_render, caption=f"{card_name_render} ({card_set_render})", width=300)
            else: st.warning("Imagen no disponible.")
            # ... (links HTML)
        with col_info:
            st.subheader(f"{card_name_render}")
            # ... (markdowns de detalle)
            if pd.notna(card_price_actual_render): st.metric(label="Precio Actual (Trend €)", value=f"€{card_price_actual_render:.2f}")
            else: st.markdown("**Precio Actual (Trend €):** N/A")
            st.markdown("---"); st.subheader("Predicción de Precio (Modelo LGBM Estimado)")
            if pipe_low_lgbm_app and pipe_high_lgbm_app and threshold_lgbm_app is not None:
                if pd.notna(card_price_actual_render) and pd.notna(card_to_display_in_detail_section.get('cm_avg7')): # Necesitamos cm_avg7 para el umbral
                     if st.button("⚡ Estimar Precio Futuro (LGBM)", key=f"predict_lgbm_btn_{card_id_render}"):
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
    elif results_df.empty and is_initial_unfiltered_load and not all_card_metadata_df.empty and all_card_metadata_df[all_card_metadata_df['rarity'] == FEATURED_RARITY].empty:
        st.info("No se encontraron cartas destacadas ni otros resultados iniciales.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Pokémon TCG Explorer v1.9 | LGBM")
