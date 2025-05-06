import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re
import logging
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import numpy as np
import os # << AÑADIR
import tensorflow as tf # << AÑADIR
import joblib # << AÑADIR (o import pickle)

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Pokémon TCG Explorer")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Constantes y Configuración de GCP ---
# ... (tu código existente aquí sin cambios) ...
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
BQML_MODEL_NAME = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.mlp_price_predictor" # Lo mantendremos por si aún lo usas
MAX_ROWS_NO_FILTER = 200

# --- RUTAS Y NOMBRES DE ARCHIVOS DEL MODELO LOCAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_files") # Carpeta contenedora

TF_SAVED_MODEL_PATH = MODEL_ARTIFACTS_DIR # Directorio del SavedModel
OHE_PKL_FILENAME = "ohe_mlp_cat.pkl"
SCALER_PKL_FILENAME = "scaler_mlp_num.pkl"
OHE_PATH = os.path.join(MODEL_ARTIFACTS_DIR, OHE_PKL_FILENAME)
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, SCALER_PKL_FILENAME)

# Nombres de columnas (DEBES AJUSTARLOS A TU ENTRENAMIENTO)
# Ejemplo:
NUMERICAL_COLS_FOR_MODEL = ['feature1_num', 'feature2_num'] # REEMPLAZA con tus nombres de columnas numéricas
CATEGORICAL_COLS_FOR_MODEL = ['feature1_cat', 'feature2_cat'] # REEMPLAZA con tus nombres de columnas categóricas
# TARGET_COL_LOG_TRANSFORMED = True # Si tu target fue transformado (ej. log) para el entrenamiento

# --- Conexión Segura a BigQuery ---
# ... (tu código existente aquí sin cambios) ...
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
if bq_client is None: st.stop()


# --- FUNCIONES DE CARGA DE MODELO Y PREPROCESADORES ---
@st.cache_resource
def load_local_tf_model(model_path):
    if not os.path.exists(os.path.join(model_path, "saved_model.pb")):
        logger.error(f"LOAD_TF_MODEL: 'saved_model.pb' no encontrado en {model_path}")
        st.error(f"Error: El modelo TensorFlow no se encuentra en la ruta esperada: {model_path}. Asegúrate de que la carpeta 'model_files' exista y contenga 'saved_model.pb', 'variables/', etc.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"LOAD_TF_MODEL: Modelo TensorFlow cargado desde: {model_path}")
        return model
    except Exception as e:
        logger.error(f"LOAD_TF_MODEL: Error cargando modelo TF desde {model_path}: {e}", exc_info=True)
        st.error(f"Error al cargar el modelo TensorFlow: {e}")
        return None

@st.cache_resource
def load_local_preprocessor(file_path, preprocessor_name="Preprocessor"):
    if not os.path.exists(file_path):
        logger.error(f"LOAD_PREPROC: {preprocessor_name} no encontrado en {file_path}")
        st.error(f"Error: El archivo {preprocessor_name} no se encuentra en: {file_path}")
        return None
    try:
        preprocessor = joblib.load(file_path) # Asumimos joblib
        logger.info(f"LOAD_PREPROC: {preprocessor_name} cargado desde: {file_path}")
        return preprocessor
    except Exception as e:
        logger.error(f"LOAD_PREPROC: Error cargando {preprocessor_name} desde {file_path}: {e}", exc_info=True)
        st.error(f"Error al cargar {preprocessor_name}: {e}")
        return None

# --- Carga de Modelo y Preprocesadores ---
logger.info("APP_INIT: Cargando modelo local y preprocesadores.")
local_mlp_model = load_local_tf_model(TF_SAVED_MODEL_PATH)
ohe_local = load_local_preprocessor(OHE_PATH, "OneHotEncoder")
scaler_local = load_local_preprocessor(SCALER_PATH, "Scaler")

# --- Funciones Auxiliares (tu código existente) ---
# ... (tu código get_latest_snapshot_table, get_true_base_name, get_card_metadata_with_base_names) ...
@st.cache_data(ttl=3600)
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    query = f"SELECT table_id FROM `{_client.project}.{BIGQUERY_DATASET}`.__TABLES__ WHERE STARTS_WITH(table_id, 'monthly_') ORDER BY table_id DESC LIMIT 1"
    try:
        results = _client.query(query).result()
        if results.total_rows > 0:
            latest_table_id = list(results)[0].table_id
            logger.info(f"SNAPSHOT_TABLE: Usando: {latest_table_id}")
            return f"{_client.project}.{BIGQUERY_DATASET}.{latest_table_id}"
        logger.warning(f"SNAPSHOT_TABLE: No se encontraron tablas 'monthly_...'.")
        st.warning(f"No se encontraron tablas 'monthly_...'.")
        return None
    except Exception as e:
        logger.error(f"SNAPSHOT_TABLE: Error: {e}", exc_info=True)
        st.error(f"Error buscando tabla snapshot: {e}.")
        return None

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
    query = f"""
    SELECT
        id, name, supertype, subtypes, rarity, set_id, set_name,
        artist, images_large, cardmarket_url, tcgplayer_url
    FROM `{CARD_METADATA_TABLE}`
    """
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            logger.warning("METADATA: DataFrame vacío de BQ.")
            st.warning("No se pudo cargar metadatos.")
            return pd.DataFrame()

        if 'cardmarket_url' not in df.columns:
            df['cardmarket_url'] = None
            logger.warning("METADATA: Columna 'cardmarket_url' no encontrada, añadida como None.")
        if 'tcgplayer_url' not in df.columns:
            df['tcgplayer_url'] = None
            logger.warning("METADATA: Columna 'tcgplayer_url' no encontrada, añadida como None.")

        df['base_pokemon_name'] = df.apply(lambda row: get_true_base_name(row['name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES), axis=1)
        logger.info(f"METADATA: Cargados {len(df)} metadatos.")
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("METADATA: Error 'db-dtypes'.", exc_info=True); st.error("Error: Falta 'db-dtypes'.")
        else: logger.error(f"METADATA: Error: {e}", exc_info=True); st.error(f"Error metadatos: {e}.")
        return pd.DataFrame()

# --- FUNCIÓN DE PREDICCIÓN CON MODELO LOCAL ---
def predict_price_with_local_model(
    model, # El modelo TensorFlow cargado
    ohe,   # El OneHotEncoder cargado
    scaler, # El Scaler cargado
    card_data_series: pd.Series # Una Series de Pandas con los datos de la carta
) -> float | None:
    """
    Preprocesa los datos de una carta y realiza una predicción con el modelo local.
    ¡DEBES ADAPTAR ESTA FUNCIÓN A LAS CARACTERÍSTICAS EXACTAS DE TU MODELO!
    """
    if not model or not ohe or not scaler:
        logger.error("PREDICT_LOCAL: Modelo o preprocesadores no cargados.")
        return None

    try:
        # 1. Crear DataFrame para preprocesamiento (con una sola fila)
        #    Usa los nombres de columnas que usaste en el ENTRENAMIENTO.
        #    Asegúrate de que card_data_series contenga estos campos.
        #    EJEMPLO: Debes mapear los datos de 'card_data_series' a las columnas esperadas.
        #             Si 'card_data_series' tiene 'rarity' y tu modelo espera 'card_rarity':
        #             current_input_dict = {
        #                 'card_rarity': card_data_series.get('rarity'),
        #                 'set_name_feature': card_data_series.get('set_name'),
        #                 'current_price_log': np.log(card_data_series.get('price')) if card_data_series.get('price') > 0 else 0,
        #                 # ... otras características ...
        #             }
        #    DEBES CONSTRUIR ESTE DICCIONARIO O DATAFRAME CUIDADOSAMENTE.
        #    Por ahora, asumiré que NUMERICAL_COLS_FOR_MODEL y CATEGORICAL_COLS_FOR_MODEL
        #    contienen los nombres correctos y que card_data_series tiene esos campos.

        input_dict = {}
        # Mapeo de ejemplo (¡¡¡DEBES AJUSTAR ESTO!!!)
        # Asume que card_data_series tiene 'price', 'rarity', 'set_name'
        # Y tu modelo espera 'log_price_t0', 'rarity_model', 'set_name_model'
        # Esto es solo un ejemplo para ilustrar el mapeo.
        required_model_input_fields = NUMERICAL_COLS_FOR_MODEL + CATEGORICAL_COLS_FOR_MODEL
        
        # Ejemplo de cómo podrías extraer features de card_data_series
        # Esto es MUY ESPECÍFICO A TU MODELO Y DATOS
        # NECESITAS REVISAR CÓMO PREPARASTE TUS DATOS PARA EL ENTRENAMIENTO
        data_for_df = {}
        if 'price' in card_data_series and pd.notna(card_data_series['price']) and card_data_series['price'] > 0:
            data_for_df['log_price_t0'] = np.log(card_data_series['price']) # Ejemplo de feature numérica
        else:
            data_for_df['log_price_t0'] = 0 # O algún valor de imputación si es necesario

        data_for_df['rarity_model_input'] = card_data_series.get('rarity', 'Unknown') # Ejemplo de feature categórica
        data_for_df['set_name_model_input'] = card_data_series.get('set_name', 'Unknown') # Ejemplo de feature categórica
        # ... añade más mapeos según las features que tu modelo espera.

        # REEMPLAZA NUMERICAL_COLS_FOR_MODEL y CATEGORICAL_COLS_FOR_MODEL con los nombres
        # de las columnas que tu modelo realmente espera DESPUÉS de este mapeo.
        # Ejemplo:
        # NUMERICAL_COLS_ACTUAL = ['log_price_t0']
        # CATEGORICAL_COLS_ACTUAL = ['rarity_model_input', 'set_name_model_input']

        # Esto es una suposición, debes verificarlo:
        NUMERICAL_COLS_ACTUAL = [col for col in NUMERICAL_COLS_FOR_MODEL if col in data_for_df]
        CATEGORICAL_COLS_ACTUAL = [col for col in CATEGORICAL_COLS_FOR_MODEL if col in data_for_df]

        if not NUMERICAL_COLS_ACTUAL and not CATEGORICAL_COLS_ACTUAL:
             logger.error("PREDICT_LOCAL: No se pudieron extraer características numéricas o categóricas para el modelo.")
             st.warning("No se pudieron preparar los datos para el modelo local.")
             return None

        current_input_df = pd.DataFrame([data_for_df])
        logger.info(f"PREDICT_LOCAL: DataFrame de entrada para preprocesar: {current_input_df.to_dict()}")

        # 2. Aplicar preprocesamiento
        processed_parts = []
        if NUMERICAL_COLS_ACTUAL:
            numerical_features_scaled = scaler.transform(current_input_df[NUMERICAL_COLS_ACTUAL])
            processed_parts.append(numerical_features_scaled)
            logger.info(f"PREDICT_LOCAL: Features numéricas escaladas: {numerical_features_scaled}")

        if CATEGORICAL_COLS_ACTUAL:
            categorical_features_encoded_sparse = ohe.transform(current_input_df[CATEGORICAL_COLS_ACTUAL])
            categorical_features_encoded_dense = categorical_features_encoded_sparse.toarray() # Keras suele necesitar denso
            processed_parts.append(categorical_features_encoded_dense)
            logger.info(f"PREDICT_LOCAL: Features categóricas codificadas (shape): {categorical_features_encoded_dense.shape}")

        if not processed_parts:
            logger.error("PREDICT_LOCAL: No se procesaron características.")
            return None

        # 3. Combinar características preprocesadas
        # El orden debe ser el mismo que durante el entrenamiento
        final_input_array = np.concatenate(processed_parts, axis=1)
        logger.info(f"PREDICT_LOCAL: Array final para el modelo (shape): {final_input_array.shape}")

        # 4. Realizar Predicción
        prediction_raw = model.predict(final_input_array)
        predicted_value = prediction_raw[0][0] # Asume que el modelo devuelve [[valor]]
        logger.info(f"PREDICT_LOCAL: Predicción cruda: {prediction_raw}, valor extraído: {predicted_value}")

        # 5. Postprocesar predicción (si es necesario, ej. si predice logaritmo)
        # if TARGET_COL_LOG_TRANSFORMED:
        #    predicted_value = np.exp(predicted_value)
        #    logger.info(f"PREDICT_LOCAL: Predicción post-exp: {predicted_value}")

        return float(predicted_value)

    except Exception as e:
        logger.error(f"PREDICT_LOCAL: Error durante preprocesamiento o predicción local: {e}", exc_info=True)
        st.error(f"Error al usar modelo local para predicción: {e}")
        import traceback
        st.text_area("Stack Trace (Modelo Local):", traceback.format_exc(), height=150)
        return None


# --- Carga de Datos Inicial ---
# ... (tu código existente aquí sin cambios, asegúrate que LATEST_SNAPSHOT_TABLE se define) ...
logger.info("APP_INIT: Cargando datos iniciales.")
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
all_card_metadata_df = get_card_metadata_with_base_names(bq_client)
if not LATEST_SNAPSHOT_TABLE or all_card_metadata_df.empty:
    logger.critical("APP_INIT: Datos esenciales no cargados. Stop.")
    st.error("Datos esenciales no cargados. Stop.")
    st.stop()
logger.info("APP_INIT: Datos iniciales OK.")

# --- Sidebar ---
# ... (tu código existente aquí sin cambios) ...
st.title("Explorador de Cartas Pokémon TCG")
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
else: name_options_list = []; logger.warning(f"SIDEBAR: Columna '{name_col_for_options}' no encontrada.")
selected_names_to_filter = st.sidebar.multiselect(name_label, name_options_list, key="ms_names_filter_v3")
if selected_names_to_filter and name_col_for_options in options_df_for_filters.columns: options_df_for_filters = options_df_for_filters[options_df_for_filters[name_col_for_options].isin(selected_names_to_filter)]
rarity_options_list = sorted(options_df_for_filters['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Rareza(s):", rarity_options_list, key="ms_rarities_filter_v3")
sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1, key="rd_sort_order_v3")
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"

# --- Fetch Card Data (tu código existente) ---
# ... (tu código fetch_card_data sin cambios) ...
@st.cache_data(ttl=600)
def fetch_card_data(_client: bigquery.Client, latest_table_path: str, supertype_ui_filter: str | None,
                    sets_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
                    sort_direction: str, full_metadata_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"FETCH_DATA: Ini. SType:{supertype_ui_filter}, Sets:{len(sets_ui_filter)}, Names:{len(names_ui_filter)}, Rars:{len(rarities_ui_filter)}")
    ids_to_query_df = full_metadata_df.copy()
    if supertype_ui_filter and supertype_ui_filter != "Todos": ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]
    if rarities_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:
        actual_name_col_to_filter = 'base_pokemon_name' if supertype_ui_filter == 'Pokémon' else 'name'
        if actual_name_col_to_filter in ids_to_query_df.columns: ids_to_query_df = ids_to_query_df[ids_to_query_df[actual_name_col_to_filter].isin(names_ui_filter)]
        else: logger.warning(f"FETCH_DATA: Col '{actual_name_col_to_filter}' no existe.")
    if ids_to_query_df.empty: logger.info("FETCH_DATA: No IDs post-filter."); return pd.DataFrame()
    list_of_card_ids = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids: logger.info("FETCH_DATA: Lista IDs vacía."); return pd.DataFrame()

    query_sql = f"""
    SELECT
        meta.id, meta.name AS pokemon_name, meta.supertype, meta.set_name,
        meta.rarity, meta.artist, meta.images_large AS image_url,
        meta.cardmarket_url, meta.tcgplayer_url, prices.cm_trendPrice AS price
    FROM `{CARD_METADATA_TABLE}` AS meta
    JOIN `{latest_table_path}` AS prices ON meta.id = prices.id
    WHERE meta.id IN UNNEST(@card_ids_param)
    ORDER BY prices.cm_trendPrice {sort_direction}
    """
    query_params = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids)]
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    logger.info(f"FETCH_DATA: SQL BQ para {len(list_of_card_ids)} IDs. Orden: {sort_direction}")
    try:
        results_df = _client.query(query_sql, job_config=job_config).to_dataframe()
        results_df['price'] = pd.to_numeric(results_df['price'], errors='coerce')
        logger.info(f"FETCH_DATA: BQ OK. Filas: {len(results_df)}.")
        return results_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("FETCH_DATA: Err 'db-dtypes'.", exc_info=True); st.error("Error: Falta 'db-dtypes'.")
        else: logger.error(f"FETCH_DATA: Err BQ: {e}", exc_info=True); st.error(f"Error BQ: {e}.")
        return pd.DataFrame()

results_df = fetch_card_data(bq_client, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets, selected_names_to_filter, selected_rarities, sort_sql, all_card_metadata_df)
logger.info(f"MAIN_APP: results_df cargado con {len(results_df)} filas.")

# --- Área Principal: Visualización de Resultados (tu código AgGrid) ---
# ... (tu código AgGrid y manejo de selección existente aquí sin cambios) ...
st.header("Resultados")

if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None
    logger.info("SESSION_STATE_INIT: 'selected_card_id_from_grid' inicializado a None.")

logger.info(f"AGGRID_RENDERING: ID en session_state ANTES de AgGrid: {st.session_state.get('selected_card_id_from_grid')}")

results_df_for_aggrid_display = results_df
is_initial_unfiltered_load = (not selected_sets and not selected_names_to_filter and not selected_rarities and (selected_supertype == "Todos" or not selected_supertype))
if is_initial_unfiltered_load and len(results_df) > MAX_ROWS_NO_FILTER:
    logger.info(f"AGGRID_RENDERING: Limitando display a {MAX_ROWS_NO_FILTER} filas de {len(results_df)}.")
    st.info(f"Mostrando los primeros {MAX_ROWS_NO_FILTER} de {len(results_df)} resultados. Aplica filtros.")
    results_df_for_aggrid_display = results_df.head(MAX_ROWS_NO_FILTER)

grid_response = None
if not results_df_for_aggrid_display.empty:
    display_columns_mapping = {'id': 'ID', 'pokemon_name': 'Nombre Carta', 'supertype': 'Categoría',
                               'set_name': 'Set', 'rarity': 'Rareza', 'artist': 'Artista',
                               'price': 'Precio (Trend €)'}
    cols_in_df = [col for col in display_columns_mapping.keys() if col in results_df_for_aggrid_display.columns]
    final_display_df = results_df_for_aggrid_display[cols_in_df].copy()
    final_display_df.rename(columns=display_columns_mapping, inplace=True)
    price_display_col_name = display_columns_mapping.get('price')
    if price_display_col_name and price_display_col_name in final_display_df.columns:
         final_display_df[price_display_col_name] = final_display_df[price_display_col_name].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")

    gb = GridOptionsBuilder.from_dataframe(final_display_df)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_grid_options(domLayout='normal')
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
    gridOptions = gb.build()

    st.write("Haz clic en una fila de la tabla para ver sus detalles:")
    grid_response = AgGrid(
        final_display_df, gridOptions=gridOptions, height=500, width='100%',
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=False, allow_unsafe_jscode=True,
        key='pokemon_aggrid_static_key_v3_links', # Key estática
    )
else:
    logger.info("AGGRID_RENDERING: No hay datos para mostrar en AgGrid.")

if grid_response:
    logger.info(f"AGGRID_HANDLER_ENTRY: Verificando grid_response. Tipo de selected_rows: {type(grid_response.get('selected_rows'))}")
    newly_selected_id_from_grid = None
    selected_rows_data = grid_response.get('selected_rows')

    if isinstance(selected_rows_data, pd.DataFrame) and not selected_rows_data.empty:
        try:
            first_selected_row_series = selected_rows_data.iloc[0]
            if 'ID' in first_selected_row_series:
                newly_selected_id_from_grid = first_selected_row_series['ID']
            else: logger.warning("AGGRID_HANDLER: Columna 'ID' no encontrada en la fila seleccionada del DataFrame.")
            if newly_selected_id_from_grid: logger.info(f"AGGRID_HANDLER (DataFrame): Fila seleccionada. ID: {newly_selected_id_from_grid}")
            else: logger.warning("AGGRID_HANDLER: 'ID' es None o vacío del DataFrame."); newly_selected_id_from_grid = None
        except IndexError: logger.warning("AGGRID_HANDLER: Error de índice (DataFrame vacío inesperadamente?)."); newly_selected_id_from_grid = None
        except Exception as e: logger.error(f"AGGRID_HANDLER: Error procesando fila (DataFrame): {e}", exc_info=True); newly_selected_id_from_grid = None
    elif isinstance(selected_rows_data, list) and selected_rows_data:
        try:
            row_data = selected_rows_data[0]
            if isinstance(row_data, dict):
                newly_selected_id_from_grid = row_data.get('ID')
                if newly_selected_id_from_grid: logger.info(f"AGGRID_HANDLER (List): Fila seleccionada. ID: {newly_selected_id_from_grid}")
                else: logger.warning("AGGRID_HANDLER (List): 'ID' no encontrado o None.")
            else: logger.warning(f"AGGRID_HANDLER (List): Fila no es dict: {type(row_data)}")
        except IndexError: logger.warning("AGGRID_HANDLER (List): selected_rows vacía.")
        except Exception as e: logger.error(f"AGGRID_HANDLER (List): Error procesando fila: {e}", exc_info=True); newly_selected_id_from_grid = None
    else:
        logger.debug(f"AGGRID_HANDLER: No hay filas seleccionadas válidas.")

    current_session_id = st.session_state.get('selected_card_id_from_grid')
    logger.info(f"AGGRID_HANDLER: Current session ID: {current_session_id}, Newly selected from grid: {newly_selected_id_from_grid}")

    if newly_selected_id_from_grid is not None and newly_selected_id_from_grid != current_session_id:
        logger.info(f"AGGRID_HANDLER: DETECTADO CAMBIO DE SELECCIÓN! De '{current_session_id}' a '{newly_selected_id_from_grid}'. RE-EJECUTANDO.")
        st.session_state.selected_card_id_from_grid = newly_selected_id_from_grid
        st.rerun()
    else:
        logger.debug(f"AGGRID_HANDLER: Sin cambio de selección o nueva selección es None.")


st.divider()
st.header("Detalle de Carta")
# --- MODIFICACIÓN: Área de Detalle de Carta y Predicción ---
card_to_display_details = None
id_from_session = st.session_state.get('selected_card_id_from_grid')
logger.info(f"DETAIL_DISPLAY_ENTRY: ID para detalles (de session_state): {id_from_session}")

if id_from_session:
    if not results_df.empty:
        matched_rows_in_results_df = results_df[results_df['id'] == id_from_session]
        if not matched_rows_in_results_df.empty:
            card_to_display_details = matched_rows_in_results_df.iloc[0]
            logger.info(f"DETAIL_DISPLAY: Carta encontrada para ID '{id_from_session}'. Nombre: {card_to_display_details.get('pokemon_name')}")
        else:
            logger.warning(f"DETAIL_DISPLAY: ID '{id_from_session}' NO ENCONTRADO en results_df actual (total {len(results_df)} filas).")
    else:
        logger.warning(f"DETAIL_DISPLAY: results_df está vacío, no se puede buscar ID '{id_from_session}'.")

if card_to_display_details is None and not results_df.empty:
    card_to_display_details = results_df.iloc[0]
    fallback_id = card_to_display_details.get('id')
    logger.info(f"DETAIL_DISPLAY: Usando FALLBACK a la primera carta de results_df. ID: {fallback_id}. Nombre: {card_to_display_details.get('pokemon_name')}")
    if id_from_session is None or (fallback_id and id_from_session != fallback_id):
        if fallback_id and pd.notna(fallback_id):
            if st.session_state.get('selected_card_id_from_grid') != fallback_id :
                logger.info(f"DETAIL_DISPLAY: Actualizando session_state con ID de fallback '{fallback_id}' (era '{st.session_state.get('selected_card_id_from_grid')}').")
                st.session_state.selected_card_id_from_grid = fallback_id


if card_to_display_details is not None and isinstance(card_to_display_details, pd.Series) and not card_to_display_details.empty:
    logger.info(f"DETAIL_DISPLAY: RENDERIZANDO detalles para: {card_to_display_details.get('pokemon_name')} (ID: {card_to_display_details.get('id')})")

    card_name_detail = card_to_display_details.get('pokemon_name', "N/A")
    card_id_detail = card_to_display_details.get('id', "N/A")
    card_set_detail = card_to_display_details.get('set_name', "N/A")
    card_image_url_detail = card_to_display_details.get('image_url', None)
    card_supertype_detail = card_to_display_details.get('supertype', "N/A")
    card_rarity_detail = card_to_display_details.get('rarity', "N/A")
    card_artist_detail = card_to_display_details.get('artist', None)
    card_price_actual = card_to_display_details.get('price', None)
    card_market_url_detail = card_to_display_details.get('cardmarket_url', None)
    tcgplayer_url_detail = card_to_display_details.get('tcgplayer_url', None)

    col1, col2 = st.columns([1, 2])
    with col1:
        if pd.notna(card_image_url_detail):
            st.image(card_image_url_detail, caption=f"{card_name_detail} - {card_set_detail}", width=300)
        else:
            st.warning("Imagen no disponible.")

        links_html_parts = []
        if pd.notna(card_market_url_detail) and isinstance(card_market_url_detail, str) and card_market_url_detail.startswith("http"):
            links_html_parts.append(f"<a href='{card_market_url_detail}' target='_blank' style='display: inline-block; margin-top: 5px; margin-right: 10px; padding: 5px 10px; background-color: #FFCB05; color: #2a75bb; text-align: center; border-radius: 5px; text-decoration: none; font-weight: bold;'>Cardmarket</a>")
        if pd.notna(tcgplayer_url_detail) and isinstance(tcgplayer_url_detail, str) and tcgplayer_url_detail.startswith("http"):
            links_html_parts.append(f"<a href='{tcgplayer_url_detail}' target='_blank' style='display: inline-block; margin-top: 5px; padding: 5px 10px; background-color: #007bff; color: white; text-align: center; border-radius: 5px; text-decoration: none; font-weight: bold;'>TCGplayer</a>")
        if links_html_parts:
            st.markdown(" ".join(links_html_parts), unsafe_allow_html=True)
        elif not (pd.notna(card_market_url_detail) and isinstance(card_market_url_detail, str) and card_market_url_detail.startswith("http")) and \
             not (pd.notna(tcgplayer_url_detail) and isinstance(tcgplayer_url_detail, str) and tcgplayer_url_detail.startswith("http")):
            st.caption("Links de compra no disponibles.")

    with col2:
        st.subheader(f"{card_name_detail}")
        st.markdown(f"**ID:** `{card_id_detail}`")
        st.markdown(f"**Categoría:** {card_supertype_detail}")
        st.markdown(f"**Set:** {card_set_detail}")
        st.markdown(f"**Rareza:** {card_rarity_detail}")
        if pd.notna(card_artist_detail) and card_artist_detail:
             st.markdown(f"**Artista:** {card_artist_detail}")

        if pd.notna(card_price_actual):
             st.metric(label="Precio Actual (Trend €)", value=f"€{card_price_actual:.2f}")
        else:
             st.markdown("**Precio Actual (Trend €):** N/A")

        st.markdown("---")
        st.subheader("Predicción de Precio Local (Próximo Mes)")

        # Botón para predicción con modelo local
        if local_mlp_model and ohe_local and scaler_local: # Asegurarse que los modelos locales estén cargados
            if st.button("🧠 Predecir Precio Futuro (Modelo Local)", key=f"predict_local_btn_{card_id_detail}"):
                if pd.notna(card_price_actual): # Necesitamos el precio actual para las features
                    with st.spinner("Calculando predicción con modelo local..."):
                        # Pasar la fila completa 'card_to_display_details'
                        predicted_price_local_ml = predict_price_with_local_model(
                            local_mlp_model, ohe_local, scaler_local, card_to_display_details
                        )

                    if predicted_price_local_ml is not None:
                        delta_value_local = predicted_price_local_ml - card_price_actual
                        delta_color_local = "normal"
                        if delta_value_local > 0: delta_color_local = "inverse"
                        elif delta_value_local < 0: delta_color_local = "normal"

                        st.metric(label="Precio Predicho (Modelo Local)", value=f"€{predicted_price_local_ml:.2f}",
                                  delta=f"{delta_value_local:.2f}€",
                                  delta_color=delta_color_local)
                    else:
                        st.warning("No se pudo obtener la predicción de precio desde el modelo local.")
                else:
                    st.warning("No se puede predecir con el modelo local sin un precio actual válido para usar como feature.")
        else:
            st.warning("El modelo local o los preprocesadores no están cargados. No se puede predecir.")

        # (Opcional) Mantener la predicción con BQML si aún la quieres como alternativa
        # st.subheader("Predicción de Precio BQML (Próximo Mes)")
        # if st.button("🧠 Predecir Precio Futuro (BQML)", key=f"predict_bqml_btn_{card_id_detail}"):
        #     ... (tu código de predicción BQML aquí) ...

else:
    logger.info("DETAIL_DISPLAY: No hay carta para mostrar en detalles al final.")
    st.info("Haz clic en una carta en la tabla de resultados para ver sus detalles o aplica filtros.")

if not results_df_for_aggrid_display.empty: pass
elif not results_df.empty and results_df_for_aggrid_display.empty :
    logger.info(f"DISPLAY_MSG: results_df tiene {len(results_df)} filas, pero display limitado/vacío.")
    st.info(f"Se encontraron {len(results_df)} resultados. Aplica filtros más específicos.")
else:
    logger.info("DISPLAY_MSG: results_df está vacío.")
    if bq_client and LATEST_SNAPSHOT_TABLE:
        st.info("No se encontraron cartas con los filtros seleccionados.")

st.sidebar.info("Pokémon TCG Explorer - Enlaces y Predicción")
