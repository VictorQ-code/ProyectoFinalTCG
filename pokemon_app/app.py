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
except KeyError:
    st.error("Error Crítico: Configuración de GCP no encontrada en secrets.toml."); st.stop()

BIGQUERY_DATASET = "pokemon_dataset"
CARD_METADATA_TABLE = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.card_metadata"
MAX_ROWS_NO_FILTER = 200

# --- RUTAS Y NOMBRES DE ARCHIVOS DE MODELOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES_DIR = os.path.join(BASE_DIR, "model_files")
MLP_ARTIFACTS_SUBDIR = "mlp_v1"
MLP_SAVED_MODEL_PATH = os.path.join(MODEL_FILES_DIR, MLP_ARTIFACTS_SUBDIR)
MLP_OHE_PKL_FILENAME = "ohe_mlp_cat.pkl"
MLP_SCALER_PKL_FILENAME = "scaler_mlp_num.pkl"
MLP_OHE_PATH = os.path.join(MLP_SAVED_MODEL_PATH, MLP_OHE_PKL_FILENAME)
MLP_SCALER_PATH = os.path.join(MLP_SAVED_MODEL_PATH, MLP_SCALER_PKL_FILENAME)
LGBM_MODELS_SUBDIR = os.path.join(MODEL_FILES_DIR, "lgbm_models")
PIPE_LOW_PKL_FILENAME = "modelo_pipe_low.pkl"
PIPE_HIGH_PKL_FILENAME = "modelo_pipe_high.pkl"
THRESHOLD_JSON_FILENAME = "threshold.json"
PIPE_LOW_LGBM_PATH = os.path.join(LGBM_MODELS_SUBDIR, PIPE_LOW_PKL_FILENAME)
PIPE_HIGH_LGBM_PATH = os.path.join(LGBM_MODELS_SUBDIR, PIPE_HIGH_PKL_FILENAME)
THRESHOLD_LGBM_PATH = os.path.join(LGBM_MODELS_SUBDIR, THRESHOLD_JSON_FILENAME)

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

# --- FUNCIONES DE CARGA DE ARTEFACTOS ---
# (Pega aquí las definiciones de load_tf_model_as_layer, load_sklearn_pipeline, load_joblib_preprocessor, load_json_config de la v1.22)
# ...

# --- Carga de TODOS los Modelos y Preprocesadores ---
# ... (Pega aquí la carga de todos los modelos de la v1.22) ...

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
        return None, None # Asegurar que devuelve tupla
    except Exception as e:
        logger.error(f"SNAPSHOT_TABLE: Error: {e}", exc_info=True)
        st.error(f"Error crítico al obtener información del último snapshot: {e}")
        return None, None # Asegurar que devuelve tupla

# ... (Pega aquí get_true_base_name y get_card_metadata_with_base_names de v1.22) ...

@st.cache_data(ttl=600)
def fetch_card_data_from_bq(
    _client: bigquery.Client, latest_table_path_param: str, snapshot_date_param: pd.Timestamp,
    supertype_ui_filter: str | None, sets_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
    sort_direction: str, full_metadata_df_param: pd.DataFrame
) -> pd.DataFrame:
    logger.info(f"FETCH_BQ_DATA: Ini. SType:{supertype_ui_filter}, Sets:{len(sets_ui_filter)}, Names:{len(names_ui_filter)}, Rars:{len(rarities_ui_filter)}")
    if not latest_table_path_param:
        logger.error("FETCH_BQ_DATA_FAIL: 'latest_table_path_param' es None.")
        st.error("Error Interno: No se pudo determinar la tabla de precios.")
        return pd.DataFrame() # Devolver DataFrame vacío
    # ... (resto de la lógica de pre-filtrado de IDs sin cambios) ...
    ids_to_query_df = full_metadata_df_param.copy()
    if supertype_ui_filter and supertype_ui_filter != "Todos": ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]
    if rarities_ui_filter: ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:
        name_col_to_use_for_filter = 'base_pokemon_name_display' if supertype_ui_filter == 'Pokémon' and 'base_pokemon_name_display' in ids_to_query_df.columns else 'name'
        if name_col_to_use_for_filter in ids_to_query_df.columns: ids_to_query_df = ids_to_query_df[ids_to_query_df[name_col_to_use_for_filter].isin(names_ui_filter)]
    if ids_to_query_df.empty: logger.info("FETCH_BQ_DATA: No hay IDs que coincidan."); return pd.DataFrame()
    list_of_card_ids_to_query = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids_to_query: logger.info("FETCH_BQ_DATA: Lista IDs vacía."); return pd.DataFrame()

    snapshot_date_str_for_query = snapshot_date_param.strftime('%Y-%m-%d')
    query_sql_template = f"""
    SELECT
        meta.id,
        meta.name,                         -- Columna original 'name'
        meta.name AS pokemon_name,         -- Alias para modelos
        meta.supertype,
        meta.subtypes,
        meta.types,
        meta.set_name,
        meta.rarity,
        meta.artist,                       -- Columna original 'artist'
        meta.artist AS artist_name,        -- Alias para modelos
        meta.images_large AS image_url,
        meta.cardmarket_url,
        meta.tcgplayer_url,
        prices.cm_averageSellPrice AS price,
        prices.cm_trendPrice,
        prices.cm_avg1,
        prices.cm_avg7,
        prices.cm_avg30,
        DATE('{snapshot_date_str_for_query}') AS fecha_snapshot
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
        price_cols_to_convert = ['price', 'cm_trendPrice', 'cm_avg1', 'cm_avg7', 'cm_avg30']
        for pcol in price_cols_to_convert:
            if pcol in results_from_bq_df.columns:
                 results_from_bq_df[pcol] = pd.to_numeric(results_from_bq_df[pcol], errors='coerce')
        results_from_bq_df['days_since_prev_snapshot'] = 30.0
        logger.info("FETCH_BQ_DATA: Añadida columna 'days_since_prev_snapshot'.")
        logger.info(f"FETCH_BQ_DATA: Consulta a BQ OK. Filas: {len(results_from_bq_df)}.")
        logger.debug(f"FETCH_BQ_DATA: Columnas en results_df: {results_from_bq_df.columns.tolist()}")
        return results_from_bq_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): logger.error("FETCH_BQ_DATA_FAIL: Error de 'db-dtypes'.", exc_info=True); st.error("Error de Dependencia: Falta 'db-dtypes'.")
        else: logger.error(f"FETCH_BQ_DATA_FAIL: Error BQ: {e}", exc_info=True); st.error(f"Error al obtener datos de cartas: {e}.")
        return pd.DataFrame() # <-- ASEGURAR DEVOLVER DataFrame VACÍO

# --- FUNCIONES DE PREDICCIÓN ---
# ... (Pega aquí las definiciones de predict_price_with_mlp y predict_price_with_lgbm_pipelines_app de la v1.22)

# --- Carga de Datos Inicial de BigQuery ---
logger.info("APP_INIT: Cargando datos iniciales de BigQuery.")
LATEST_SNAPSHOT_TABLE_PATH, LATEST_SNAPSHOT_DATE = get_latest_snapshot_info(bq_client)
all_card_metadata_df = get_card_metadata_with_base_names(bq_client)

if not LATEST_SNAPSHOT_TABLE_PATH or LATEST_SNAPSHOT_DATE is None or all_card_metadata_df.empty:
    logger.critical("APP_INIT_FAIL: Datos esenciales de BigQuery no cargados.")
    st.error("Error Crítico: No se pudieron cargar los datos esenciales de BigQuery.")
    st.stop()
logger.info("APP_INIT: Datos iniciales de BigQuery cargados OK.")

# --- Sidebar y Filtros ---
st.sidebar.header("Filtros y Opciones")
options_df_for_filters = all_card_metadata_df.copy() # Usar metadatos completos para opciones de filtro
supertype_options_list = sorted(options_df_for_filters['supertype'].dropna().unique().tolist())
select_supertype_options = ["Todos"] + supertype_options_list if supertype_options_list else ["Todos"]
selected_supertype = st.sidebar.selectbox("Categoría:", select_supertype_options, index=0, key="sb_supertype_filter_v3")
if selected_supertype != "Todos": options_df_for_filters = options_df_for_filters[options_df_for_filters['supertype'] == selected_supertype]
set_options_list = sorted(options_df_for_filters['set_name'].dropna().unique().tolist())
selected_sets = st.sidebar.multiselect("Set(s):", set_options_list, key="ms_sets_filter_v3")
if selected_sets: options_df_for_filters = options_df_for_filters[options_df_for_filters['set_name'].isin(selected_sets)]
# Para el filtro de nombre en la UI, usamos 'name' o 'base_pokemon_name_display'
name_label = "Nombre de Carta:"; name_col_for_options_ui = 'name' # Usar 'name' para el filtro UI
if selected_supertype == 'Pokémon': name_col_for_options_ui = 'base_pokemon_name_display'; name_label = "Pokémon (Nombre Base):"
elif selected_supertype != "Todos": name_label = f"Nombre ({selected_supertype}):"

if name_col_for_options_ui in options_df_for_filters.columns: name_options_list = sorted(options_df_for_filters[name_col_for_options_ui].dropna().unique().tolist())
else: name_options_list = []; logger.warning(f"SIDEBAR: Columna '{name_col_for_options_ui}' no para filtro nombre.")
selected_names_to_filter = st.sidebar.multiselect(name_label, name_options_list, key="ms_names_filter_v3")
if selected_names_to_filter and name_col_for_options_ui in options_df_for_filters.columns: options_df_for_filters = options_df_for_filters[options_df_for_filters[name_col_for_options_ui].isin(selected_names_to_filter)]
rarity_options_list = sorted(options_df_for_filters['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Rareza(s):", rarity_options_list, key="ms_rarities_filter_v3")
sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1, key="rd_sort_order_v3")
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"

# --- Carga de results_df ---
logger.info("MAIN_APP: Fetcheando resultados principales de BigQuery (basado en filtros de sidebar).")
results_df = fetch_card_data_from_bq( # Esta función ahora devuelve 'name', 'artist', Y TAMBIÉN 'pokemon_name', 'artist_name'
    bq_client, LATEST_SNAPSHOT_TABLE_PATH, LATEST_SNAPSHOT_DATE,
    selected_supertype, selected_sets, selected_names_to_filter, selected_rarities,
    sort_sql, all_card_metadata_df
)
logger.info(f"MAIN_APP: 'results_df' cargado con {len(results_df)} filas (reflejando filtros).")

# --- Inicializar session_state y Lógica de Fallback ---
if 'selected_card_id_from_grid' not in st.session_state: st.session_state.selected_card_id_from_grid = None
if st.session_state.selected_card_id_from_grid is None and not results_df.empty:
    cards_with_price = results_df[pd.notna(results_df['price'])] # 'price' es el nombre correcto
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
             display_cards_df = special_illustration_rares.sample(n=num_cards_to_show, random_state=None).reset_index(drop=True)
        else: display_cards_df = pd.DataFrame()
        if not display_cards_df.empty:
             cols = st.columns(num_cards_to_show)
             for i, card in display_cards_df.iterrows():
                 if i < len(cols):
                     with cols[i]:
                         card_name_featured = card.get('name', 'N/A') # Usar 'name' para mostrar
                         card_set_featured = card.get('set_name', 'N/A')
                         image_url_featured = card.get('image_url')
                         if pd.notna(image_url_featured): st.image(image_url_featured, width=150, caption=card_set_featured)
                         else: st.warning("Imagen no disp."); st.caption(f"{card_name_featured} ({card_set_featured})")
             st.markdown("---")
    if special_illustration_rares.empty and results_df.empty and is_initial_unfiltered_load and bq_client and LATEST_SNAPSHOT_TABLE_PATH:
         st.info("No se encontraron cartas con la rareza destacada o con precio en la base de datos actual.")

elif not is_initial_unfiltered_load:
    st.header("Resultados de Cartas")
    results_df_for_aggrid_display = results_df
    if len(results_df) > MAX_ROWS_NO_FILTER :
        st.info(f"Mostrando los primeros {MAX_ROWS_NO_FILTER} de {len(results_df)} resultados. Aplica filtros.")
        results_df_for_aggrid_display = results_df.head(MAX_ROWS_NO_FILTER)
    if not results_df_for_aggrid_display.empty:
        # AgGrid usa 'name' y 'artist' de results_df para mostrar
        display_columns_mapping = {'id': 'ID', 'name': 'Nombre Carta', 'supertype': 'Categoría', 'set_name': 'Set', 'rarity': 'Rareza', 'artist': 'Artista', 'price': 'Precio (€)'}
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
        grid_response = AgGrid( final_display_df_aggrid, gridOptions=gridOptions, height=500, width='100%', data_return_mode=DataReturnMode.AS_INPUT, update_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=False, allow_unsafe_jscode=True, key='pokemon_aggrid_main_display_vFINAL_LGBM_5') # Nueva key
        if grid_response:
            selected_rows_data = grid_response.get('selected_rows')
            if isinstance(selected_rows_data, list) and selected_rows_data:
                try:
                    if isinstance(selected_rows_data[0], dict):
                        newly_selected_id = selected_rows_data[0].get('ID')
                        current_id = st.session_state.get('selected_card_id_from_grid')
                        if newly_selected_id is not None and newly_selected_id != current_id:
                            st.session_state.selected_card_id_from_grid = newly_selected_id
                            st.rerun()
                except Exception as e_ag: logger.error(f"AGGRID_HANDLER_ERR: {e_ag}", exc_info=True)
            elif isinstance(selected_rows_data, pd.DataFrame) and not selected_rows_data.empty:
                try:
                    newly_selected_id = selected_rows_data.iloc[0]['ID']
                    current_id = st.session_state.get('selected_card_id_from_grid')
                    if newly_selected_id is not None and newly_selected_id != current_id:
                        st.session_state.selected_card_id_from_grid = newly_selected_id
                        st.rerun()
                except Exception as e_ag_df: logger.error(f"AGGRID_HANDLER_DF_ERR: {e_ag_df}", exc_info=True)
    else: st.info("No hay cartas que coincidan con los filtros aplicados.")

# --- Sección de Detalle de Carta Seleccionada y Predicción ---
if st.session_state.selected_card_id_from_grid is not None:
    st.divider(); st.header("Detalle de Carta Seleccionada")
    card_to_display_in_detail_section = None
    id_for_detail_view = st.session_state.selected_card_id_from_grid
    # source_df_for_details ahora SIEMPRE tendrá 'name', 'artist', 'pokemon_name', 'artist_name'
    source_df_for_details = results_df if (not results_df.empty and id_for_detail_view in results_df['id'].values) else all_card_metadata_df

    if not source_df_for_details.empty:
        matched_rows = source_df_for_details[source_df_for_details['id'] == id_for_detail_view]
        if not matched_rows.empty: card_to_display_in_detail_section = matched_rows.iloc[0]
        else: st.session_state.selected_card_id_from_grid = None; st.rerun()
    else: st.error("Error: No se cargaron metadatos."); st.stop()

    if card_to_display_in_detail_section is not None:
        card_id_render = card_to_display_in_detail_section.get('id', "N/A")
        card_name_render = card_to_display_in_detail_section.get('name', "N/A") # Para UI, usar 'name'
        card_set_render = card_to_display_in_detail_section.get('set_name', "N/A")
        card_image_url_render = card_to_display_in_detail_section.get('image_url', None)
        card_supertype_render = card_to_display_in_detail_section.get('supertype', "N/A")
        card_rarity_render = card_to_display_in_detail_section.get('rarity', "N/A")
        card_artist_render = card_to_display_in_detail_section.get('artist', None) # Para UI, usar 'artist'
        card_price_actual_render = card_to_display_in_detail_section.get('price', None)
        if pd.isna(card_price_actual_render) and id_for_detail_view and not results_df.empty and id_for_detail_view in results_df['id'].values:
             price_check = results_df[results_df['id'] == id_for_detail_view]['price'].iloc[0]
             if pd.notna(price_check): card_price_actual_render = price_check;
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
            if pd.notna(card_price_actual_render): st.metric(label="Precio Actual (€)", value=f"€{card_price_actual_render:.2f}")
            else: st.markdown("**Precio Actual (€):** N/A")
            st.markdown("---"); st.subheader("Estimaciones de Precio")

            # Botón de Predicción MLP
            if mlp_model_layer and mlp_ohe and mlp_scaler:
                if pd.notna(card_price_actual_render):
                    if st.button("🔮 Estimar Precio Futuro (MLP)", key=f"predict_mlp_btn_{card_id_render}"):
                        with st.spinner("Calculando estimación futura (MLP)..."):
                            # card_to_display_in_detail_section ahora tiene 'name' y 'artist'
                            # la función predict_price_with_mlp los mapeará a 'pokemon_name' y 'artist_name'
                            pred_price_mlp = predict_price_with_mlp(mlp_model_layer, mlp_ohe, mlp_scaler, card_to_display_in_detail_section)
                        if pred_price_mlp is not None:
                            delta_mlp = pred_price_mlp - card_price_actual_render
                            delta_color_mlp = "normal" if delta_mlp < -0.01 else ("inverse" if delta_mlp > 0.01 else "off")
                            st.metric(label="Estimado Futuro (MLP)", value=f"€{pred_price_mlp:.2f}", delta=f"{delta_mlp:+.2f}€ vs Actual", delta_color=delta_color_mlp)
                        else: st.warning("No se pudo obtener estimación futura (MLP).")
                else:
                    st.info("Estimación futura (MLP) no posible sin precio actual para esta carta.")
            else:
                st.caption("Modelo MLP para estimación futura no disponible.")

            # Botón de Predicción LGBM
            if lgbm_pipeline_high or lgbm_pipeline_low:
                # Para el botón, verificar las columnas que necesita la función predict_price_with_lgbm_pipelines_app
                # ANTES de que haga su mapeo interno. Así que verificamos 'name', 'artist', 'price', _LGBM_THRESHOLD_COLUMN_APP y las categóricas.
                required_lgbm_button_cols = ['price', _LGBM_THRESHOLD_COLUMN_APP] + \
                                            ['name', 'artist'] + \
                                            [col for col in _LGBM_CAT_FEATURES_INPUT if col not in ['pokemon_name', 'artist_name']]
                required_lgbm_button_cols = list(set(required_lgbm_button_cols))


                can_predict_lgbm = all(
                    col in card_to_display_in_detail_section.index and \
                    pd.notna(card_to_display_in_detail_section.get(col)) for col in required_lgbm_button_cols
                )

                if can_predict_lgbm:
                     if st.button("⚖️ Calcular Precio Justo (LGBM)", key=f"predict_lgbm_btn_{card_id_render}"):
                         with st.spinner("Calculando precio justo (LGBM)..."):
                             # predict_price_with_lgbm_pipelines_app mapeará 'name'->'pokemon_name', 'artist'->'artist_name' internamente
                             pred_price_lgbm, pipeline_lgbm_used = predict_price_with_lgbm_pipelines_app(
                                 lgbm_pipeline_low, lgbm_pipeline_high, lgbm_threshold_value,
                                 card_to_display_in_detail_section
                             )
                         if pred_price_lgbm is not None and card_price_actual_render is not None:
                             delta_lgbm = pred_price_lgbm - card_price_actual_render
                             st.metric(label=f"Precio Justo Estimado ({pipeline_lgbm_used})", value=f"€{pred_price_lgbm:.2f}", delta=f"{delta_lgbm:+.2f}€ vs Actual", delta_color="off")
                         elif pred_price_lgbm is not None:
                              st.metric(label=f"Precio Justo Estimado ({pipeline_lgbm_used})", value=f"€{pred_price_lgbm:.2f}")
                         else: st.warning("No se pudo obtener el precio justo (LGBM).")
                else:
                     missing_pred_cols_lgbm = [col for col in required_lgbm_button_cols if col not in card_to_display_in_detail_section.index or pd.isna(card_to_display_in_detail_section.get(col))]
                     st.info(f"Datos insuficientes para estimación de precio justo LGBM (faltan o son NaN: {', '.join(missing_pred_cols_lgbm)}).")
            else:
                 st.caption("Modelos LGBM para estimación de precio justo no disponibles.")
else:
    if results_df.empty and not is_initial_unfiltered_load: st.info("No se encontraron cartas con los filtros seleccionados.")
    elif results_df.empty and is_initial_unfiltered_load:
        if not all_card_metadata_df.empty and not all_card_metadata_df[all_card_metadata_df['rarity'] == FEATURED_RARITY].empty: pass
        else: st.info("No se encontraron cartas destacadas ni otros resultados iniciales.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Pokémon TCG Explorer v1.22 | MLP & LGBM")
