import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re
import logging
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Pokémon TCG Explorer")
logging.basicConfig(level=logging.INFO)

# --- Constantes y Configuración de GCP ---
try:
    GCP_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
except KeyError:
    st.error("Error: 'project_id' no encontrado en los secrets de Streamlit.")
    st.stop()
except Exception as e:
    st.error(f"Error inesperado al leer secrets: {e}")
    st.stop()

BIGQUERY_DATASET = "pokemon_dataset"
CARD_METADATA_TABLE = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.card_metadata"

# --- Conexión Segura a BigQuery ---
@st.cache_resource
def connect_to_bigquery():
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Error: Sección [gcp_service_account] no encontrada en los secrets.")
            return None
        creds_json = dict(st.secrets["gcp_service_account"])
        required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"]
        missing_keys = [key for key in required_keys if key not in creds_json or not creds_json[key]]
        if missing_keys:
             st.error(f"Error: Faltan claves en [gcp_service_account]: {', '.join(missing_keys)}")
             return None
        creds = service_account.Credentials.from_service_account_info(creds_json)
        client = bigquery.Client(credentials=creds, project=GCP_PROJECT_ID)
        logging.info("Conexión a BigQuery establecida.")
        return client
    except Exception as e:
        st.error(f"Error al conectar con BigQuery: {e}.")
        logging.error(f"Error al conectar con BigQuery: {e}", exc_info=True)
        return None

bq_client = connect_to_bigquery()
if bq_client is None:
    st.stop()

# --- Funciones Auxiliares ---
@st.cache_data(ttl=3600)
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    query = f"SELECT table_id FROM `{_client.project}.{BIGQUERY_DATASET}`.__TABLES__ WHERE STARTS_WITH(table_id, 'monthly_') ORDER BY table_id DESC LIMIT 1"
    try:
        results = _client.query(query).result()
        if results.total_rows > 0:
            latest_table_id = list(results)[0].table_id
            return f"{_client.project}.{BIGQUERY_DATASET}.{latest_table_id}"
        st.warning(f"No se encontraron tablas 'monthly_...' en '{BIGQUERY_DATASET}'.")
        return None
    except Exception as e:
        st.error(f"Error buscando tabla snapshot: {e}.")
        return None

POKEMON_SUFFIXES_TO_REMOVE = [
    ' VMAX', ' VSTAR', ' V-UNION', ' V', ' GX', ' EX', ' BREAK', ' Prism Star', ' Star',
    ' Radiant', ' δ', ' Tag Team', ' & ', ' Light', ' Dark', ' ◇', ' ☆',
]
MULTI_WORD_BASE_NAMES = ["Mr. Mime", "Mime Jr.", "Farfetch'd", "Sirfetch'd", "Ho-Oh", "Porygon-Z", "Type: Null", "Tapu Koko", "Tapu Lele", "Tapu Bulu", "Tapu Fini", "Mr. Rime", "Indeedee M", "Indeedee F"]

def get_true_base_name(name_str, supertype, suffixes, multi_word_bases):
    if not isinstance(name_str, str) or supertype != 'Pokémon':
        return name_str
    for mw_base in multi_word_bases:
        if name_str.startswith(mw_base):
            potential_base = mw_base
            remaining_part = name_str[len(mw_base):].strip()
            temp_remaining_name = remaining_part
            for suffix in suffixes:
                if temp_remaining_name.startswith(suffix.strip()):
                    temp_remaining_name = temp_remaining_name[len(suffix.strip()):].strip()
                    break
            return potential_base
    cleaned_name = name_str
    for suffix in suffixes:
        if cleaned_name.endswith(suffix):
            cleaned_name = cleaned_name[:-len(suffix)].strip()
    return cleaned_name if cleaned_name else name_str

@st.cache_data(ttl=3600)
def get_card_metadata_with_base_names(_client: bigquery.Client) -> pd.DataFrame:
    query = f"SELECT id, name, supertype, subtypes, rarity, set_id, set_name, artist, images_large FROM `{CARD_METADATA_TABLE}`"
    try:
        df = _client.query(query).to_dataframe()
        if df.empty:
            st.warning("No se pudo cargar metadatos de cartas.")
            return pd.DataFrame()
        df['base_pokemon_name'] = df.apply(
            lambda row: get_true_base_name(row['name'], row['supertype'], POKEMON_SUFFIXES_TO_REMOVE, MULTI_WORD_BASE_NAMES),
            axis=1
        )
        return df
    except Exception as e:
        if "db-dtypes" in str(e).lower():
             st.error("Error: Falta 'db-dtypes'. Añádelo a requirements.txt.")
        else:
            st.error(f"Error al obtener metadatos de cartas: {e}.")
        return pd.DataFrame()

LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
all_card_metadata_df = get_card_metadata_with_base_names(bq_client)

if not LATEST_SNAPSHOT_TABLE or all_card_metadata_df.empty:
    st.error("No se pueden cargar datos esenciales. La aplicación no puede continuar.")
    st.stop()

# --- Lógica Principal de la Aplicación Streamlit ---
st.title("Explorador de Cartas Pokémon TCG")

# --- Barra Lateral: Filtros y Controles ---
st.sidebar.header("Filtros y Opciones")
options_df_for_filters = all_card_metadata_df.copy()

supertype_options_list = sorted(options_df_for_filters['supertype'].dropna().unique().tolist())
select_supertype_options = ["Todos"] + supertype_options_list if supertype_options_list else ["Todos"]
selected_supertype = st.sidebar.selectbox("Categoría:", select_supertype_options, index=0, key="sb_supertype")

if selected_supertype != "Todos":
    options_df_for_filters = options_df_for_filters[options_df_for_filters['supertype'] == selected_supertype]

set_options_list = sorted(options_df_for_filters['set_name'].dropna().unique().tolist())
selected_sets = st.sidebar.multiselect("Set(s):", set_options_list, key="ms_sets")

if selected_sets:
    options_df_for_filters = options_df_for_filters[options_df_for_filters['set_name'].isin(selected_sets)]

name_label = "Nombre de Carta:"
name_col_for_options = 'name'
if selected_supertype == 'Pokémon':
    name_col_for_options = 'base_pokemon_name'
    name_label = "Pokémon (Nombre Base):"
elif selected_supertype != "Todos":
    name_label = f"Nombre ({selected_supertype}):"

name_options_list = sorted(options_df_for_filters[name_col_for_options].dropna().unique().tolist())
selected_names_to_filter = st.sidebar.multiselect(name_label, name_options_list, key="ms_names")

if selected_names_to_filter:
    options_df_for_filters = options_df_for_filters[options_df_for_filters[name_col_for_options].isin(selected_names_to_filter)]

rarity_options_list = sorted(options_df_for_filters['rarity'].dropna().unique().tolist())
selected_rarities = st.sidebar.multiselect("Rareza(s):", rarity_options_list, key="ms_rarities")

sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1, key="rd_sort")
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"


@st.cache_data(ttl=600)
def fetch_card_data(_client: bigquery.Client, latest_table_path: str, supertype_ui_filter: str | None,
                    sets_ui_filter: list, names_ui_filter: list, rarities_ui_filter: list,
                    sort_direction: str, full_metadata_df: pd.DataFrame) -> pd.DataFrame:
    ids_to_query_df = full_metadata_df.copy()
    if supertype_ui_filter and supertype_ui_filter != "Todos":
        ids_to_query_df = ids_to_query_df[ids_to_query_df['supertype'] == supertype_ui_filter]
    if sets_ui_filter:
        ids_to_query_df = ids_to_query_df[ids_to_query_df['set_name'].isin(sets_ui_filter)]
    if rarities_ui_filter:
        ids_to_query_df = ids_to_query_df[ids_to_query_df['rarity'].isin(rarities_ui_filter)]
    if names_ui_filter:
        name_col_to_filter_on = 'base_pokemon_name' if supertype_ui_filter == 'Pokémon' else 'name'
        ids_to_query_df = ids_to_query_df[ids_to_query_df[name_col_to_filter_on].isin(names_ui_filter)]

    if ids_to_query_df.empty: return pd.DataFrame()
    list_of_card_ids = ids_to_query_df['id'].unique().tolist()
    if not list_of_card_ids: return pd.DataFrame()

    query_sql = f"""
    SELECT meta.id, meta.name AS pokemon_name, meta.supertype, meta.set_name, meta.rarity, meta.artist, 
           meta.images_large AS image_url, prices.cm_trendPrice AS price
    FROM `{CARD_METADATA_TABLE}` AS meta JOIN `{latest_table_path}` AS prices ON meta.id = prices.id
    WHERE meta.id IN UNNEST(@card_ids_param) ORDER BY prices.cm_trendPrice {sort_direction}"""
    query_params = [bigquery.ArrayQueryParameter("card_ids_param", "STRING", list_of_card_ids)]
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    logging.info(f"Ejecutando consulta con {len(list_of_card_ids)} IDs. Orden: {sort_direction}")
    try:
        query_job = _client.query(query_sql, job_config=job_config)
        results_df = query_job.to_dataframe()
        results_df['price'] = pd.to_numeric(results_df['price'], errors='coerce')
        return results_df
    except Exception as e:
        if "db-dtypes" in str(e).lower(): st.error("Error: Falta 'db-dtypes'.")
        else: st.error(f"Error en consulta principal: {e}.")
        return pd.DataFrame()

results_df = fetch_card_data(bq_client, LATEST_SNAPSHOT_TABLE, selected_supertype, selected_sets,
                             selected_names_to_filter, selected_rarities, sort_sql, all_card_metadata_df)

# --- Área Principal: Visualización de Resultados ---
st.header("Resultados")

# Variable de estado para guardar el ID de la carta seleccionada en AgGrid
if 'selected_card_id_from_grid' not in st.session_state:
    st.session_state.selected_card_id_from_grid = None

if not results_df.empty:
    display_columns_mapping = {
        'id': 'ID', 'pokemon_name': 'Nombre Carta', 'supertype': 'Categoría',
        'set_name': 'Set', 'rarity': 'Rareza', 'artist': 'Artista', 'price': 'Precio (Trend €)'
    }
    actual_columns_to_display = [col for col in display_columns_mapping.keys() if col in results_df.columns]
    display_df_for_aggrid = results_df[actual_columns_to_display].copy()
    display_df_for_aggrid.rename(columns=display_columns_mapping, inplace=True)
    
    price_display_column = display_columns_mapping.get('price')
    if price_display_column and price_display_column in display_df_for_aggrid.columns:
         display_df_for_aggrid[price_display_column] = display_df_for_aggrid[price_display_column].apply(
             lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A"
         )

    gb = GridOptionsBuilder.from_dataframe(display_df_for_aggrid)
    gb.configure_selection(selection_mode='single', use_checkbox=False, pre_selected_rows=None) # No pre-selección
    # gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_grid_options(domLayout='normal') # Para control de altura
    gridOptions = gb.build()

    st.write("Haz clic en una fila de la tabla para ver sus detalles:")
    grid_response = AgGrid(
        display_df_for_aggrid,
        gridOptions=gridOptions,
        height=400, # Altura fija para la tabla
        width='100%',
        data_return_mode=DataReturnMode.AS_INPUT, # Devuelve datos como se pasaron
        update_mode=GridUpdateMode.MODEL_CHANGED, # Actualiza cuando el modelo cambia
        fit_columns_on_grid_load=False, # Evita que se ajuste al cargar, puede ser lento
        allow_unsafe_jscode=True, # Para algunas funcionalidades
        key='pokemon_results_grid', # Clave única para el componente
        # reload_data=True # Considerar si es necesario para forzar recarga
    )

    # Actualizar el estado de la carta seleccionada si se hace clic en AgGrid
    if grid_response['selected_rows']:
        # El ID original está en la columna 'ID' de display_df_for_aggrid,
        # que mapea a 'id' en results_df.
        # AgGrid devuelve la fila con los nombres de columna de display_df_for_aggrid
        selected_id_from_aggrid_display = grid_response['selected_rows'][0]['ID']
        st.session_state.selected_card_id_from_grid = selected_id_from_aggrid_display
        # st.experimental_rerun() # Forzar un rerun para actualizar la sección de detalles

    st.divider()
    st.header("Detalle de Carta")

    card_to_display_details_df_row = None
    
    # Usar el ID de la sesión si existe (seleccionado desde AgGrid)
    if st.session_state.selected_card_id_from_grid:
        # Buscar en el results_df original usando el 'id' (que es el valor de 'ID' en la tabla)
        match = results_df[results_df['id'] == st.session_state.selected_card_id_from_grid]
        if not match.empty:
            card_to_display_details_df_row = match.iloc[0]
    elif not results_df.empty: # Si no hay selección de grid pero hay resultados, tomar el primero
        card_to_display_details_df_row = results_df.iloc[0]
        # Actualizar el estado de sesión con el primero para consistencia
        if 'id' in card_to_display_details_df_row:
             st.session_state.selected_card_id_from_grid = card_to_display_details_df_row['id']


    if card_to_display_details_df_row is not None:
        card_name_detail = card_to_display_details_df_row.get('pokemon_name', "N/A")
        card_id_detail = card_to_display_details_df_row.get('id', "N/A")
        card_set_detail = card_to_display_details_df_row.get('set_name', "N/A")
        card_image_url_detail = card_to_display_details_df_row.get('image_url', None)
        card_supertype_detail = card_to_display_details_df_row.get('supertype', "N/A")
        card_rarity_detail = card_to_display_details_df_row.get('rarity', "N/A")
        card_artist_detail = card_to_display_details_df_row.get('artist', None)
        card_price_detail = card_to_display_details_df_row.get('price', None)

        col1, col2 = st.columns([1, 2])
        with col1:
            if pd.notna(card_image_url_detail):
                st.image(card_image_url_detail, caption=f"{card_name_detail} - {card_set_detail}", width=300)
            else:
                st.warning("Imagen no disponible.")
        with col2:
            st.subheader(f"{card_name_detail}")
            st.markdown(f"**ID:** `{card_id_detail}`")
            st.markdown(f"**Categoría:** {card_supertype_detail}")
            st.markdown(f"**Set:** {card_set_detail}")
            st.markdown(f"**Rareza:** {card_rarity_detail}")
            if pd.notna(card_artist_detail) and card_artist_detail:
                 st.markdown(f"**Artista:** {card_artist_detail}")
            if pd.notna(card_price_detail):
                 st.metric(label="Precio (Trend €)", value=f"€{card_price_detail:.2f}")
            else:
                 st.markdown("**Precio (Trend €):** N/A")
    else:
        st.info("Haz clic en una carta en la tabla de resultados para ver sus detalles.")

else:
    if bq_client and LATEST_SNAPSHOT_TABLE:
        st.info("No se encontraron cartas con los filtros seleccionados.")

st.sidebar.info("Pokémon TCG Explorer v3 (AgGrid)")
