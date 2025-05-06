import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import json # Aunque no se usa directamente, es bueno tenerlo si se trabaja con JSONs
import logging
from datetime import datetime # No se usa actualmente, pero puede ser útil para futuras expansiones

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Pokémon TCG Explorer")
logging.basicConfig(level=logging.INFO) # Configura el logging básico

# --- Constantes y Configuración de GCP ---
# Intenta obtener el project_id de los secrets, maneja el posible KeyError
try:
    GCP_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
except KeyError:
    st.error("Error: 'project_id' no encontrado en los secrets de Streamlit ([gcp_service_account]). Verifica tu configuración.")
    st.stop()
except Exception as e: # Captura cualquier otra excepción al leer secrets
    st.error(f"Error inesperado al leer secrets: {e}")
    st.stop()


BIGQUERY_DATASET = "pokemon_dataset" # Asegúrate que este es el nombre correcto de tu dataset
CARD_METADATA_TABLE = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.card_metadata"
# Asume que tu modelo BQML está en el mismo dataset (ajusta si es necesario)
# BQML_MODEL_NAME = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.mlp_price_predictor" # Comentado si no se usa
# Opcional: Endpoint del microservicio de predicción
# PREDICTION_API_ENDPOINT = "URL_DE_TU_ENDPOINT_CLOUDRUN" # Comentado si no se usa


# --- Conexión Segura a BigQuery ---
@st.cache_resource # Cachea el recurso de conexión para eficiencia
def connect_to_bigquery():
    """Establece una conexión segura con BigQuery usando Service Account."""
    try:
        # Verifica si la sección principal de secrets está cargada
        if "gcp_service_account" not in st.secrets:
            st.error("Error: Sección [gcp_service_account] no encontrada en los secrets de Streamlit.")
            return None # Devuelve None para indicar fallo

        creds_json = dict(st.secrets["gcp_service_account"])

        # Verifica campos esenciales antes de crear credenciales
        required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"]
        missing_keys = [key for key in required_keys if key not in creds_json or not creds_json[key]]
        if missing_keys:
             st.error(f"Error: Faltan claves esenciales en los secrets de [gcp_service_account]: {', '.join(missing_keys)}")
             return None # Devuelve None para indicar fallo

        creds = service_account.Credentials.from_service_account_info(creds_json)
        client = bigquery.Client(credentials=creds, project=GCP_PROJECT_ID)
        logging.info("Conexión a BigQuery establecida correctamente.")
        return client
    except Exception as e:
        st.error(f"Error al conectar con BigQuery: {e}. Verifica el formato y contenido de tus secrets.")
        logging.error(f"Error al conectar con BigQuery: {e}", exc_info=True)
        return None # Devuelve None para indicar fallo

bq_client = connect_to_bigquery()

# Detiene la ejecución de la app si la conexión a BigQuery falló
if bq_client is None:
    st.stop()


# --- Funciones Auxiliares para Consultas ---

@st.cache_data(ttl=3600) # Cachea los resultados por 1 hora
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    """Encuentra la tabla de snapshot mensual más reciente."""
    # Usa el project_id y dataset directamente en la consulta INFORMATION_SCHEMA
    query = f"""
        SELECT table_id
        FROM `{_client.project}.{BIGQUERY_DATASET}`.__TABLES__
        WHERE STARTS_WITH(table_id, 'monthly_')
        ORDER BY table_id DESC
        LIMIT 1
    """
    try:
        results = _client.query(query).result()
        if results.total_rows > 0:
            latest_table_id = list(results)[0].table_id
            latest_table_full_path = f"{_client.project}.{BIGQUERY_DATASET}.{latest_table_id}"
            logging.info(f"Tabla de snapshot más reciente encontrada: {latest_table_full_path}")
            return latest_table_full_path
        else:
            logging.warning(f"No se encontraron tablas de snapshot mensuales ('monthly_YYYY_MM_DD') en el dataset {BIGQUERY_DATASET}.")
            st.warning(f"No se encontraron tablas de precios mensuales ('monthly_...') en el dataset '{BIGQUERY_DATASET}'.")
            return None
    except Exception as e:
        st.error(f"Error al buscar la tabla de snapshot más reciente: {e}. ¿Tiene la cuenta de servicio permisos para ver metadatos (ej. `roles/bigquery.metadataViewer`) en el dataset '{BIGQUERY_DATASET}'?")
        logging.error(f"Error al buscar la tabla de snapshot más reciente: {e}", exc_info=True)
        return None

@st.cache_data(ttl=3600) # Cachea los resultados por 1 hora
def get_distinct_values(_client: bigquery.Client, column_name: str) -> list:
    """Obtiene valores distintos para un campo de la tabla de metadatos."""
    query = f"SELECT DISTINCT {column_name} FROM `{CARD_METADATA_TABLE}` WHERE {column_name} IS NOT NULL ORDER BY {column_name}"
    try:
        results = _client.query(query).to_dataframe()
        return results[column_name].tolist()
    except Exception as e:
         # Verifica específicamente el error de db-dtypes
        if "db-dtypes" in str(e).lower(): # Convertir a minúsculas para comparación robusta
             st.error(f"Error al obtener valores para '{column_name}': Falta el paquete 'db-dtypes'. Añádelo a tu requirements.txt y reinicia.")
             logging.error("Error de dependencia: db-dtypes no encontrado al obtener valores distintos.", exc_info=True)
        else:
            st.error(f"Error al obtener valores distintos para '{column_name}' desde '{CARD_METADATA_TABLE}': {e}. Verifica el nombre de la tabla, la columna y los permisos.")
            logging.error(f"Error al obtener valores distintos para {column_name}: {e}", exc_info=True)
        return [] # Retorna lista vacía en caso de error

# --- Obtener la tabla más reciente y detener si no se encuentra ---
LATEST_SNAPSHOT_TABLE = get_latest_snapshot_table(bq_client)
if not LATEST_SNAPSHOT_TABLE:
    st.error("No se pudo determinar la tabla de precios más reciente. La aplicación no puede continuar.")
    st.stop()

# --- Lógica Principal de la Aplicación Streamlit ---
st.title("Explorador de Cartas Pokémon TCG")

# --- Barra Lateral: Filtros y Controles ---
st.sidebar.header("Filtros y Opciones")

# Cargar opciones para los filtros (cacheado)
# Ejecutar solo si bq_client es válido (aunque ya se detiene arriba si es None)
if bq_client:
    set_options = get_distinct_values(bq_client, "set_name")
    pokemon_options = get_distinct_values(bq_client, "name")
    rarity_options = get_distinct_values(bq_client, "rarity")
else: # Fallback si algo inesperado ocurre y bq_client es None aquí
    set_options, pokemon_options, rarity_options = [], [], []


# Widgets de filtro
selected_sets = st.sidebar.multiselect("Filtrar por Set:", set_options)
selected_pokemons = st.sidebar.multiselect("Filtrar por Pokémon:", pokemon_options)
selected_rarities = st.sidebar.multiselect("Filtrar por Rareza:", rarity_options)

# Widget de ordenamiento
sort_order = st.sidebar.radio("Ordenar por Precio (Trend):", ("Ascendente", "Descendente"), index=1)
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"

# --- Construcción y Ejecución de la Consulta Principal ---
@st.cache_data(ttl=600) # Cachea los datos filtrados por 10 minutos
def fetch_card_data(_client: bigquery.Client, latest_table: str, sets: list, pokemons: list, rarities: list, sort: str) -> pd.DataFrame:
    """Construye y ejecuta la consulta dinámica para obtener datos de cartas."""
    # Nombres de columna basados en esquemas: id, images_large (como image_url), cm_trendPrice (como price)
    base_query = f"""
    SELECT
        c.id,
        c.name AS pokemon_name,
        c.set_name,
        c.rarity,
        c.artist,
        c.images_large AS image_url, # URL de la imagen
        p.cm_trendPrice AS price    # Precio a usar (Cardmarket Trend Price)
    FROM
        `{CARD_METADATA_TABLE}` AS c
    JOIN
        `{latest_table}` AS p ON c.id = p.id # Une usando la columna 'id'
    WHERE 1=1
    """

    params = []
    filter_clauses = []

    if sets:
        filter_clauses.append("c.set_name IN UNNEST(@sets)")
        params.append(bigquery.ArrayQueryParameter("sets", "STRING", sets))
    if pokemons:
        filter_clauses.append("c.name IN UNNEST(@pokemons)")
        params.append(bigquery.ArrayQueryParameter("pokemons", "STRING", pokemons))
    if rarities:
        filter_clauses.append("c.rarity IN UNNEST(@rarities)")
        params.append(bigquery.ArrayQueryParameter("rarities", "STRING", rarities))

    if filter_clauses:
        base_query += " AND " + " AND ".join(filter_clauses)

    base_query += f" ORDER BY price {sort}" # Ordena por el alias 'price'

    job_config = bigquery.QueryJobConfig(query_parameters=params)

    # Logging de parámetros corregido y mejorado
    param_details = []
    for p in params:
        if isinstance(p, bigquery.ArrayQueryParameter):
            param_details.append((p.name, f"ARRAY<{p.parameter_type.type_}>", p.values))
        elif isinstance(p, bigquery.ScalarQueryParameter):
            param_details.append((p.name, p.type_, p.value))
        else:
            param_details.append((p.name, "UNKNOWN_TYPE", "UNKNOWN_VALUE"))
    logging.info(f"Ejecutando consulta: {base_query} con parámetros (nombre, tipo, valores): {param_details}")

    try:
        query_job = _client.query(base_query, job_config=job_config)
        results_df = query_job.to_dataframe()
        # Convierte la columna de precio a numérico, errores a NaN
        results_df['price'] = pd.to_numeric(results_df['price'], errors='coerce')
        logging.info(f"Consulta ejecutada. Se obtuvieron {len(results_df)} filas.")
        return results_df
    except Exception as e:
        if "db-dtypes" in str(e).lower():
             st.error("Error: Falta la librería 'db-dtypes'. Asegúrate de que esté en requirements.txt y reinicia la app.")
             logging.error("Error de dependencia: db-dtypes no encontrado al ejecutar consulta.", exc_info=True)
        else:
            st.error(f"Error al ejecutar la consulta principal: {e}. Revisa los nombres de columna, tablas y permisos.")
            logging.error(f"Error al ejecutar la consulta principal: {e}", exc_info=True)
        return pd.DataFrame() # Retorna DataFrame vacío en caso de error

# Ejecutar la consulta con los filtros actuales (solo si bq_client es válido)
if bq_client and LATEST_SNAPSHOT_TABLE: # LATEST_SNAPSHOT_TABLE también debe ser válido
    results_df = fetch_card_data(
        bq_client,
        LATEST_SNAPSHOT_TABLE,
        selected_sets,
        selected_pokemons,
        selected_rarities,
        sort_sql
    )
else:
    results_df = pd.DataFrame() # DataFrame vacío si no se puede consultar


# --- Área Principal: Visualización de Resultados ---
st.header("Resultados")

if not results_df.empty:
    # Prepara el DataFrame para mostrar: selecciona columnas y renombra
    display_columns_mapping = {
        'id': 'ID',
        'pokemon_name': 'Pokémon',
        'set_name': 'Set',
        'rarity': 'Rareza',
        'artist': 'Artista',
        'price': 'Precio (Trend €)' # Ajusta la etiqueta si usaste otro precio
    }
    # Filtra solo las columnas que existen en results_df para evitar errores
    actual_columns_to_display = [col for col in display_columns_mapping.keys() if col in results_df.columns]
    display_df = results_df[actual_columns_to_display].copy()
    display_df.rename(columns=display_columns_mapping, inplace=True)

    # Formatea la columna de precio para mostrar con símbolo y decimales, si existe
    price_display_column = display_columns_mapping.get('price') # Obtiene el nombre renombrado
    if price_display_column and price_display_column in display_df.columns:
         display_df[price_display_column] = display_df[price_display_column].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    st.header("Detalle de Carta")

    # Usa la columna 'id' del DataFrame original results_df para generar opciones
    # Asegúrate de que 'pokemon_name' e 'id' existen en results_df antes de crear opciones
    if 'pokemon_name' in results_df.columns and 'id' in results_df.columns:
        card_options = results_df['pokemon_name'].astype(str) + " (" + results_df['id'].astype(str) + ")"
        selected_card_display = st.selectbox("Selecciona una carta para ver detalles:", options=card_options)

        if selected_card_display:
            selected_card_id_str = selected_card_display[selected_card_display.rfind("(")+1:selected_card_display.rfind(")")]
            card_details = results_df[results_df['id'] == selected_card_id_str].iloc[0]

            col1, col2 = st.columns([1, 2])

            with col1:
                if 'image_url' in card_details and pd.notna(card_details['image_url']):
                    st.image(card_details['image_url'], caption=f"{card_details['pokemon_name']} - {card_details['set_name']}", width=300)
                else:
                    st.warning("Imagen no disponible.")

            with col2:
                st.subheader(f"{card_details['pokemon_name']}")
                if 'id' in card_details: st.markdown(f"**ID:** `{card_details['id']}`")
                if 'set_name' in card_details: st.markdown(f"**Set:** {card_details['set_name']}")
                if 'rarity' in card_details: st.markdown(f"**Rareza:** {card_details['rarity']}")
                if 'artist' in card_details and pd.notna(card_details['artist']) and card_details['artist']:
                     st.markdown(f"**Artista:** {card_details['artist']}")
                if 'price' in card_details:
                     st.metric(label="Precio (Trend €)", value=f"€{card_details['price']:.2f}" if pd.notna(card_details['price']) else "N/A")


                # --- (Opcional) Integración con Predicción de Precios ---
                # st.subheader("Predicción de Precio (Próximo Mes) [Opcional]")
                # ... (Código de predicción aquí, asegurándose de usar las columnas correctas) ...

    else: # Si 'pokemon_name' o 'id' no están en results_df (puede pasar si la consulta falló mal)
        st.warning("No se pueden mostrar detalles de cartas, datos insuficientes.")

else:
    if bq_client and LATEST_SNAPSHOT_TABLE:
        st.info("No se encontraron cartas con los filtros seleccionados o hubo un error al consultar los datos. Verifica los filtros o los logs para más detalles.")
    # Si bq_client o LATEST_SNAPSHOT_TABLE fallaron, el error ya se mostró antes.

# --- Footer opcional ---
st.sidebar.info("Aplicación Pokémon TCG Explorer")
