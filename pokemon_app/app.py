import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import logging
from datetime import datetime

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Pokémon TCG Explorer")
logging.basicConfig(level=logging.INFO)

# --- Constantes y Configuración de GCP ---
# Intenta obtener el project_id de los secrets, maneja el posible KeyError
try:
    GCP_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
except KeyError:
    st.error("Error: 'project_id' no encontrado en los secrets de Streamlit ([gcp_service_account]). Verifica tu configuración.")
    st.stop()
except Exception as e:
    st.error(f"Error inesperado al leer secrets: {e}")
    st.stop()


BIGQUERY_DATASET = "pokemon_dataset" # Asegúrate que este es el nombre correcto de tu dataset
CARD_METADATA_TABLE = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.card_metadata"
# Asume que tu modelo BQML está en el mismo dataset (ajusta si es necesario)
# BQML_MODEL_NAME = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.mlp_price_predictor" # Comentado si no se usa
# Opcional: Endpoint del microservicio de predicción
# PREDICTION_API_ENDPOINT = "URL_DE_TU_ENDPOINT_CLOUDRUN" # Comentado si no se usa


# --- Conexión Segura a BigQuery ---
@st.cache_resource # Cachea el recurso de conexión
def connect_to_bigquery():
    """Establece una conexión segura con BigQuery usando Service Account."""
    try:
        # Verifica si los secrets están cargados
        if "gcp_service_account" not in st.secrets:
            st.error("Error: Sección [gcp_service_account] no encontrada en los secrets de Streamlit.")
            return None # Devuelve None para indicar fallo

        creds_json = dict(st.secrets["gcp_service_account"])

        # Verifica campos esenciales antes de crear credenciales
        required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"]
        missing_keys = [key for key in required_keys if key not in creds_json or not creds_json[key]]
        if missing_keys:
             st.error(f"Error: Faltan claves en los secrets de [gcp_service_account]: {', '.join(missing_keys)}")
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

# Detiene la ejecución si la conexión falló
if bq_client is None:
    st.stop()


# --- Funciones Auxiliares para Consultas ---

@st.cache_data(ttl=3600) # Cachea los resultados por 1 hora
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    """Encuentra la tabla de snapshot mensual más reciente."""
    query = f"""
        SELECT table_id
        FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}`.__TABLES__
        WHERE STARTS_WITH(table_id, 'monthly_')
        ORDER BY table_id DESC
        LIMIT 1
    """
    try:
        results = _client.query(query).result()
        if results.total_rows > 0:
            latest_table_id = list(results)[0].table_id
            latest_table_full_path = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{latest_table_id}"
            logging.info(f"Tabla de snapshot más reciente encontrada: {latest_table_full_path}")
            return latest_table_full_path
        else:
            logging.warning(f"No se encontraron tablas de snapshot mensuales ('monthly_YYYY_MM_DD') en el dataset {BIGQUERY_DATASET}.")
            st.warning(f"No se encontraron tablas de precios mensuales ('monthly_...') en el dataset '{BIGQUERY_DATASET}'.")
            return None
    except Exception as e:
        st.error(f"Error al buscar la tabla de snapshot más reciente: {e}. ¿Tiene la cuenta de servicio permisos para ver metadatos?")
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
        if "db-dtypes" in str(e):
             st.error(f"Error al obtener valores para '{column_name}': Falta el paquete 'db-dtypes'. Añádelo a tu requirements.txt y reinicia.")
             logging.error("Error de dependencia: db-dtypes no encontrado al obtener valores distintos.", exc_info=True)
        else:
            st.error(f"Error al obtener valores distintos para '{column_name}' desde '{CARD_METADATA_TABLE}': {e}. Verifica el nombre de la tabla y la columna.")
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
set_options = get_distinct_values(bq_client, "set_name")
pokemon_options = get_distinct_values(bq_client, "name")
rarity_options = get_distinct_values(bq_client, "rarity")

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
    # Nombres de columna corregidos: id, images_large, cm_trendPrice
    # Alias usados: image_url, price
    base_query = f"""
    SELECT
        c.id,                   # Identificador único
        c.name AS pokemon_name,
        c.set_name,
        c.rarity,
        c.artist,
        c.images_large AS image_url, # URL de la imagen (usando alias)
        p.cm_trendPrice AS price    # Precio a usar (usando alias) - CAMBIA cm_trendPrice si prefieres otro
    FROM
        `{CARD_METADATA_TABLE}` AS c
    JOIN
        `{latest_table}` AS p ON c.id = p.id # Une usando la columna 'id'
    WHERE 1=1
    """

    params = []
    # param_types = [] # Esta variable no se usa, se puede eliminar si no se necesita para algo más
    filter_clauses = []

    # Añadir filtros dinámicamente
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

    # Añadir ordenamiento por el alias 'price'
    base_query += f" ORDER BY price {sort}"

    # Configurar la consulta parametrizada
    job_config = bigquery.QueryJobConfig(query_parameters=params)

    # --- LÍNEA DE LOGGING CORREGIDA ---
    param_names_types = [(p.name, p.type_) for p in params]
    logging.info(f"Ejecutando consulta: {base_query} con parámetros (nombre, tipo): {param_names_types}")
    # --- FIN DE LA CORRECCIÓN ---

    try:
        query_job = _client.query(base_query, job_config=job_config)
        results_df = query_job.to_dataframe()
        # Convierte la columna de precio a numérico, errores a NaN
        results_df['price'] = pd.to_numeric(results_df['price'], errors='coerce')
        logging.info(f"Consulta ejecutada. Se obtuvieron {len(results_df)} filas.")
        return results_df
    except Exception as e:
         # Verifica específicamente el error de db-dtypes
        if "db-dtypes" in str(e):
             st.error("Error: Falta la librería 'db-dtypes'. Asegúrate de que esté en requirements.txt y reinicia la app.")
             logging.error("Error de dependencia: db-dtypes no encontrado al ejecutar consulta.", exc_info=True)
        else:
            st.error(f"Error al ejecutar la consulta principal: {e}. Revisa los nombres de columna y tablas en la consulta.")
            logging.error(f"Error al ejecutar la consulta principal: {e}", exc_info=True)
        return pd.DataFrame() # Retorna DataFrame vacío en caso de error

# Ejecutar la consulta con los filtros actuales
results_df = fetch_card_data(
    bq_client,
    LATEST_SNAPSHOT_TABLE,
    selected_sets,
    selected_pokemons,
    selected_rarities,
    sort_sql
)

# --- Área Principal: Visualización de Resultados ---
st.header("Resultados")

if not results_df.empty:
    # Prepara el DataFrame para mostrar: selecciona columnas y renombra
    display_columns = {
        'id': 'ID',
        'pokemon_name': 'Pokémon',
        'set_name': 'Set',
        'rarity': 'Rareza',
        'artist': 'Artista',
        'price': 'Precio (Trend €)' # Ajusta la etiqueta si usaste otro precio
    }
    # Filtra solo las columnas que existen en results_df para evitar errores
    columns_to_display = [col for col in display_columns.keys() if col in results_df.columns]
    display_df = results_df[columns_to_display].copy()
    display_df.rename(columns=display_columns, inplace=True)

    # Formatea la columna de precio para mostrar con símbolo y decimales
    if 'Precio (Trend €)' in display_df.columns:
         display_df['Precio (Trend €)'] = display_df['Precio (Trend €)'].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    st.header("Detalle de Carta")

    # Usa la columna 'id' del DataFrame original results_df para generar opciones
    card_options = results_df['pokemon_name'] + " (" + results_df['id'] + ")"
    selected_card_display = st.selectbox("Selecciona una carta para ver detalles:", options=card_options)

    if selected_card_display:
        # Extrae el 'id' del string seleccionado
        selected_card_id_str = selected_card_display[selected_card_display.rfind("(")+1:selected_card_display.rfind(")")]

        # Busca la fila completa en el DataFrame original usando el 'id'
        card_details = results_df[results_df['id'] == selected_card_id_str].iloc[0]

        col1, col2 = st.columns([1, 2]) # Columna para imagen, columna para texto

        with col1:
            # Usa la columna 'image_url' (alias de images_large)
            if pd.notna(card_details['image_url']):
                st.image(card_details['image_url'], caption=f"{card_details['pokemon_name']} - {card_details['set_name']}", width=300) # Ancho aumentado
            else:
                st.warning("Imagen no disponible.")

        with col2:
            st.subheader(f"{card_details['pokemon_name']}")
            st.markdown(f"**ID:** `{card_details['id']}`") # Muestra el 'id'
            st.markdown(f"**Set:** {card_details['set_name']}")
            st.markdown(f"**Rareza:** {card_details['rarity']}")
            # Muestra el artista solo si no es nulo/vacío
            if pd.notna(card_details['artist']) and card_details['artist']:
                 st.markdown(f"**Artista:** {card_details['artist']}")
            # Usa la columna 'price' (alias de cm_trendPrice) para la métrica
            st.metric(label="Precio (Trend €)", value=f"€{card_details['price']:.2f}" if pd.notna(card_details['price']) else "N/A")


            # --- (Opcional) Integración con Predicción de Precios ---
            # ¡IMPORTANTE! Si activas esto, debes asegurarte de que tu modelo
            # BQML o API use las columnas correctas ('id', 'cm_trendPrice', etc.)
            # y que la consulta ML.PREDICT esté adaptada.

            # st.subheader("Predicción de Precio (Próximo Mes) [Opcional]")
            # if st.button("Predecir Precio Futuro"):
            #     # -- Opción 1: Usando BigQuery ML --
            #     @st.cache_data(ttl=600)
            #     def get_price_prediction_bqml(_client: bigquery.Client, model_name: str, card_identifier: str) -> float | None:
            #         """Obtiene la predicción de precio usando un modelo BQML."""
            #         # ¡ESTA CONSULTA DEBE SER ADAPTADA A TU MODELO REAL!
            #         # Necesitará las features exactas que el modelo espera.
            #         # Probablemente requiera unir tabla actual y anterior.
            #         predict_query = f"""
            #         SELECT predicted_price # Ajusta el nombre de la columna predicha
            #         FROM ML.PREDICT(MODEL `{model_name}`,
            #             (
            #             -- Aquí necesitas construir las features EXACTAS para tu modelo
            #             -- Ejemplo MUY simplificado:
            #             SELECT
            #                 c.* EXCEPT(id), -- Excluye id si no lo usa el modelo
            #                 p.cm_trendPrice,
            #                 LOG(p.cm_trendPrice) as price_t0_log_placeholder, -- Ejemplo
            #                 30 as days_diff_placeholder -- Ejemplo
            #             FROM `{CARD_METADATA_TABLE}` c
            #             JOIN `{LATEST_SNAPSHOT_TABLE}` p ON c.id = p.id -- Usa 'id'
            #             WHERE c.id = @card_identifier -- Usa 'id'
            #             )
            #         )
            #         """
            #         params = [bigquery.ScalarQueryParameter("card_identifier", "STRING", card_identifier)]
            #         job_config = bigquery.QueryJobConfig(query_parameters=params)
            #         try:
            #             pred_job = _client.query(predict_query, job_config=job_config)
            #             pred_results = pred_job.to_dataframe()
            #             if not pred_results.empty:
            #                 # Ajusta según lo que prediga el modelo (log, valor directo, etc.)
            #                 predicted_value = pred_results.iloc[0, 0] # Obtiene la primera columna predicha
            #                 # Podría necesitarse: predicted_price = np.exp(predicted_value)
            #                 logging.info(f"Predicción BQML para {card_identifier}: {predicted_value}")
            #                 return predicted_value
            #             else: return None
            #         except Exception as e:
            #             st.error(f"Error al obtener predicción de BQML: {e}")
            #             logging.error(f"Error BQML para {card_identifier}: {e}", exc_info=True)
            #             return None

            #     predicted_price = get_price_prediction_bqml(bq_client, BQML_MODEL_NAME, selected_card_id_str)
            #     if predicted_price is not None:
            #         st.metric(label="Precio Predicho (Próximo Mes)", value=f"€{predicted_price:.2f}") # Ajusta formato/moneda
            #     else:
            #         st.warning("No se pudo obtener la predicción de precio.")

                # -- Opción 2: Llamando a un Microservicio (Cloud Run) --
                # ... (Código para llamar a API externa, necesitaría ajuste similar de features) ...

else:
    # Mensaje si no hay resultados o si hubo error en la consulta
    if LATEST_SNAPSHOT_TABLE: # Solo muestra si la búsqueda inicial de tabla fue exitosa
        st.info("No se encontraron cartas con los filtros seleccionados o hubo un error al consultar los datos. Verifica los filtros o los logs.")

