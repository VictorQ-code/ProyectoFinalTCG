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
GCP_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
BIGQUERY_DATASET = "pokemon_dataset"
CARD_METADATA_TABLE = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.card_metadata"
# Asume que tu modelo BQML está en el mismo dataset
BQML_MODEL_NAME = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.mlp_price_predictor"
# Opcional: Endpoint del microservicio de predicción
# PREDICTION_API_ENDPOINT = "URL_DE_TU_ENDPOINT_CLOUDRUN"

# --- Conexión Segura a BigQuery ---
@st.cache_resource # Cachea el recurso de conexión
def connect_to_bigquery():
    """Establece una conexión segura con BigQuery usando Service Account."""
    try:
        creds_json = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(creds_json)
        client = bigquery.Client(credentials=creds, project=GCP_PROJECT_ID)
        logging.info("Conexión a BigQuery establecida correctamente.")
        return client
    except Exception as e:
        st.error(f"Error al conectar con BigQuery: {e}")
        logging.error(f"Error al conectar con BigQuery: {e}", exc_info=True)
        st.stop() # Detiene la ejecución si no se puede conectar

bq_client = connect_to_bigquery()

# --- Funciones Auxiliares para Consultas ---

@st.cache_data(ttl=3600) # Cachea los resultados por 1 hora
def get_latest_snapshot_table(_client: bigquery.Client) -> str | None:
    """Encuentra la tabla de snapshot mensual más reciente."""
    query = f"""
        SELECT table_id
        FROM `{BIGQUERY_DATASET}.__TABLES__`
        WHERE STARTS_WITH(table_id, 'monthly_')
        ORDER BY table_id DESC
        LIMIT 1
    """
    try:
        results = _client.query(query).result()
        if results.total_rows > 0:
            latest_table = list(results)[0].table_id
            logging.info(f"Tabla de snapshot más reciente encontrada: {latest_table}")
            return f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{latest_table}"
        else:
            logging.warning("No se encontraron tablas de snapshot mensuales ('monthly_YYYY_MM_DD').")
            return None
    except Exception as e:
        st.error(f"Error al buscar la tabla de snapshot más reciente: {e}")
        logging.error(f"Error al buscar la tabla de snapshot más reciente: {e}", exc_info=True)
        return None

@st.cache_data(ttl=3600) # Cachea los resultados por 1 hora
def get_distinct_values(_client: bigquery.Client, column_name: str) -> list:
    """Obtiene valores distintos para un campo de la tabla de metadatos."""
    query = f"SELECT DISTINCT {column_name} FROM `{CARD_METADATA_TABLE}` ORDER BY {column_name}"
    try:
        results = _client.query(query).to_dataframe()
        return results[column_name].dropna().tolist()
    except Exception as e:
        st.error(f"Error al obtener valores distintos para {column_name}: {e}")
        logging.error(f"Error al obtener valores distintos para {column_name}: {e}", exc_info=True)
        return []

# --- Obtener la tabla más reciente ---
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
sort_order = st.sidebar.radio("Ordenar por Precio:", ("Ascendente", "Descendente"), index=1)
sort_sql = "ASC" if sort_order == "Ascendente" else "DESC"

# --- Construcción y Ejecución de la Consulta Principal ---
@st.cache_data(ttl=600) # Cachea los datos filtrados por 10 minutos
def fetch_card_data(_client: bigquery.Client, latest_table: str, sets: list, pokemons: list, rarities: list, sort: str) -> pd.DataFrame:
    """Construye y ejecuta la consulta dinámica para obtener datos de cartas."""
    base_query = f"""
    SELECT
        c.card_id,
        c.name AS pokemon_name,
        c.set_name,
        c.rarity,
        c.artist,
        c.image_url,
        p.price
    FROM
        `{CARD_METADATA_TABLE}` AS c
    JOIN
        `{latest_table}` AS p ON c.card_id = p.card_id
    WHERE 1=1
    """

    # Añadir filtros dinámicamente y preparar parámetros
    params = []
    param_types = []
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

    # Añadir ordenamiento
    base_query += f" ORDER BY p.price {sort}"

    # Configurar la consulta parametrizada
    job_config = bigquery.QueryJobConfig(query_parameters=params)

    logging.info(f"Ejecutando consulta: {base_query} con parámetros: {[(p.name, p.array_values) for p in params]}")

    try:
        query_job = _client.query(base_query, job_config=job_config)
        results_df = query_job.to_dataframe()
        logging.info(f"Consulta ejecutada. Se obtuvieron {len(results_df)} filas.")
        return results_df
    except Exception as e:
        st.error(f"Error al ejecutar la consulta principal: {e}")
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
    st.dataframe(results_df, use_container_width=True)

    st.divider()
    st.header("Detalle de Carta")

    # Selector para elegir una carta de los resultados
    card_options = results_df['pokemon_name'] + " (" + results_df['card_id'] + ")"
    selected_card_display = st.selectbox("Selecciona una carta para ver detalles:", options=card_options)

    if selected_card_display:
        # Extraer card_id del string seleccionado
        selected_card_id = selected_card_display[selected_card_display.rfind("(")+1:selected_card_display.rfind(")")]

        # Encontrar los datos de la carta seleccionada en el DataFrame
        card_details = results_df[results_df['card_id'] == selected_card_id].iloc[0]

        col1, col2 = st.columns([1, 2]) # Columna para imagen, columna para texto

        with col1:
            if pd.notna(card_details['image_url']):
                st.image(card_details['image_url'], caption=f"{card_details['pokemon_name']} - {card_details['set_name']}", width=250)
            else:
                st.warning("Imagen no disponible.")

        with col2:
            st.subheader(f"{card_details['pokemon_name']}")
            st.markdown(f"**ID:** `{card_details['card_id']}`")
            st.markdown(f"**Set:** {card_details['set_name']}")
            st.markdown(f"**Rareza:** {card_details['rarity']}")
            st.markdown(f"**Artista:** {card_details['artist']}")
            st.metric(label="Precio Actual (USD)", value=f"${card_details['price']:.2f}" if pd.notna(card_details['price']) else "N/A")

            # --- (Opcional) Integración con Predicción de Precios ---
            st.subheader("Predicción de Precio (Próximo Mes)")

            # Botón para activar la predicción
            if st.button("Predecir Precio Futuro"):
                # -- Opción 1: Usando BigQuery ML --
                @st.cache_data(ttl=600)
                def get_price_prediction_bqml(_client: bigquery.Client, model_name: str, card_id: str) -> float | None:
                    """Obtiene la predicción de precio usando un modelo BQML."""
                    # Nota: Esta consulta es un EJEMPLO. Debe adaptarse EXACTAMENTE
                    # a las características que espera tu modelo BQML 'mlp_price_predictor'.
                    # Necesitarás probablemente unir con la tabla anterior para price_t0_log y days_diff.
                    # Asumimos que el modelo necesita el card_id y busca internamente las features.
                    # O podrías pasar todas las features requeridas explícitamente.

                    # Ejemplo simplificado (¡AJUSTAR A TU MODELO REAL!):
                    # Supongamos que el modelo sólo necesita el precio actual y metadatos.
                    # Necesitarías una query que prepare las features exactamente como espera el modelo.
                    # Esto podría involucrar unir la tabla actual y la anterior.
                    # Aquí un ejemplo MUY simplificado asumiendo que el modelo
                    # puede inferir todo desde el card_id y el último precio.
                    predict_query = f"""
                    SELECT predicted_price
                    FROM ML.PREDICT(MODEL `{model_name}`,
                        (
                        SELECT
                            c.*, -- Todas las columnas de metadatos
                            p.price,
                            -- Aquí necesitarías calcular price_t0_log, days_diff, etc.
                            -- uniéndote con la tabla de snapshot anterior.
                            -- Por simplicidad, omitimos esto aquí. ¡DEBES AGREGARLO!
                            LOG(p.price) as price_t0_log_placeholder, -- ¡EJEMPLO!
                            30 as days_diff_placeholder -- ¡EJEMPLO!
                        FROM `{CARD_METADATA_TABLE}` c
                        JOIN `{LATEST_SNAPSHOT_TABLE}` p ON c.card_id = p.card_id
                        WHERE c.card_id = @card_id
                        )
                    )
                    """
                    params = [bigquery.ScalarQueryParameter("card_id", "STRING", card_id)]
                    job_config = bigquery.QueryJobConfig(query_parameters=params)

                    try:
                        pred_job = _client.query(predict_query, job_config=job_config)
                        pred_results = pred_job.to_dataframe()
                        if not pred_results.empty:
                            # BQML a menudo predice el log del precio, así que podríamos necesitar exp()
                            # Ajusta según cómo entrenaste tu modelo.
                            # Asumimos que 'predicted_price' es el nombre del campo de salida.
                            predicted_value = pred_results['predicted_price'].iloc[0]
                            # Podría necesitarse: predicted_price = np.exp(predicted_value)
                            logging.info(f"Predicción BQML para {card_id}: {predicted_value}")
                            return predicted_value
                        else:
                            logging.warning(f"BQML no devolvió predicción para {card_id}.")
                            return None
                    except Exception as e:
                        st.error(f"Error al obtener predicción de BQML: {e}")
                        logging.error(f"Error al obtener predicción de BQML para {card_id}: {e}", exc_info=True)
                        return None

                predicted_price = get_price_prediction_bqml(bq_client, BQML_MODEL_NAME, selected_card_id)

                if predicted_price is not None:
                     # Asumiendo que la predicción es directa (ajustar si es log, etc.)
                    st.metric(label="Precio Predicho (Próximo Mes)", value=f"${predicted_price:.2f}")
                else:
                    st.warning("No se pudo obtener la predicción de precio.")

                # -- Opción 2: Llamando a un Microservicio (Cloud Run) --
                # import requests
                # @st.cache_data(ttl=600)
                # def get_price_prediction_api(card_data: pd.Series) -> float | None:
                #     """Llama a la API de predicción externa."""
                #     # 1. Preparar los datos de entrada para la API
                #     #    Necesitarás obtener price_t0_log, days_diff y las features
                #     #    one-hot codificadas. Esto podría requerir otra consulta a BQ.
                #     features_for_api = {
                #         "price_t0_log": ..., # Calcular o buscar
                #         "days_diff": ...,    # Calcular o buscar
                #         "rarity": card_data['rarity'], # La API deberá hacer one-hot
                #         "set_name": card_data['set_name'], # La API deberá hacer one-hot
                #         # ... otras features ...
                #     }
                #     try:
                #         response = requests.post(PREDICTION_API_ENDPOINT, json=features_for_api, timeout=10)
                #         response.raise_for_status() # Lanza excepción si hay error HTTP
                #         prediction = response.json().get("predicted_price")
                #         logging.info(f"Predicción API para {card_data['card_id']}: {prediction}")
                #         # Podría necesitarse np.exp(prediction) si la API devuelve log(precio)
                #         return prediction
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error al llamar a la API de predicción: {e}")
                #         logging.error(f"Error al llamar a la API para {card_data['card_id']}: {e}", exc_info=True)
                #         return None

                # predicted_price_api = get_price_prediction_api(card_details)
                # if predicted_price_api is not None:
                #     st.metric(label="Precio Predicho (API)", value=f"${predicted_price_api:.2f}")
                # else:
                #     st.warning("No se pudo obtener la predicción de precio (API).")

else:
    st.info("No se encontraron cartas con los filtros seleccionados.")

# --- Footer opcional ---
st.sidebar.info("Aplicación desarrollada por [Tu Nombre/Equipo]")