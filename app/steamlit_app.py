import streamlit as st
from google.cloud import bigquery
import pandas as pd

# Inicializa el cliente de BigQuery (usa GOOGLE_APPLICATION_CREDENTIALS)
client = bigquery.Client()

st.set_page_config(page_title="Pokémon Price Predictor", layout="centered")
st.title("🔮 Pokémon Card Price Predictor")

card_id = st.text_input("Introduce el ID de la carta", "")
if st.button("Predecir precio"):
    if not card_id:
        st.warning("Debes escribir un card_id.")
    else:
        # Construye la SQL que invoque tu modelo remoto ML.PREDICT
        sql = f"""
        SELECT
          EXP(predicted_price_t1_log) - 1 AS predicted_price_eur
        FROM
          ML.PREDICT(
            MODEL `pokemon-cards-project.pokemon_dataset.mlp_price_predictor`,
            (
              SELECT
                p0.id                          AS card_id,
                LN(p0.cm_averageSellPrice+1)   AS price_t0_log,
                DATE_DIFF(DATE("2025-04-30"), DATE("2025-04-01"), DAY) AS days_diff,
                m.artist                       AS artist_name,
                m.name                         AS pokemon_name,
                m.rarity,
                m.set_name,
                m.types,
                m.supertype,
                m.subtypes
              FROM
                `pokemon-cards-project.pokemon_dataset.monthly_2025_04_01` AS p0
              LEFT JOIN
                `pokemon-cards-project.pokemon_dataset.card_metadata` AS m
                ON p0.id = m.id
              WHERE
                p0.id = @card_id
            )
          )
        """
        job = client.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("card_id", "STRING", card_id)
                ]
            )
        )
        df = job.result().to_dataframe()
        if df.empty:
            st.error("No se encontró esa carta en los datos.")
        else:
            price = df.loc[0, "predicted_price_eur"]
            st.metric("Precio previsto (€)", f"{price:.2f}")
