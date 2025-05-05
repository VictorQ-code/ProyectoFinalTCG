import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd

# ——————————— Autenticación BigQuery ———————————
bq_secrets = st.secrets["bigquery"]
credentials = service_account.Credentials.from_service_account_info(bq_secrets)
client = bigquery.Client(
    credentials=credentials,
    project=bq_secrets["project_id"],
)

# ——————— Obtener snapshots más recientes ———————
@st.cache_data(ttl=3600)
def get_latest_two_snapshots():
    tables = client.list_tables(f"{client.project}.pokemon_dataset")
    monthly = sorted([t.table_id for t in tables if t.table_id.startswith("monthly_")])
    if len(monthly) < 1:
        return None
    return monthly[-1]  # Usamos sólo la tabla más reciente para precios

latest_snapshot = get_latest_two_snapshots()

# ——————— Cargar valores únicos de metadata ———————
@st.cache_data(ttl=3600)
def load_distinct(field):
    query = f"SELECT DISTINCT {field} FROM `{client.project}.pokemon_dataset.card_metadata`"
    df = client.query(query).to_dataframe() if latest_snapshot else pd.DataFrame()
    return df[field].dropna().tolist()

sets     = st.sidebar.multiselect("Filtrar por Set",     load_distinct("set_name"))
pokemons = st.sidebar.multiselect("Filtrar por Pokémon", load_distinct("name"))
rarities = st.sidebar.multiselect("Filtrar por Rareza", load_distinct("rarity"))

# ——————— Construir cláusula WHERE dinámica ———————
clauses = []
if sets:
    clauses.append("m.set_name IN UNNEST(@sets)")
if pokemons:
    clauses.append("m.name IN UNNEST(@pokemons)")
if rarities:
    clauses.append("m.rarity IN UNNEST(@rarities)")
where = " AND ".join(clauses) if clauses else "TRUE"

# ——————— Cargar cartas filtradas ———————
@st.cache_data(ttl=600)
def load_cards(sets, pokemons, rarities):
    if not latest_snapshot:
        return pd.DataFrame()
    sql = f"""
    SELECT
      p0.id           AS card_id,
      m.name          AS pokemon_name,
      m.set_name,
      m.rarity,
      m.artist        AS artist_name,
      m.image_url,
      p0.cm_averageSellPrice AS price
    FROM `{client.project}.pokemon_dataset.{latest_snapshot}` p0
    LEFT JOIN `{client.project}.pokemon_dataset.card_metadata` m
      ON p0.id = m.id
    WHERE {where}
    """
    params = []
    if sets:
        params.append(bigquery.ArrayQueryParameter("sets", "STRING", sets))
    if pokemons:
        params.append(bigquery.ArrayQueryParameter("pokemons", "STRING", pokemons))
    if rarities:
        params.append(bigquery.ArrayQueryParameter("rarities", "STRING", rarities))
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    return client.query(sql, job_config=job_config).to_dataframe()

df_cards = load_cards(sets, pokemons, rarities)

# ——————— Ordenar por precio ———————
sort_order = st.sidebar.radio("Ordenar por precio", ("Ascendente", "Descendente"))
ascending = sort_order == "Ascendente"
df_cards = df_cards.sort_values("price", ascending=ascending)

st.title("🔍 Explorador de Cartas Pokémon")

# ——————— Mostrar tabla de resultados ———————
if df_cards.empty:
    st.info("No hay cartas que coincidan con los filtros.")
else:
    st.dataframe(df_cards[["card_id","pokemon_name","set_name","rarity","artist_name","price"]])

    # ——————— Selección de carta para detalles ———————
    selected = st.selectbox("Selecciona una carta para ver detalles", df_cards["card_id"].tolist())
    if selected:
        card = df_cards[df_cards["card_id"] == selected].iloc[0]
        st.image(card["image_url"], width=200)
        st.subheader(f"{card['pokemon_name']} ({card['card_id']})")
        st.write(f"**Set:** {card['set_name']}")
        st.write(f"**Rareza:** {card['rarity']}")
        st.write(f"**Artista:** {card['artist_name']}")
        st.write(f"**Precio:** €{card['price']:.2f}")
