# --------------------------------------------------------------
# Script completo en R: consultas a BigQuery + gráficos vistosos
# --------------------------------------------------------------

# 1) Instala y carga los paquetes necesarios (solo la 1ª vez)
install.packages(c("DBI", "bigrquery", "ggplot2"))
library(DBI)
library(bigrquery)
library(ggplot2)

# 2) Parámetros de tu proyecto BigQuery
project_id <- "pokemon-cards-project"
dataset_id <- "pokemon_dataset"

# 3) Autenticación (se abrirá tu navegador)
bq_auth()

# 4) Conexión a BigQuery vía DBI
con <- dbConnect(
  bigquery(),
  project = project_id,
  dataset = dataset_id,
  billing = project_id
)

# 5) Definición de las consultas

sql_sets <- "
SELECT
  p.set_name                     AS expansion,
  ROUND(AVG(p.cm_averageSellPrice),2) AS precio_medio_set
FROM `pokemon-cards-project.pokemon_dataset.monthly_2025_04_30` AS p
WHERE p.set_name <> 'POP Series 5'
GROUP BY expansion
ORDER BY precio_medio_set DESC
LIMIT 10;
"

sql_cartas <- "
SELECT
  p.name                         AS carta,
  ROUND(p.cm_averageSellPrice,2) AS precio_promedio
FROM `pokemon-cards-project.pokemon_dataset.monthly_2025_04_30` AS p
ORDER BY precio_promedio DESC
LIMIT 10;
"

sql_max_por_artista <- "
SELECT
  m.artist                       AS artista,
  MAX(p.cm_averageSellPrice)     AS precio_maximo
FROM `pokemon-cards-project.pokemon_dataset.monthly_2025_04_30` AS p
JOIN `pokemon-cards-project.pokemon_dataset.card_metadata` AS m
  ON p.id = m.id
GROUP BY artista
ORDER BY precio_maximo DESC
LIMIT 10;
"

sql_promedio_por_artista <- "
SELECT
  m.artist                           AS artista,
  ROUND(AVG(p.cm_averageSellPrice),2) AS precio_medio
FROM `pokemon-cards-project.pokemon_dataset.monthly_2025_04_30` AS p
JOIN `pokemon-cards-project.pokemon_dataset.card_metadata` AS m
  ON p.id = m.id
GROUP BY artista
ORDER BY precio_medio DESC
LIMIT 10;
"

sql_por_rareza <- "
SELECT
  m.rarity                         AS rareza,
  ROUND(AVG(p.cm_averageSellPrice),2) AS precio_medio
FROM `pokemon-cards-project.pokemon_dataset.monthly_2025_04_30` AS p
JOIN `pokemon-cards-project.pokemon_dataset.card_metadata` AS m
  ON p.id = m.id
GROUP BY rareza
ORDER BY precio_medio DESC
LIMIT 10;
"

sql_max_por_pokemon <- "
SELECT
  p.name                          AS carta,
  MAX(p.cm_averageSellPrice)      AS precio_maximo
FROM `pokemon-cards-project.pokemon_dataset.monthly_2025_04_30` AS p
GROUP BY carta
ORDER BY precio_maximo DESC
LIMIT 10;
"

# 6) Ejecución de consultas y descarga de resultados
df_sets         <- dbGetQuery(con, sql_sets)
df_cartas       <- dbGetQuery(con, sql_cartas)
df_artista_max  <- dbGetQuery(con, sql_max_por_artista)
df_artista_prom <- dbGetQuery(con, sql_promedio_por_artista)
df_rareza       <- dbGetQuery(con, sql_por_rareza)
df_pokemon_max  <- dbGetQuery(con, sql_max_por_pokemon)

# 7) Gráficos vistosos con ggplot2

# 7.1 Top 10 Sets Más Caros
ggplot(df_sets, aes(x = reorder(expansion, precio_medio_set), y = precio_medio_set)) +
  geom_col(fill = "#1f77b4") +
  coord_flip() +
  labs(
    title = "Top 10 Sets Más Caros - Precio Medio De Sus Cartas",
    x = "Expansión",
    y = "Precio Medio (€)"
  ) +
  theme_minimal(base_size = 14)

# 7.2 Top 10 Cartas Más Caras
ggplot(df_cartas, aes(x = reorder(carta, precio_promedio), y = precio_promedio)) +
  geom_col(fill = "#ff7f0e") +
  coord_flip() +
  labs(
    title = "Top 10 Cartas Más Caras - Precios Promedio",
    x = "Carta",
    y = "Precio Promedio (€)"
  ) +
  theme_minimal(base_size = 14)

# 7.3 Top Carta Más Cara por Artista
ggplot(df_artista_max, aes(x = reorder(artista, precio_maximo), y = precio_maximo)) +
  geom_col(fill = "#2ca02c") +
  coord_flip() +
  labs(
    title = "Top Carta Más Cara por Artista",
    x = "Artista",
    y = "Precio Máximo (€)"
  ) +
  theme_minimal(base_size = 14)

# 7.4 Artistas con Cartas Más Caras (Precio Medio)
ggplot(df_artista_prom, aes(x = reorder(artista, precio_medio), y = precio_medio)) +
  geom_col(fill = "#d62728") +
  coord_flip() +
  labs(
    title = "Top Artistas con Cartas Más Caras - Precio Medio",
    x = "Artista",
    y = "Precio Medio (€)"
  ) +
  theme_minimal(base_size = 14)

# 7.5 Rarezas con Precios Más Caros
ggplot(df_rareza, aes(x = reorder(rareza, precio_medio), y = precio_medio)) +
  geom_col(fill = "#9467bd") +
  coord_flip() +
  labs(
    title = "Top Rarezas con Precios Más Caros",
    x = "Rareza",
    y = "Precio Medio (€)"
  ) +
  theme_minimal(base_size = 14)

# 7.6 Pokémon con Precios Más Caros
ggplot(df_pokemon_max, aes(x = reorder(carta, precio_maximo), y = precio_maximo)) +
  geom_col(fill = "#8c564b") +
  coord_flip() +
  labs(
    title = "Top 10 Pokémon con Precios Más Caros",
    x = "Pokémon",
    y = "Precio Máximo (€)"
  ) +
  theme_minimal(base_size = 14)

# 8) Desconexión
dbDisconnect(con)
