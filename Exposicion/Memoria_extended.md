# Memoria del Proyecto: Explorador Interactivo y Predictor de Precios de Cartas Pokémon TCG


---

## 1. Introducción y Contexto del Proyecto

El mercado de cartas coleccionables de Pokémon TCG ha experimentado un crecimiento significativo en los últimos años, con fluctuaciones de precios influenciadas por diversos factores como la rareza, el set de expansión, el artista y la demanda del mercado. Para coleccionistas y entusiastas, tener acceso a información actualizada y estimaciones de precios futuros es de gran valor.

Este proyecto tiene como objetivo desarrollar una aplicación interactiva que permita explorar una base de datos de cartas Pokémon TCG y, fundamentalmente, ofrecer una predicción estimada del precio futuro de una carta seleccionada utilizando técnicas de **Machine Learning** enfocado a no solo coleccionistas y entusiastas sino a inversores que quieran diversificar su portfolio con otro tipo de inversiones.

La aplicación se concibe como una herramienta accesible y visualmente atractiva, desplegada en la nube para facilitar su uso por parte de la comunidad.

## 2. Adquisición y Preprocesamiento de Datos


El proyecto requería fundamentalmente dos tipos de datos:

1.  **Metadatos de las Cartas:** Información estática o semi-estática como nombre, set de expansión, número dentro del set, rareza, artista, tipo de carta (Pokémon, Entrenador, Energía), tipos elementales asociados, supertipos y subtipos.
2.  **Datos de Precios:** Historial de precios de mercado, idealmente con granularidad temporal (ej. mensual, semanal) para poder modelar su evolución.

Los **metadatos** se obtuvieron de una fuente fiable: el dataset proporcionado por la página oficial de Pokémon TCG, que ofrece información estructurada sobre las cartas publicadas.

La adquisición de **datos de precios**, particularmente históricos, presentó un desafío significativo. La principal y más completa fuente de datos históricos de precios para Pokémon TCG es la plataforma Cardmarket. Sin embargo, el acceso al historial detallado de precios está tras un muro de pago, imposibilitando la descarga masiva y automatizada para un proyecto de este tipo.

Ante esta restricción, se exploró la posibilidad de obtener los precios mediante web scraping. Se desarrollaron varios bots con diferentes estrategias:

*   **Bot "Walle":** Intentaba localizar las cartas utilizando el número dentro del set y el nombre del set. Esta estrategia falló debido a la complejidad e inconsistencia del algoritmo interno de búsqueda de la página web, que dificultaba la identificación fiable y unívoca de cada carta.
*   **Bot "Firulai":** Buscaba aprovechar la estructura de los URLs de las cartas. Si bien muchas URLs compartían un patrón, se encontró que en varios sets, la URL de una carta incluía un parámetro numérico aleatorio (entre 1 y 100) asociado al Pokémon. Dada la existencia de más de 19,000 cartas, intentar probar 100 variantes de URL por cada carta se convirtió en un proceso prohibitivamente ineficiente y extremadamente lento.
*   **Bot "Mapache":** Optó por un enfoque más robusto utilizando los filtros de búsqueda de la propia página y navegando a través de las páginas de resultados. Este bot logró obtener precios *actuales* con alta consistencia, incluso si el proceso era un poco más lento. No obstante, al intentar acceder al historial de precios, la página implementaba medidas anti-bot agresivas: la posición en el código del gráfico de historial cambiaba dinámicamente en cada recarga, y la información de precios estaba incrustada dentro de un elemento `<canvas>`, haciendo imposible extraer los datos numéricamente mediante scraping estándar.

Tras estos intentos fallidos de obtener el historial completo mediante scraping, se encontró una **comunidad que proporcionaba acceso a una API** con datos de precios. Esta API, aunque no ofrecía el historial completo deseado, proporcionaba de manera fiable los precios de los **últimos 30 días**.

Este hallazgo definió la estrategia de adquisición de datos a largo plazo: para construir un historial de precios significativo (necesario para modelos de series temporales), se requeriría la ejecución **manual** de un script propio a principios de cada mes para obtener y almacenar un snapshot mensual de los precios disponibles a través de esta API comunitaria.

La base de datos fundamental del proyecto reside en **Google BigQuery**, organizada en las siguientes estructuras, que almacenan los datos obtenidos (metadatos y los snapshots de precios mensuales acumulados):

*   **Tablas de Precios Mensuales** (`monthly_YYYY_MM_DD`): Contienen los snapshots periódicos del precio de mercado (`cm_averageSellPrice`) para cada carta (`id`), asociados a una fecha específica.
*   **Tabla de Metadatos** (`card_metadata`): Almacena la información descriptiva sobre cada carta (`id`, `name`, `artist`, `rarity`, `set_name`, `types`, `supertype`, `subtypes`, etc.).

El proceso de adquisición de datos *para la aplicación en tiempo real* implica:

*   Conexión segura a `BigQuery` utilizando el cliente Python oficial y autenticación vía Service Account gestionada a través de `Streamlit Secrets`.
*   Consulta dinámica a la tabla de precios más reciente (`monthly_*` con sufijo más reciente) para obtener los precios actuales de las cartas.
*   Consulta a la tabla `card_metadata` para obtener las características descriptivas de cada carta.
*   Unión (`merge`) de los datos de precios con los metadatos en memoria para crear un conjunto de datos coherente para la visualización y la predicción.

El preprocesamiento de datos para el **modelo de Machine Learning** (realizado offline en un entorno como Google Colab durante la fase de entrenamiento) incluyó:

*   **Feature Engineering:** Creación de características temporales y de precio relevantes, como el logaritmo del precio en el tiempo actual (`price_t0_log = log(1 + price_t0)`) y la diferencia en días entre snapshots (`days_diff`). Para el modelo `MLP` implementado, `days_diff` fue un valor constante (29.0) determinado por la diferencia entre los dos snapshots usados para su entrenamiento específico. Para el modelo `LSTM` (ver Trabajo Futuro), se diseñó el uso de secuencias de precios y diferencias de días variables (`days_since_prev`).
*   **Transformaciones:** Aplicación de `StandardScaler` a las características numéricas (`price_t0_log`, `days_diff`) y `OneHotEncoder` a las características categóricas (`artist_name`, `pokemon_name`, `rarity`, `set_name`, `types`, `supertype`, `subtypes`). Es crucial que el `OneHotEncoder` se entrenara (método `.fit()`) con los valores exactos (y su representación, ej. si `types`/`subtypes` eran listas o strings) tal como vienen de la base de datos para poder replicar la transformación (`.transform()`) fielmente en la inferencia.
## 3. Desarrollo del Modelo de Machine Learning (MLP)

Durante la fase de desarrollo, la exploración inicial de potenciales arquitecturas y estrategias de modelado se llevó a cabo utilizando herramientas de minería de datos como **Orange Data Mining**. Estos análisis exploratorios sugirieron que las cartas de precios significativamente diferentes (por ejemplo, por encima y por debajo de un umbral de 50 euros) exhibían patrones distintos.

Esta observación llevó a la hipótesis de entrenar **dos modelos separados**: uno optimizado para cartas de alto valor (> 50€) y otro para cartas de bajo valor (<= 50€). Las pruebas iniciales de este enfoque dual mostraron resultados muy prometedores al evaluar el rendimiento sobre el conjunto de datos de entrenamiento disponible (split 80/20), sugiriendo que estratificar los datos por precio podría mejorar la precisión.

Sin embargo, la limitación fundamental del proyecto residía en la **falta de un histórico de precios extenso** (como se detalló en la Sección 2), disponiendo inicialmente solo de dos snapshots de precios. El objetivo principal del proyecto es la **predicción de precios futuros**, no la predicción del precio actual sobre datos existentes. Se determinó que, a pesar de los buenos resultados *sobre el split de entrenamiento*, el enfoque de dos modelos no era robusto ni adecuado para predecir un horizonte futuro con tan poca profundidad temporal real.

Ante esta restricción de datos históricos, la estrategia se reorientó hacia el desarrollo de un modelo que pudiera aprovechar al máximo la información disponible en dos snapshots para realizar una predicción a un horizonte fijo. Esto condujo a la selección e implementación del modelo **MLP (Multi-Layer Perceptron) Cross-Sectional**, diseñado para predecir el precio futuro basándose en un único snapshot de precio actual (`price_t0`) y metadatos. La calidad de los modelos fue realizada mediante un EDA para ver el fallo porcentual, el rango y error medio del modelo para decidir si se continuaba o ser iban variando las características del mismo.

Las características de este modelo implementado son:

*   **Características de Entrada:** Las características preprocesadas consisten en la concatenación de las 2 características numéricas escaladas (`price_t0_log`, `days_diff`) y las 7 características categóricas codificadas mediante One-Hot Encoding (que resultan en 4863 columnas One-Hot, según la inspección del encoder guardado). El número total de características de entrada para el modelo es 4865.
*   **Arquitectura:** Un `MLP` simple con capas densas (`Dense`) y `Dropout` para regularización. La arquitectura específica implementada es: Capa de entrada (4865 neuronas, implícita por la forma de entrada) -> Capa `Dense` (16 neuronas, activación `ReLU`, `dropout` 0.3) -> Capa `Dense` (16 neuronas, activación `ReLU`, `dropout` 0.3) -> Capa de Salida (1 neurona, activación lineal).
*   **Variable Objetivo (y):** El modelo fue entrenado para predecir el logaritmo transformado del precio futuro (`price_t1_log = log(1 + price_t1)`).
*   **Entrenamiento:** El modelo se entrenó en Google Colab utilizando Huber loss y evaluado con Mean Absolute Error (`MAE`) y `MSE`. Se aplicó sample weighting si se consideró necesario.
*   **Exportación:** El modelo entrenado y los preprocesadores (`StandardScaler`, `OneHotEncoder`) se exportaron y guardaron. El modelo TensorFlow se guardó en formato `SavedModel` (`v2`). Los preprocesadores se guardaron utilizando `Joblib` (`.pkl`).

_(Nota: El **Pipeline B: LSTM (Long Short-Term Memory) Time-Series** mencionado previamente, diseñado para secuencias de precios históricos, se considera parte del **Trabajo Futuro** del proyecto debido a la actual limitación de datos históricos disponibles para su entrenamiento y aplicación efectiva.)_


## 4. Implementación y Despliegue de la Aplicación

La arquitectura de datos del proyecto se diseñó para facilitar la ingesta periódica y el acceso eficiente. Los snapshots mensuales de precios obtenidos a través de la API (como se detalla en la Sección 2) siguen un flujo automatizado:

*   Un **script o bot** es responsable de consumir la API a principios de cada mes y descargar los datos de precios disponibles (los últimos 30 días).
*   Estos archivos descargados se almacenan temporalmente en una ubicación intermedia, como una carpeta compartida en **Google Drive**. Esto actúa como un punto de recogida centralizado. Además de en local descargando y almacenando los csv en un entorno físico para tener más seguridad.
*   Un **segundo proceso automatizado** se encarga de monitorizar esta ubicación en Google Drive, recoger los nuevos archivos de precios y realizar la ingesta en **Google BigQuery**. Cada snapshot mensual se carga en una tabla separada (`monthly_YYYY_MM_DD`), manteniendo un registro histórico de los precios a lo largo del tiempo.

**Google BigQuery** se seleccionó como el repositorio central de datos por sus capacidades de procesamiento escalable, su integración con el ecosistema de Google Cloud Platform y, fundamentalmente, por su interfaz de consulta basada en **SQL**. Esto permite una extracción de datos flexible y eficiente para diversos propósitos, incluyendo:

*   Servir datos a la aplicación interactiva `Streamlit`.
*   Facilitar el análisis exploratorio y la preparación de datos para el entrenamiento de modelos (realizado offline).
*   Permitir la conexión con herramientas de Business Intelligence como **Power BI** u otras, para visualizaciones y análisis más profundos sobre el histórico de datos (disponible localmente si se descarga de BigQuery).

La **aplicación interactiva** se ha desarrollado utilizando `Streamlit` y se despliega en `Streamlit Cloud` directamente desde un repositorio de `GitHub`. La aplicación interactúa con esta infraestructura de datos de la siguiente manera:

*   **Carga de Datos para la App:** Al inicio de la aplicación y con cada interacción que cambia los filtros de visualización (ubicados en la sidebar), la aplicación consulta `Google BigQuery`. Utiliza el cliente Python oficial y autenticación segura vía `Streamlit Secrets` para acceder a los datos. Se realizan consultas dinámicas para obtener:
    *   Todos los metadatos de cartas (`all_card_metadata_df`) desde la tabla `card_metadata`.
    *   Los datos de precios **más recientes** disponibles, consultando la tabla `monthly_*` con el sufijo de fecha más reciente.
    *   Los datos de precios se unen (`merge`) con los metadatos en memoria para crear un conjunto de datos (`results_df`) coherente y enriquecido, listo para la visualización y la selección de cartas.
*   **Carga del Modelo Local:** El modelo `MLP` (formato `SavedModel`) y sus preprocesadores (`.pkl`), que fueron exportados offline tras el entrenamiento (ver Sección 3), se incluyen directamente en el repositorio de GitHub (`model_files/`). Se cargan en memoria al inicio de la aplicación utilizando las funciones `@st.cache_resource` para asegurar que solo se carguen una vez por sesión de usuario, optimizando el rendimiento. Para cargar el `SavedModel` en el entorno de Keras 3 compatible con Streamlit, se utiliza `tf.keras.layers.TFSMLayer`, tratándolo como una capa funcional.
*   **Lógica de Visualización:** La aplicación presenta diferentes secciones dependiendo del estado:
    *   **Carga Inicial (sin filtros aplicados):** Para una experiencia de usuario inicial más visual, se muestra una selección aleatoria de cartas destacadas (filtrando por rareza 'Special Illustration Rare') como imágenes grandes (`st.image`), con el set en el pie de foto. La tabla principal de resultados (`st-aggrid`) se oculta en este estado. Se selecciona automáticamente una carta aleatoria *que tenga precio registrado* para mostrar sus detalles por defecto.
    *   **Al Aplicar Filtros (en la sidebar):** La sección de cartas destacadas se oculta. Se muestra la tabla principal de resultados ("Resultados de Cartas") utilizando la librería `st-aggrid`, que proporciona una visualización interactiva, paginada y con capacidades de búsqueda/ordenación sobre el `results_df` filtrado. La tabla muestra únicamente las cartas que coinciden con los criterios seleccionados por el usuario en la sidebar.
    *   **Sección de Detalle de Carta:** Esta sección es un panel fijo visible cuando una carta está seleccionada (ya sea por la selección automática inicial o por un clic del usuario en la tabla AgGrid). Muestra la imagen ampliada de la carta, metadatos clave relevantes para el coleccionista (nombre, set, rareza, artista, etc.), su precio actual (si está disponible en los datos cargados), y enlaces externos a plataformas de venta como Cardmarket.
*   **Predicción en Inferencia:** Al seleccionar una carta en la sección de detalles, si esta carta tiene un precio actual registrado en el último snapshot de datos cargado *y* el modelo `MLP` se ha cargado correctamente:
    *   Se extraen los datos relevantes (`price`, metadatos descriptivos) de la carta seleccionada (`card_data_series`) directamente del DataFrame cargado.
    *   Se prepara la entrada para el modelo replicando **exactamente** el pipeline de preprocesamiento utilizado durante el entrenamiento offline (descrito en Sección 2). Esto incluye:
        *   Cálculo de la transformación logarítmica del precio actual: `price_t0_log = log(1 + price_actual)`.
        *   Uso del valor `days_diff` constante (29.0) con el que el modelo `MLP` fue entrenado.
        *   Mapeo cuidadoso de las columnas categóricas de la carta seleccionada (`artist`, `pokemon_name`, `rarity`, `set_name`, `types`, `supertype`, `subtypes`) a los nombres y formatos esperados por el `OneHotEncoder` entrenado. Esto incluye manejar posibles valores nulos (NaN) y la conversión a string, replicando el tratamiento del entrenamiento.
        *   Aplicación del `scaler_local_preprocessor` (cargado localmente) a las características numéricas preparadas.
        *   Aplicación del `ohe_local_preprocessor` (cargado localmente) a las características categóricas preparadas.
        *   Concatenación final de todas las características preprocesadas en el orden correcto para formar el array de entrada final para el modelo, con la forma esperada `(1, 4865)`.
    *   Se realiza la predicción llamando a la capa `TFSMLayer` cargada (`model_layer`), pasándole el input preparado como un diccionario utilizando `model_layer(**input_dict)`.
    *   La predicción cruda obtenida (que está en escala logarítmica) se post-procesa aplicando la inversa de la transformación: `np.expm1()` para obtener el precio estimado en euros.
    *   La predicción de precio futuro se muestra en la interfaz de usuario, típicamente utilizando un componente `st.metric` para resaltar el valor y, opcionalmente, la diferencia o porcentaje de cambio respecto al precio actual.

## 5. Visualización y Análisis de Datos

La información recopilada y almacenada en Google BigQuery se utiliza de diversas maneras para facilitar tanto la exploración interactiva como el análisis estratégico, orientado a la toma de decisiones, como la identificación de oportunidades de inversión.

Dentro de la **aplicación interactiva Streamlit**, la visualización se centra en la exploración a nivel de carta individual y de colecciones filtradas:

*   **Tabla Interactiva (`st-aggrid`):** Presenta los datos de las cartas que coinciden con los criterios de búsqueda y filtrado del usuario. Permite ordenar, buscar y paginar de manera eficiente a través del catálogo, mostrando información clave como nombre, set, rareza y precio actual. Su propósito es facilitar la navegación y descubrimiento de cartas de interés.
*   **Sección de Detalle de Carta:** Al seleccionar una carta, se muestra una vista ampliada que incluye la imagen de la carta, sus metadatos completos y su precio actual. Esta sección es crucial porque también presenta la **predicción de precio futuro** generada por el modelo ML, ofreciendo una estimación directa y accesible para el usuario interesado en el potencial de revalorización.
*   **Cartas Destacadas (Carga Inicial):** Aunque no son interactivas en sí mismas, las imágenes de cartas destacadas en la pantalla inicial sirven como un gancho visual y una forma rápida de mostrar ejemplos de cartas interesantes en la base de datos.

Además de la aplicación interactiva, los datos centralizados en **Google BigQuery** son accesibles mediante **SQL**, lo que permite su fácil integración con herramientas de Business Intelligence (BI) como **Power BI**.

Mediante la conexión de Power BI a BigQuery, se ha creado un **dashboard de análisis** con el objetivo de proporcionar una visión más agregada y profunda de los datos. Este dashboard facilita:

*   **Análisis de Tendencias Históricas:** Visualización de la evolución de precios a lo largo del tiempo para sets, rarezas, o grupos de cartas específicos (a medida que se acumule histórico en BigQuery).
*   **Comparativas de Rendimiento:** Analizar qué sets o rarezas han tenido un mejor desempeño en el mercado.
*   **Distribución de Precios:** Entender la dispersión de precios dentro de un set o a través de diferentes rarezas.
*   **Identificación de Oportunidades:** Utilizar los datos para identificar cartas o sets que podrían estar infravalorados o mostrando tendencias de crecimiento, sirviendo de base para decisiones de inversión o coleccionismo más informadas.

Mientras que la aplicación Streamlit es una herramienta de consulta y predicción a demanda, el dashboard de Power BI actúa como una herramienta de análisis estratégico, permitiendo explorar patrones y tendencias a un nivel más granular y agregado, complementando así la funcionalidad del proyecto.

## 6. Trabajo Futuro

Este proyecto sienta las bases para futuras mejoras y expansiones significativas:

*   **Continuación y Expansión de la Adquisición Histórica:** Mantener y optimizar la ejecución mensual del script de adquisición de precios para construir gradualmente un historial de datos más profundo y de mayor granularidad en BigQuery. Este historial es crucial para entrenar y validar modelos de series temporales más robustos y mejorar la precisión general.
*   **Automatización Completa del Flujo de Datos:** Desarrollar mecanismos para automatizar por completo el pipeline de datos, desde la obtención de los snapshots mensuales de la API (eliminando la necesidad de ejecución manual) hasta la ingesta automática en BigQuery y la actualización periódica o a demanda de los datos disponibles para la aplicación Streamlit.
*   **Integración y Uso del Modelo LSTM:** Adaptar la aplicación Streamlit para cargar y utilizar el modelo `LSTM` (mencionado conceptualmente en la sección 3) a medida que se acumule suficiente historial de precios (`n_dates > 2`). Esto implicaría modificar la lógica de predicción para construir secuencias de entrada adecuadas para el `LSTM` y considerar la posibilidad de utilizar dinámicamente el modelo más apropiado según los datos históricos disponibles para una carta.
*   **Selección Dinámica del Horizonte de Predicción:** Si se implementa el `LSTM` o se reentrena un `MLP` con `days_diff` variable utilizando más histórico, permitir al usuario seleccionar interactivamente el número de días hacia el futuro para el cual desea la predicción, ofreciendo mayor flexibilidad y personalización.
*   **Visualizaciones Gráficas Avanzadas y de Predicción:** Integrar en la sección de detalle de la carta (dentro de la aplicación Streamlit) gráficos interactivos que muestren la evolución histórica del precio de la carta (utilizando los datos acumulados en BigQuery) y superpongan la predicción del modelo, ya sea como un punto futuro, una línea de pronóstico o un intervalo de confianza, facilitando la comprensión visual del potencial de la carta.
*   **Implementación de un Chatbot Asistente:** Explorar la integración de un chatbot dentro de la aplicación que pueda responder preguntas comunes sobre cartas, sets, rarezas, tendencias generales del mercado, y ofrecer consejos básicos basados en los datos disponibles y las predicciones del modelo (por ejemplo, "¿Esta carta es una buena inversión a corto plazo según la predicción?").
*   **Mejora Continua del Pipeline de Preprocesamiento:** Refinar el tratamiento de datos faltantes, valores inconsistentes (especialmente en campos como `types` y `subtypes`), y explorar técnicas de ingeniería de características más avanzadas a medida que se disponga de más datos y se desarrollen modelos más complejos.
*   **Escalabilidad y Optimización del Despliegue:** Evaluar la necesidad de migrar componentes del sistema (como la carga de modelos) a soluciones más robustas y escalables (ej. TensorFlow Serving, Vertex AI Endpoints) si la carga de usuarios o el tamaño de los modelos aumentan, optimizando el rendimiento y la latencia.
*   **Exploración de Factores de Precio Adicionales:** Investigar la posibilidad de incorporar otras fuentes de datos o características que podrían influir en el precio, como la popularidad online del Pokémon, su relevancia en el juego competitivo (TCG), anuncios oficiales, eventos comunitarios, etc.
---


