# Memoria del Proyecto: Explorador Interactivo y Predictor de Precios de Cartas Pokémon TCG

**Estudiante:** [Tu Nombre Aquí]
**Programa:** [Tu Programa de Master Aquí]
**Fecha:** [Fecha de Exposición/Entrega]
**Repositorio GitHub:** [Enlace a Tu Repositorio GitHub Aquí]

---

## 1. Introducción y Contexto del Proyecto

El mercado de cartas coleccionables de Pokémon TCG ha experimentado un crecimiento significativo en los últimos años, con fluctuaciones de precios influenciadas por diversos factores como la rareza, el set de expansión, el artista y la demanda del mercado. Para coleccionistas y entusiastas, tener acceso a información actualizada y estimaciones de precios futuros es de gran valor.

Este proyecto tiene como objetivo desarrollar una aplicación interactiva que permita explorar una base de datos de cartas Pokémon TCG y, fundamentalmente, ofrecer una predicción estimada del precio futuro de una carta seleccionada utilizando técnicas de **Machine Learning**.

La aplicación se concibe como una herramienta accesible y visualmente atractiva, desplegada en la nube para facilitar su uso por parte de la comunidad.

## 2. Adquisición y Preprocesamiento de Datos

La base de datos fundamental del proyecto reside en **Google BigQuery**, organizada en las siguientes estructuras:

*   **Tablas de Precios Mensuales** (`monthly_YYYY_MM_DD`): Contienen snapshots periódicos del precio de mercado (`cm_averageSellPrice`) para cada carta (`id`), asociados a una fecha específica.
*   **Tabla de Metadatos** (`card_metadata`): Almacena información estática y semi-estática sobre cada carta, incluyendo su identificador (`id`), nombre (`name`), artista (`artist`), rareza (`rarity`), set de expansión (`set_name`), tipos (`types`), supertipo (`supertype`) y subtipos (`subtypes`), así como URLs de imágenes y plataformas de venta.

El proceso de adquisición de datos para la aplicación implica:

*   Conexión segura a `BigQuery` utilizando el cliente Python oficial y autenticación vía Service Account gestionada a través de `Streamlit Secrets`.
*   Consulta dinámica a la tabla de precios más reciente (`monthly_*` con sufijo más reciente) para obtener los precios actuales de las cartas.
*   Consulta a la tabla `card_metadata` para obtener las características descriptivas de cada carta.
*   Unión (`merge`) de los datos de precios con los metadatos en memoria para crear un conjunto de datos coherente para la visualización y la predicción.

El preprocesamiento de datos para el **modelo de Machine Learning** (realizado offline en un entorno como Google Colab durante la fase de entrenamiento) incluyó:

*   **Feature Engineering:** Creación de características temporales y de precio relevantes, como el logaritmo del precio en el tiempo actual (`price_t0_log = log(1 + price_t0)`) y la diferencia en días entre snapshots (`days_diff`). Para el modelo `MLP` implementado, `days_diff` fue un valor constante determinado por la diferencia entre los dos snapshots usados para su entrenamiento específico. Para el modelo `LSTM` (ver Trabajo Futuro), se diseñó el uso de secuencias de precios y diferencias de días variables (`days_since_prev`).
*   **Transformaciones:** Aplicación de `StandardScaler` a las características numéricas (`price_t0_log`, `days_diff`) y `OneHotEncoder` a las características categóricas (`artist_name`, `pokemon_name`, `rarity`, `set_name`, `types`, `supertype`, `subtypes`). Es crucial que el `OneHotEncoder` se entrenara (método `.fit()`) con los valores exactos (y su representación, ej. si `types`/`subtypes` eran listas o strings) tal como vienen de la base de datos para poder replicar la transformación (`.transform()`) fielmente en la inferencia.

## 3. Desarrollo del Modelo de Machine Learning (MLP)

Durante la fase de desarrollo, se exploraron dos pipelines de modelado:

*   **Pipeline A: MLP (Multi-Layer Perceptron) Cross-Sectional:** Diseñado para predecir el precio futuro basándose en un único snapshot de precio actual (`price_t0`) y metadatos. Este pipeline es el que se ha implementado y desplegado en la aplicación `Streamlit` actual.
    *   **Características de Entrada:** Las características preprocesadas consisten en la concatenación de las 2 características numéricas escaladas (`price_t0_log`, `days_diff`) y las 7 características categóricas codificadas mediante One-Hot Encoding (que resultan en 4863 columnas One-Hot, según la inspección del encoder guardado). El número total de características de entrada para el modelo es 4865.
    *   **Arquitectura:** Un `MLP` simple con capas densas (`Dense`) y `Dropout` para regularización. La arquitectura específica implementada es: Capa de entrada (4865 neuronas, implícita por la forma de entrada) -> Capa `Dense` (16 neuronas, activación `ReLU`, `dropout` 0.3) -> Capa `Dense` (16 neuronas, activación `ReLU`, `dropout` 0.3) -> Capa de Salida (1 neurona, activación lineal).
    *   **Variable Objetivo (y):** El modelo fue entrenado para predecir el logaritmo transformado del precio futuro (`price_t1_log = log(1 + price_t1)`).
    *   **Entrenamiento:** El modelo se entrenó en Google Colab utilizando Huber loss y evaluado con Mean Absolute Error (`MAE`) y `MSE`. Se aplicó sample weighting si se consideró necesario.
    *   **Exportación:** El modelo entrenado y los preprocesadores (`StandardScaler`, `OneHotEncoder`) se exportaron y guardaron. El modelo TensorFlow se guardó en formato `SavedModel` (`v2`). Los preprocesadores se guardaron utilizando `Joblib` (`.pkl`).

*   **Pipeline B: LSTM (Long Short-Term Memory) Time-Series:** Diseñado para modelar secuencias de precios históricos cuando se dispone de más de dos snapshots. Este pipeline se exploró conceptualmente y se planea integrar en el **Trabajo Futuro**.

## 4. Implementación y Despliegue de la Aplicación

La aplicación interactiva se ha desarrollado utilizando `Streamlit` y se despliega en `Streamlit Cloud` directamente desde un repositorio de `GitHub`.

La arquitectura de la aplicación incluye:

*   **Carga de Datos:** Al inicio de la aplicación y con cada interacción que cambia los filtros, se consulta `BigQuery` para obtener los metadatos (`all_card_metadata_df`) y los datos de precios actuales (`results_df`) basándose en la selección del usuario en la sidebar.
*   **Carga del Modelo Local:** El modelo `MLP` (formato `SavedModel`) y sus preprocesadores (`.pkl`) se incluyen en el repositorio de GitHub (`model_files/`). Se cargan en memoria al inicio de la aplicación utilizando las funciones `@st.cache_resource` para asegurar que solo se carguen una vez por sesión. Para el `SavedModel` en Keras 3, se utiliza `tf.keras.layers.TFSMLayer` para cargarlo como una capa funcional.
*   **Lógica de Visualización:** La aplicación presenta diferentes secciones dependiendo del estado:
    *   **Carga Inicial (sin filtros):** Muestra una selección aleatoria de cartas destacadas (rareza 'Special Illustration Rare') como imágenes con el set en el pie de foto (utilizando `st.image`). La tabla principal de resultados se oculta. Se selecciona automáticamente una carta aleatoria con precio para mostrar sus detalles.
    *   **Al Aplicar Filtros:** La sección de cartas destacadas se oculta. Se muestra la tabla principal de resultados ("Resultados de Cartas") utilizando `st-aggrid` para una visualización interactiva y paginada. La tabla muestra las cartas que coinciden con los filtros de la sidebar.
    *   **Sección de Detalle de Carta:** Esta sección siempre está visible cuando una carta está seleccionada (ya sea por la selección automática inicial, un clic en la tabla AgGrid). Muestra la imagen de la carta, metadatos clave (nombre, set, rareza, artista, etc.), precio actual y enlaces a plataformas de venta.
*   **Predicción en Inferencia:** Al seleccionar una carta en la sección de detalles (si tiene precio actual y el modelo cargó):
    *   Se extraen los datos relevantes (`price`, metadatos) de la carta seleccionada (`card_data_series`).
    *   Se replica el exacto pipeline de preprocesamiento del entrenamiento:
        *   Cálculo de `price_t0_log = log(1 + price_actual)`.
        *   Uso del `days_diff` constante (29.0) del entrenamiento del `MLP`.
        *   Mapeo de las 7 columnas categóricas a los nombres esperados por el `OHE` (`artist_name -> artist`, `pokemon_name -> pokemon_name`, etc.), manejando la conversión a string y NaNs/listas según se procesaron en el entrenamiento.
        *   Aplicación del `scaler_local_preprocessor` a las numéricas.
        *   Aplicación del `ohe_local_preprocessor` a las categóricas.
        *   Concatenación de las características preprocesadas en el orden correcto para formar el array de entrada final (shape `(1, 4865)`).
    *   Se llama a la capa `TFSMLayer` con el input preparado utilizando `model_layer(**input_dict)`.
    *   La predicción cruda obtenida se post-procesa aplicando `np.expm1()` para revertir la transformación logarítmica y obtener el precio en euros.
    *   La predicción final se muestra en la UI utilizando `st.metric`, incluyendo la diferencia con el precio actual.

## 5. Desafíos y Soluciones Implementadas

Durante el desarrollo, se enfrentaron varios desafíos técnicos, comunes al integrar modelos ML en aplicaciones web:

*   **Serialización y Carga de Artefactos ML:** Asegurar que el modelo TensorFlow (`SavedModel`) y los preprocesadores de Scikit-learn (`.pkl`) se guardaran y cargaran correctamente en el entorno de `Streamlit Cloud`. La solución implicó incluir los archivos en el repositorio de GitHub (`model_files/`) y usar `joblib.load` y `tf.keras.layers.TFSMLayer`.
*   **Compatibilidad de Keras 3 con SavedModel:** El cambio en la API de `tf.keras.models.load_model` en Keras 3 para `SavedModel` requirió el uso de `tf.keras.layers.TFSMLayer`. Depurar la forma correcta de llamar a esta capa (pasando el diccionario con `**`) fue clave, utilizando `saved_model_cli` para inspeccionar la firma del modelo.
*   **Consistencia del Preprocesamiento:** El error `ValueError: The feature names should match...` del scaler/`OHE` al no recibir las columnas con los nombres y en el formato exacto del entrenamiento fue un desafío importante. Se resolvió mediante una inspección detallada de los objetos scaler y `OHE` guardados (usando código de inspección en un notebook) para determinar los nombres de columnas esperados, y ajustando la lógica de mapeo en la función `predict_price_with_local_tf_layer` para replicar fielmente el pipeline de entrenamiento, incluyendo el manejo de NaNs y la conversión a string para categóricas.
*   **Lógica de Interfaz de Usuario en Streamlit:** Controlar la visibilidad de diferentes secciones (destacadas, tabla, detalles) basado en la interacción del usuario (carga inicial vs filtros aplicados) y mantener el estado de la carta seleccionada en `st.session_state` requirió una estructuración cuidadosa del código y el uso de `st.rerun()`. La implementación de imágenes destacadas visuales no clicables y la ocultación de la tabla por defecto en la carga inicial fueron ajustes finos en esta lógica.
*   **Latencia y Caché:** El uso extensivo de `@st.cache_data` y `@st.cache_resource` fue fundamental para minimizar las consultas a `BigQuery` y la recarga del modelo/preprocesadores, mejorando el rendimiento percibido de la aplicación.

## 6. Resultados y Evaluación (MLP Actual)

La aplicación implementada permite explorar la base de datos y obtener una predicción de precio futuro utilizando el modelo `MLP` entrenado con 2 snapshots de precios.

La calidad de la predicción del modelo `MLP` se evaluó durante el entrenamiento utilizando métricas como `MAE` en un conjunto de validación. [**Opcional:** Incluye aquí el valor del MAE que obtuviste en tu entrenamiento, ej: "Se obtuvo un MAE en el conjunto de validación de aproximadamente **[Valor MAE]** euros."]

Es importante notar que, dado que este `MLP` fue entrenado con un `days_diff` constante (29.0), su capacidad para predecir con otros horizontes temporales es limitada. La predicción refleja la estimación para un horizonte de 29 días (o el valor de `DEFAULT_DAYS_DIFF_FOR_PREDICTION`).

## 7. Trabajo Futuro

Este proyecto sienta las bases para futuras mejoras y expansiones:

*   **Integración del Modelo LSTM:** Adaptar la aplicación `Streamlit` para cargar y utilizar el modelo `LSTM` (Pipeline B) cuando se disponga de suficientes snapshots de precios históricos para una carta (`n_dates > 2`). Esto implicaría modificar la lógica de predicción para construir secuencias de entrada y usar el modelo `LSTM` en lugar del `MLP`.
*   **Selección Dinámica del Horizonte de Predicción:** Si se implementa el `LSTM` (o se reentrena un `MLP` con `days_diff` variable), permitir al usuario seleccionar el número de días hacia el futuro para la predicción.
*   **Mejora del Pipeline de Preprocesamiento:** Investigar si el manejo actual de `types` y `subtypes` (especialmente si son listas) es el óptimo para el modelo, o si una estrategia diferente (ej. ingeniería de características adicionales a partir de ellos) podría mejorar la precisión.
*   **Visualizaciones Avanzadas:** Añadir gráficos de series temporales mostrando el historial de precios de una carta seleccionada, comparando el precio real con las predicciones a lo largo del tiempo.
*   **Escalabilidad y Despliegue:** Explorar opciones más avanzadas para la gestión de modelos (ej. TensorFlow Serving, Vertex AI Endpoints) si la carga de `SavedModel` directamente en `Streamlit` no es suficiente para cargas de trabajo mayores.
*   **Consideración de Otros Factores:** Incorporar otras características potenciales que podrían influir en el precio, como la popularidad actual del Pokémon, su jugabilidad en el TCG, eventos especiales, etc.

## 8. Conclusión

Este proyecto ha demostrado la viabilidad de construir una aplicación web interactiva en `Streamlit`, integrada con `Google BigQuery` y un modelo de `Machine Learning` local (`MLP`) para la predicción de precios en tiempo (casi) real. Se superaron desafíos clave en la carga de modelos `SavedModel` en Keras 3 y la replicación precisa del pipeline de preprocesamiento. La arquitectura implementada proporciona una base sólida para futuras iteraciones, incluyendo la integración de modelos más complejos como el `LSTM` para mejorar la capacidad de predicción ante una mayor disponibilidad de datos históricos.

---

**Nota:** Recuerda reemplazar los placeholders `[Tu Nombre Aquí]`, `[Tu Programa de Master Aquí]`, `[Fecha de Exposición/Entrega]`, `[Enlace a Tu Repositorio GitHub Aquí]` y opcionalmente incluir el valor del MAE si lo tienes disponible.
