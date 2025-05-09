# Memoria del Proyecto: Explorador Interactivo y Predictor de Precios de Cartas Pokémon TCG

---

## Índice

1. [Introducción y Contexto](#1-introducción-y-contexto)  
2. [Adquisición y Preprocesamiento de Datos](#2-adquisición-y-preprocesamiento-de-datos)  
   - Metadatos  
   - Datos de Precios  
   - Bots de scraping  
   - API comunitaria  
   - Estructura en BigQuery  
   - Flujo de adquisición en tiempo real  
   - Preprocesamiento para ML  
3. [Desarrollo del Modelo de Machine Learning (MLP)](#3-desarrollo-del-modelo-de-machine-learning-mlp)  
   - Hipótesis inicial: dos modelos  
   - Estrategia final: MLP Cross-Sectional  
   - Detalles de la arquitectura  
4. [Implementación y Despliegue de la Aplicación](#4-implementación-y-despliegue-de-la-aplicación)  
   - Flujo automatizado de snapshots  
   - Por qué BigQuery  
   - Funcionamiento de la app Streamlit  
5. [Visualización y Análisis de Datos](#5-visualización-y-análisis-de-datos)  
   - Dentro de Streamlit  
   - Dashboard en Power BI  
6. [Trabajo Futuro](#6-trabajo-futuro)

---

## 1. Introducción y Contexto

El mercado de cartas coleccionables de **Pokémon TCG** ha crecido mucho recientemente, con fluctuaciones de precio según:

- Rareza  
- Set de expansión  
- Artista  
- Demanda del mercado  

> **Objetivo:**  
> Desarrollar una aplicación interactiva en la nube que permita:
> 1. Explorar la base de datos de cartas  
> 2. Predecir su precio futuro con **Machine Learning**  
>  
> Dirigida a coleccionistas, entusiastas e inversores.

---

## 2. Adquisición y Preprocesamiento de Datos

### Tipos de datos

1. **Metadatos de las cartas**  
   - Nombre, set, número, rareza, artista  
   - Tipo (Pokémon, Entrenador, Energía), supertipos, subtipos, tipos elementales  
2. **Datos de precios**  
   - Historial de mercado (idealmente mensual o semanal)

### Obtención de metadatos

- Fuente: dataset oficial de Pokémon TCG

### Desafío con datos de precios

- **Cardmarket**: full histórico tras muro de pago  
- Solución inicial: web scraping con tres bots:

  - **Walle**: buscaba por número y set → muy inconsistente  
  - **Firulai**: probaba 100 variantes de URL → demasiado lento  
  - **Mapache**: navegaba resultados y filtros → un poco lento, pero no puede sacar precios por canva

### API comunitaria

- Ofrece precios de los **últimos 30 días**  
- Plan:  
  1. Ejecutar manualmente un script cada mes  
  2. Guardar snapshot mensual  

### Estructura en Google BigQuery

- **Tablas de precios mensuales**: `monthly_YYYY_MM_DD`  
- **Tabla de metadatos**: `card_metadata`

### Flujo de adquisición en tiempo real

1. Conexión segura a BigQuery (Python + Service Account via Streamlit Secrets)  
2. Consulta a la tabla más reciente (`monthly_*`)  
3. Consulta a `card_metadata`  
4. Merge de precios + metadatos para visualización/predicción

### Preprocesamiento para ML (offline)

- **Feature Engineering**  
  - `price_t0_log = log(1 + price_t0)`  
  - `days_diff` (constante = 29.0)  
- **Transformaciones**
  - `days_diff` (variable, más peso conforme mas distancia entre días) 
  - `StandardScaler` → numéricas  
  - `OneHotEncoder` → categóricas  
  - Se entrena con `.fit()`, luego `.transform()` en inferencia

---

## 3. Desarrollo del Modelo de Machine Learning (MLP)

1. Exploración inicial con **Orange Data Mining**  
2. Hipótesis: dos modelos (alto valor > 50 €, bajo valor ≤ 50 €)  
   - Buen resultado en split 80/20, pero no robusto para horizonte futuro  
3. Enfoque definitivo: **MLP Cross-Sectional**  
   - **Entrada**: 2 numéricas escaladas + 7 categóricas OHE (4 863 columnas) → total **4 865 features**  
   - **Arquitectura**:  
     1. Capa densa (16 neuronas, ReLU, dropout 0.3)  
     2. Capa densa (16 neuronas, ReLU, dropout 0.3)  
     3. Salida (1 neurona, lineal)  
   - **Objetivo**: predecir `price_t1_log = log(1 + price_t1)`  
   - **Entrenamiento**: Colab + Huber loss, evaluación con MAE y MSE  
   - **Exportación**:  
     - Modelo SavedModel (TensorFlow v2)  
     - Preprocesadores en `.pkl` (Joblib)

> _Nota:_ El **Pipeline B: LSTM Time-Series** queda para Trabajo Futuro.

---

## 4. Implementación y Despliegue de la Aplicación

### Flujo automático de snapshots mensuales

1. Bot mensual → descarga CSV de la API (últimos 30 días)  
2. Carpeta en Google Drive → punto de recogida  
3. Proceso automatizado → ingesta en BigQuery (`monthly_YYYY_MM_DD`)

### ¿Por qué BigQuery?

- Procesamiento escalable  
- SQL flexible  
- Integración con GCP y BI

### App Streamlit

- **Carga de datos**  
  - Metadatos (`card_metadata`)  
  - Precios más recientes (`monthly_*`)  
- **Carga del modelo**  
  - `@st.cache_resource` para SavedModel + preprocesadores  
  - `tf.keras.layers.TFSMLayer` para inferencia  
- **Interfaz**  
  1. **Pantalla inicial**: cartas destacadas (`Special Illustration Rare`)  
  2. **Filtros**: tabla `st-aggrid` interactiva  
  3. **Detalle de carta**: imagen, metadatos, precio, enlace a Cardmarket  
  4. **Predicción**:  
     - Preparar input (log, days_diff, OHE, scaler)  
     - `model_layer(**input_dict)` → inversa `np.expm1()`  
     - Mostrar con `st.metric`

---

## 5. Visualización y Análisis de Datos

### En Streamlit

- **Tabla interactiva** (`st-aggrid`)  
- **Detalle de carta** + predicción  
- **Cartas destacadas** al inicio

### Dashboard en Power BI

- Conexión a BigQuery via SQL  
- Análisis estratégico:  
  - Tendencias históricas  
  - Comparativas por set/rareza  
  - Distribución de precios  
  - Detección de oportunidades

---

## 6. Trabajo Futuro

- **Más histórico**: ejecutar script mensual y enriquecer BigQuery  
- **Automatización total**: de snapshot → Drive → BigQuery → Streamlit  
- **Integrar LSTM**: usar modelo cuando `n_dates > 2`  
- **Horizonte dinámico**: elegir días de predicción en la UI  
- **Gráficos avanzados**: evolución + pronóstico en detalle de carta  
- **Chatbot asistente**: responder preguntas y consejos basados en datos  
- **Mejorar preprocesamiento**: datos faltantes, ingeniería de features  
- **Escalabilidad**: evaluar TensorFlow Serving o Vertex AI  
- **Nuevos factores**: popularidad online, relevancia en TCG, eventos

---  
