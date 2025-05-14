# 🎨 Explorador Interactivo y Predictor de Precios de Cartas Pokémon TCG

**Autor:** Victor Quero Valencia (Estudiante de Máster en Inteligencia Artificial)

---

## 📄 Resumen del Proyecto

Este repositorio contiene todo lo necesario para desplegar una app interactiva que permite:

- Explorar cartas de Pokémon TCG.
- Consultar sus precios en tiempo real (via BigQuery).
- Predecir el precio futuro con modelos de Machine Learning (MLP).

La aplicación está desarrollada con **Streamlit**, se conecta a **Google BigQuery** y utiliza un modelo MLP entrenado para la predicción.

---

## 🗂️ Estructura del Repositorio

```
Pokemon_TCG_Price_Predictor/
├── app.py                    # Aplicación principal (Streamlit)
├── requirements.txt         # Dependencias
├── model_files/             # Modelo ML + Preprocesadores
│   ├── saved_model.pb
│   ├── variables/
│   ├── ohe_mlp_cat.pkl
│   └── scaler_mlp_num.pkl
├── notebooks/               # Notebooks de Colab
│   ├── extraccion_limpieza.ipynb
│   ├── entrenamiento_MLP.ipynb
│   └── entrenamiento_LSTM.ipynb
├── orange/                  # Workflows de Orange (si aplica)
├── powerbi/                 # Dashboard .pbix
├── memoria_proyecto.txt     # [Opcional] Documentación
└── README.md                # Este archivo
```

> ✨ *Importante:* Asegúrate de que los nombres de archivo coincidan con los usados en `app.py`.

---

## 📈 Tecnologías Utilizadas

- **Lenguaje:** Python (ej. 3.10)
- **Framework Web:** Streamlit (ej. 1.30)
- **Cloud & Base de Datos:** Google BigQuery
- **Librerías ML y Data:**
  - TensorFlow (ej. 2.19.0)
  - Keras (ej. 3.9.2)
  - Scikit-learn
  - Pandas, NumPy, Joblib
- **Visualización:** Power BI, Orange, st-aggrid
- **Entorno de entrenamiento:** Google Colab

---

## ⚙️ Instalación y Configuración

### Prerrequisitos

- Python instalado
- Git instalado
- Entorno virtual (recomendado)
- Proyecto en Google Cloud con BigQuery habilitado
- Cuenta de Servicio con permisos en BigQuery

### Pasos:

1. **Clona el Repositorio**

```bash
git clone [URL_DEL_REPO]
cd Pokemon_TCG_Price_Predictor
```

2. **Activa un entorno virtual:**

```bash
python -m venv venv
source venv/bin/activate  # o .\venv\Scripts\activate en Windows
```

3. **Instala las dependencias:**

```bash
pip install -r requirements.txt
```

4. **Configura acceso a BigQuery con `secrets.toml`:**

Crea un archivo `.streamlit/secrets.toml` con los datos de tu cuenta de servicio:

```toml
[gcp_service_account]
type = "..."
project_id = "..."
private_key_id = "..."
private_key = "..."
client_email = "..."
client_id = "..."
... etc.
```

---

## 🔹 Ejecutar la App

Una vez todo esté listo:

```bash
streamlit run app.py
```

Esto abrirá la app en tu navegador.

---

## 🌐 Flujo General del Proyecto

1. **Datos almacenados en BigQuery.**
2. **Extracción, limpieza y entrenamiento del modelo (offline en notebooks).**
3. **Guardado de artefactos (`.pb`, `.pkl`) en `model_files/`.**
4. **Streamlit carga los modelos y se conecta a BQ.**
5. **UI permite explorar cartas y lanzar predicciones.**

---

## 🔧 Componentes Clave

- `app.py`: carga datos, modelos, muestra UI y realiza la predicción.
- `model_files/`: contiene SavedModel y preprocesadores (OHE, Scaler).
- `notebooks/`: desde la limpieza hasta el entrenamiento del MLP/LSTM.
- `powerbi/`: dashboard para visualización analítica.

---

## 🚀 Futuras Mejoras

- Soporte completo para modelo LSTM
- Elección de horizonte temporal para predicción
- Gráficas de evolución de precios en la UI
- Métricas de rendimiento del modelo
- Optimizar queries en BQ si hay latencias

---

## ⚠️ Errores Frecuentes y Soluciones

- `NameError`: Variable no definida → Verifica el orden del código
- `ValueError: unknown categories [...]`: Reentrena OHE con `handle_unknown='ignore'`
- `OSError: SavedModel no encontrado`: Verifica ruta y existencia de archivos
- `CUDA warnings`: No son críticos si usas CPU (Streamlit Cloud)
- `Shape mismatch`: Asegura que el total de features coincida con el modelo

---

## 🔒 Licencia

Este proyecto está licenciado bajo [MIT/Apache 2.0]. Consulta el archivo `LICENSE`.

---
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/VictorQ-code/Pokemon_TCG_Price_Predictor)
🚀 Proyecto realizado como parte del Máster en Inteligencia Artificial - Curso 2024/2025
