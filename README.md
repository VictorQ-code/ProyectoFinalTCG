# Pokémon TCG Data Scraper & Dashboard 

Este proyecto integra web scraping, automatización avanzada y análisis visual interactivo sobre el mercado de cartas Pokémon TCG. El objetivo principal es recopilar información precisa y generar insights útiles para coleccionistas, jugadores y analistas.

---

## 🚩 Objetivos Principales

- Automatizar la extracción de datos desde **Cardmarket** y **Pokestats**.
- Realizar limpieza eficiente y precisa de datos masivos.
- Generar un dashboard interactivo en **Power BI** para análisis visual.

---

## 📁 Estructura del Proyecto

```
Pokémon-TCG-Project/
├── data/
│   └── pokemon_tcg_full_data.csv
├── scripts/
│   ├── equivalencias_set.py
│   ├── walle.py
│   ├── trumpbot.py
│   └── firulai.py
├── outputs/
│   └── grafico_trumpbot.txt
└── dashboards/
    └── Pokemon_TCG_Dashboard.pbix
```

---

## ⚙️ Proceso de Desarrollo

### 1. Limpieza de Datos Inicial
- Lectura y filtrado del CSV con Pandas:

```python
df = pd.read_csv('pokemon_tcg_full_data.csv')
df.columns = df.columns.str.lower()
df = df[df['name'].notna() & df['set'].notna() & df['localid'].notna()]
invalid_sets = ["shining-revelry", "triumphant-light", "space-time-smackdown", "genetic-apex", "promo-a"]
df = df[~df['set'].isin(invalid_sets)]
```

### 2. Automatización con Bots (Selenium)
- **WALLE**: Scraping desde Pokestats.
- **TRUMPBOT**: Búsqueda avanzada en Cardmarket.
- **FIRULAI**: Construcción directa de URLs.

Ejemplo básico con Selenium:

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait

service = Service(executable_path='path_al_driver')
driver = webdriver.Chrome(service=service)
wait = WebDriverWait(driver, 10)
```

### 3. Extracción de Datos Históricos (Gráficas)

```python
def extraer_datos_grafica(driver):
    fechas = driver.find_elements(By.XPATH, '//script[contains(text(),"labels")]')[0].text.split('labels":[')[1].split('],"')[0].strip('"').split('","')
    precios = driver.find_elements(By.XPATH, '//script[contains(text(),"data")]')[0].text.split('data":[')[1].split(']')[0].split(',')
    return {"Fecha": fechas, "Precio": precios}
```

---

## 📊 Dashboard en Power BI

Incluye visualizaciones interactivas sobre:

- 📌 Top 10 cartas más caras
- 📌 Comparativa de sets (cantidad y valor promedio)
- 📌 Tendencias históricas de precios
- 📌 Distribución de rarezas y tipos
- 📌 Rankings interactivos de ilustradores

---

## 🧰 Tecnologías Utilizadas

- 🐍 **Python 3.10+**
- 🐼 **Pandas**
- 🚦 **Selenium WebDriver**
- 📈 **Power BI**
- ⚙️ **WebDriverWait**, **Regex**, **ast.literal_eval**

---

## 🚀 Próximas Mejoras

- 🤖 Entrenar modelos de IA para detectar errores frecuentes en datos.
- 🌐 Realizar scraping adicional desde fuentes como **Bulbapedia**.
- 📲 Desarrollar un dashboard web interactivo en tiempo real.

---

## 📜 Licencia

Este proyecto está disponible para uso educativo e investigación.

---

## 🤝 Cómo contribuir

¡Cualquier contribución es bienvenida! Por favor crea un `issue` o realiza un `pull request` con tus mejoras.

¡Gracias por visitar nuestro proyecto! 🌟🚀
