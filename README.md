# 🦝 Proyecto "Mapache" – Scraping y Análisis de Cartas Pokémon TCG 📊✨

Este proyecto se centra en la extracción automatizada, optimización y análisis visual interactivo de datos sobre cartas Pokémon TCG, utilizando un bot conocido como **Mapache**, desarrollado con Selenium y técnicas avanzadas de paralelización.

---

## 📌 Descripción General  

El proyecto automatiza la obtención de información desde CardMarket utilizando un archivo CSV como entrada inicial. El bot **Mapache** navega automáticamente, extrae datos gráficos y guarda resultados en formatos optimizados para análisis posteriores.

---

## 📁 Estructura del Proyecto

```
Proyecto-Mapache/
├── data/
│   └── pokemon_tcg_full_data.csv
├── scripts/
│   ├── equivalencias_set.py
│   └── mapache_bot.py
├── outputs/
│   ├── datos_graficos.csv
│   ├── mapache_log.txt
│   └── failed_cards.txt
└── dashboards/
    └── Pokemon_TCG_Dashboard.pbix
```

---

## ⚙️ Proceso Detallado

### 📂 Selección y Preparación del CSV
El CSV inicial debe contener los siguientes campos esenciales:

- **name** (Nombre de la carta)
- **set** (Información del set)
- **localid** (Número local de identificación)

**Filtrado y organización inicial:**
- Se separan cartas con sets válidos e inválidos.
- Se ordenan siguiendo el diccionario `cardmarket_set_map`.

```python
df = pd.read_csv('pokemon_tcg_full_data.csv')
df.columns = df.columns.str.lower()
df = df[df['name'].notna() & df['set'].notna() & df['localid'].notna()]
```

---

### 🤖 El Bot Mapache: Scraping y Automatización

**Navegación automática con Selenium:**
- Selección dinámica del set y carta.
- Manejo de pop-ups y captchas.

**Extracción de datos gráficos (JSON):**
```python
def extraer_datos_grafica(driver):
    fechas = driver.find_elements(By.XPATH, '//script[contains(text(),"labels")]')[0].text.split('labels":[')[1].split('],"')[0].strip('"').split('","')
    precios = driver.find_elements(By.XPATH, '//script[contains(text(),"data")]')[0].text.split('data":[')[1].split(']')[0].split(',')
    return {"Fecha": fechas, "Precio": precios}
```

**Gestión de errores y logs:**
- Resultados guardados en archivos CSV (`datos_graficos.csv`).
- Logs detallados (`mapache_log.txt` y `failed_cards.txt`).

---

### 🔄 Optimización y Paralelización

- Reducción inteligente de pausas.
- Procesamiento paralelo con múltiples instancias de Selenium.

```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np

chunks = np.array_split(lista_cartas, numero_nucleos)

with ProcessPoolExecutor(max_workers=numero_nucleos) as executor:
    executor.map(procesar_chunk, chunks)
```

---

## 📊 Dashboard en Power BI

Visualizaciones interactivas sobre:
- Cartas más caras.
- Distribución de rarezas.
- Análisis histórico de precios.
- Rendimiento de ilustradores.

---

## 📂 Archivos de Salida y Logs

- **`datos_graficos.csv`**: Datos de cartas y gráficos extraídos.
- **`mapache_log.txt`**: Log general del proceso con estadísticas.
- **`failed_cards.txt`**: Cartas que tuvieron errores durante el scraping.

---

## ⚙️ Requisitos e Instrucciones

### 🛠️ Requisitos

- **Python 3.x**
- Librerías:
  - `pandas`
  - `selenium`
  - `tqdm`
  - `concurrent.futures` (estándar)

- ChromeDriver compatible con versión instalada de Chrome.
- CSV inicial correctamente estructurado.
- Diccionario `cardmarket_set_map` en módulo `equivalencias_set.py`.

### ▶️ Instrucciones de Uso

1. Ajustar rutas en la sección de configuración del script.
2. Verificar instalación y compatibilidad del ChromeDriver.
3. Ejecutar `mapache_bot.py`.

---

## 🚧 Posibles Mejoras Futuras

- ✅ **Optimizar extracción de datos** mediante APIs (si están disponibles).
- ♻️ **Implementar reintentos automáticos** en casos de fallo inicial.
- ⏲️ **Adaptar tiempos de pausa dinámicamente** según rendimiento del sitio web.
- ☁️ **Desplegar en la nube** para mejorar escalabilidad y rendimiento.

---

## 🧰 Tecnologías Principales

- 🐍 **Python 3.10+**
- 🐼 **Pandas**
- 🚦 **Selenium**
- 📈 **Power BI**
- 🔄 **concurrent.futures**
- ⚙️ **WebDriverWait**, **Regex**, **ast.literal_eval**

---

## 📜 Licencia

Este proyecto está disponible para uso libre con fines educativos e investigación.

---

## 🤝 Cómo Contribuir

Cualquier contribución es bienvenida mediante:

- Creación de un **issue**.
- Realización de un **pull request**.

---

🎉 **¡Gracias por visitar y colaborar en el Proyecto Mapache!** 🚀🦝
