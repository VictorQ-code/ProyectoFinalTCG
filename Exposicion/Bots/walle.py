# Código actualizado con lógica para:
# 1. Detectar modificadores como "Lv.30", "ex", "V" y añadirlos al nombre.
# 2. Manejar autocompletado y fallback con botón "Mostrar todos".
# 3. Evitar sets digitales.
# 4. Eliminar decimales en set_id como "sv8.5" → "sv8".

import pandas as pd
import random
import ast
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
from selenium.common.exceptions import TimeoutException
from equivalencias_set import cardmarket_set_map

# === CONFIGURACIÓN ===
csv_path = r"C:\Users\vqv10\Downloads\Csv Cartas Completo\pokemon_tcg_full_data.csv"
chromedriver_path = r"C:\WebDriver\bin\chromedriver.exe"
output_txt_path = r"C:\Users\vqv10\Downloads\grafico_trumpbot.txt"

# === CARGAR CSV ===
df = pd.read_csv(csv_path)
df.columns = df.columns.str.lower()
df = df[df['name'].notna() & df['set'].notna() & df['localid'].notna()]

# === ELEGIR CARTA ===
row = df.sample(1).iloc[0]
name = row['name']
local_id = str(row['localid'])

# === DETECTAR Y NORMALIZAR ID DE SET ===
try:
    set_data = ast.literal_eval(row['set'])
    set_id = set_data.get('id', '')
except:
    set_id = ""

# Eliminar decimales tipo "sv8.5" → "sv8"
if "." in set_id:
    set_id = set_id.split(".")[0]

# Lista de sets que NO existen físicamente
invalid_sets = [
    "shining-revelry", "triumphant-light", "space-time-smackdown",
    "mythical-island", "genetic-apex", "promo-a"
]
if set_id.lower() in invalid_sets:
    print(f"⛔ Set inválido (solo digital): {set_id} — se omite esta carta.")
    exit()

# === EQUIVALENCIA CON DICCIONARIO ===
set_info = cardmarket_set_map.get(set_id, {"abbreviation": set_id})
abbreviation = set_info.get("abbreviation", set_id)
print(f"🧩 Set original: {set_id} → Cardmarket: {abbreviation}")

# === DETECTAR MODIFICADORES EN EL NOMBRE ===
modificadores = ['Lv.', 'nvl', 'Mega', 'Dark', 'Shiny']
full_name = name
for mod in modificadores:
    if mod.lower() in name.lower():
        full_name = name
        break
    elif f"{name} {mod}" in df['name'].values:
        full_name = f"{name} {mod}"

# === CONSTRUIR BÚSQUEDA ===
search_term = f"{full_name} ({abbreviation} {int(local_id)})"
print(f"🕵️ TRUMPBOT ha elegido: {search_term}")

# === INICIAR CHROME ===
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)
driver.get("https://www.cardmarket.com/en/Pokemon")

try:
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ProductSearchInput"]')))

    # === COOKIES ===
    try:
        cookies_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="CookiesConsent"]/div/div/form/div/button'))
        )
        cookies_btn.click()
        print("🍪 TRUMPBOT ha cerrado el banner de cookies.")
    except:
        print("✅ No hay cookies que cerrar.")

    # === POPOVER ===
    try:
        popover_close = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, '//div[contains(@class, "popover-header")]//button[contains(@class, "btn-link")]'))
        )
        popover_close.click()
        print("🧼 TRUMPBOT ha cerrado el popover.")
    except:
        print("✅ No hay popover visible.")

    # === BUSCAR CARTA ===
    print(f"🔍 TRUMPBOT va a buscar: {search_term}")
    search_input = driver.find_element(By.XPATH, '//*[@id="ProductSearchInput"]')
    search_input.clear()
    search_input.send_keys(search_term)

    objetivo = f"({abbreviation} {int(local_id)})"
    encontrado = False

    try:
        WebDriverWait(driver, 6).until(
            lambda d: any(objetivo in r.text for r in d.find_elements(By.XPATH, '//*[@id="AutoCompleteResult"]/a'))
        )
        resultados = driver.find_elements(By.XPATH, '//*[@id="AutoCompleteResult"]/a')
        for r in resultados:
            if objetivo in r.text:
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", r)
                    WebDriverWait(driver, 3).until(EC.element_to_be_clickable(r))
                    r.click()
                    print(f"✅ Trumpbot hizo clic en el resultado correcto: {objetivo}")
                    encontrado = True
                    break
                except:
                    ActionChains(driver).move_to_element(r).click().perform()
                    print(f"✅ Trumpbot usó ActionChains para hacer clic: {objetivo}")
                    encontrado = True
                    break
    except TimeoutException:
        print(f"🔎 No se encontró en autocompletado. Intentando 'Mostrar todos'...")

    # === FALLBACK: MOSTRAR TODOS ===
    if not encontrado:
        try:
            mostrar_todos_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, '//span[contains(text(),"Mostrar todos")]/ancestor::a'))
            )
            mostrar_todos_btn.click()
            print("📂 Trumpbot hizo clic en 'Mostrar todos'.")

            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".table-fixed tbody tr"))
            )
            resultados_tabla = driver.find_elements(By.CSS_SELECTOR, ".table-fixed tbody tr")
            for fila in resultados_tabla:
                if objetivo in fila.text:
                    link = fila.find_element(By.CSS_SELECTOR, "a")
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
                    WebDriverWait(driver, 3).until(EC.element_to_be_clickable(link))
                    link.click()
                    print(f"✅ Trumpbot accedió desde 'Mostrar todos': {objetivo}")
                    encontrado = True
                    break

            if not encontrado:
                print(f"❌ Ni siquiera desde 'Mostrar todos' se encontró: {objetivo}")
        except Exception as e:
            print(f"❌ Falló el intento con 'Mostrar todos': {e}")

finally:
    print("👋 TRUMPBOT ha terminado su misión por ahora.")
    driver.quit()


# Puedes activar esto cuando la carta esté correctamente cargada
"""
print("📈 TRUMPBOT intenta leer los datos del gráfico...")
try:
    chart_data = driver.execute_script(\"""
        const chart = Object.values(Chart.instances)[0];
        return {
            labels: chart.data.labels,
            prices: chart.data.datasets[0].data
        };
    \""")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for date, price in zip(chart_data["labels"], chart_data["prices"]):
            f.write(f"{date} - {price} €\n")
    print(f"✅ Datos del gráfico guardados en: {output_txt_path}")
except Exception as e:
    print(f"❌ No se pudo leer el gráfico: {e}")
"""

print("👋 TRUMPBOT ha terminado su misión por ahora.")
driver.quit()
