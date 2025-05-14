from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

# === CONFIGURACIÓN ===
chromedriver_path = r"C:\WebDriver\bin\chromedriver.exe"
card_id = "sv06.5-037"
url = f"https://pokestats.gg/tcg/card/{card_id}"

# === INICIAR WALLE ===
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)

print(f"🌐 WALLE está accediendo a {url}")
driver.get(url)
time.sleep(3)

import pandas as pd
import ast
import re
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from equivalencias_set import cardmarket_set_map

# === CONFIGURACIÓN ===
csv_path = r"C:\Users\vqv10\Downloads\Csv Cartas Completo\pokemon_tcg_full_data.csv"
chromedriver_path = r"C:\WebDriver\bin\chromedriver.exe"
output_log_path = r"C:\Users\vqv10\Downloads\mapache_log.txt"

invalid_sets = ["shining-revelry", "triumphant-light", "space-time-smackdown", "mythical-island", "genetic-apex",
                "promo-a"]


def limpiar_numero(localid):
    match = re.search(r'([A-Za-z]*)(\d+)([a-z]*)', localid)
    if not match:
        return [], ""
    prefix, number, suffix = match.groups()
    number_variants = [number.lstrip("0"), number.zfill(2), number.zfill(3)]
    full_number = prefix + number + suffix if prefix else number + suffix
    full_variants = list(dict.fromkeys([full_number] + number_variants))
    return full_variants, number.lstrip("0")


def abrir_link_como_humano(driver, url):
    print(f"🌐 MAPACHE abre: {url}")
    time.sleep(random.uniform(1.5, 2.5))
    driver.get(url)


def cerrar_popups(driver):
    try:
        cookies = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="CookiesConsent"]/div/div/form/div/button'))
        )
        cookies.click()
        print("🍪 Cookies cerradas")
    except Exception:
        pass
    try:
        popover = WebDriverWait(driver, 2).until(
            EC.element_to_be_clickable((By.XPATH, '//div[contains(@class, "popover-header")]//button'))
        )
        popover.click()
        print("🫧 Popover cerrado")
    except Exception:
        pass


def pausa(short=True):
    if short:
        time.sleep(random.uniform(1, 1.5))
    else:
        time.sleep(random.uniform(2, 3))


# === CARGAR CSV ===
df = pd.read_csv(csv_path)
df.columns = df.columns.str.lower()
df = df[df['name'].notna() & df['set'].notna() & df['localid'].notna()]
df = df.sample(10)

# === CONFIGURAR DRIVER ===
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)
wait = WebDriverWait(driver, 10)

# === MÉTRICAS Y RESULTADOS ===
aciertos = 0
fallos = 0
fallos_detalle = []
resultados = []  # Aquí se almacenarán los resultados: nombre de carta y link o 'fallo'

for index, row in df.iterrows():
    try:
        name = row['name']
        localid = str(row['localid'])
        try:
            set_data = ast.literal_eval(row['set'])
            set_id = set_data.get('id', '')
        except Exception:
            continue

        if set_id.lower() in invalid_sets:
            print(f"⚠️ Set inválido: {set_id}")
            resultados.append({'name': name, 'link': 'fallo'})
            continue

        set_info = cardmarket_set_map.get(set_id)
        if not set_info:
            print(f"⚠️ Set no encontrado en el diccionario: {set_id}")
            resultados.append({'name': name, 'link': 'fallo'})
            continue

        expansion = set_info['name']
        numeros_a_probar, _ = limpiar_numero(localid)
        print(f"\n🦝 MAPACHE va a buscar la carta '{name}' del set '{expansion}' con número {localid}")

        # Abrir página de búsqueda y cerrar popups
        abrir_link_como_humano(driver, "https://www.cardmarket.com/en/Pokemon/Products/Search")
        cerrar_popups(driver)

        # Seleccionar la expansión
        try:
            select = wait.until(EC.presence_of_element_located((By.NAME, 'idExpansion')))
            opciones = select.find_elements(By.TAG_NAME, 'option')
            seleccionado = False
            for option in opciones:
                if option.text.strip().lower() == expansion.lower():
                    option.click()
                    seleccionado = True
                    print(f"✅ Set seleccionado: {option.text}")
                    break
            if not seleccionado:
                print(f"❌ No se pudo seleccionar el set: {expansion}")
                resultados.append({'name': name, 'link': 'fallo'})
                continue
        except Exception as e:
            print(f"❌ Error al seleccionar set: {e}")
            resultados.append({'name': name, 'link': 'fallo'})
            continue

        # Lanzar búsqueda
        try:
            search_button = driver.find_element(By.XPATH, '//input[@type="submit" and @value="Search"]')
            search_button.click()
            print("🔍 Lanzando búsqueda...")
        except Exception as e:
            print("❌ Botón de búsqueda no encontrado")
            resultados.append({'name': name, 'link': 'fallo'})
            continue

        # Bucle de paginación
        found_card = False
        card_url = ""
        pagina = 1
        while True:
            print(f"🔄 Revisando página {pagina}...")
            pausa(short=True)
            cerrar_popups(driver)
            try:
                filas = wait.until(
                    EC.presence_of_all_elements_located((By.XPATH, '//div[contains(@id, "productRow")]'))
                )
            except TimeoutException:
                filas = driver.find_elements(By.XPATH, '//div[contains(@id, "productRow")]')
            print(f"📦 Filas encontradas: {len(filas)} en la página {pagina}")

            for i, fila in enumerate(filas, start=1):
                try:
                    num_elem = fila.find_element(By.XPATH, './/div[contains(@class, "col-number")]//span[last()]')
                    numero_fila = num_elem.text.strip().lstrip("#").lstrip("0")
                    if not numero_fila:
                        print(f"  -> Fila {i}: no se encontró número, saltando")
                        continue
                    print(f"  -> Fila {i}: número encontrado '{numero_fila}'")
                    if numero_fila in numeros_a_probar:
                        print(f"✅ Encontrado número {numero_fila} en la fila {i}. Centrando y haciendo clic...")
                        link = fila.find_element(By.XPATH, './/div[contains(@class, "col-md-8")]//a')
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", link)
                        pausa(short=True)
                        try:
                            link.click()
                        except ElementClickInterceptedException as ex:
                            print(f"  -> Error al hacer clic en fila {i}: {ex}. Intentando click por JS...")
                            driver.execute_script("arguments[0].click();", link)
                        aciertos += 1
                        found_card = True
                        pausa(short=True)
                        # Capturamos la URL de la carta antes de regresar
                        try:
                            wait.until(EC.url_changes(driver.current_url))
                        except Exception as ex:
                            print("❌ No se detectó cambio de URL:", ex)
                        card_url = driver.current_url
                        print(f"↩️ Carta encontrada: {card_url}. Deteniendo búsqueda en esta carta.")
                        break
                except Exception as e:
                    try:
                        fila_html = fila.get_attribute("outerHTML")
                    except Exception:
                        fila_html = "No se pudo obtener el HTML"
                    print(f"  -> Error en fila {i}: {e}. HTML (primeros 200 caracteres): {fila_html[:200]}...")
                    continue
            if found_card:
                break
            else:
                try:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    pausa(short=True)
                    next_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//a[@aria-label="Next page"]')))
                    print("➡️ Haciendo clic en Next page...")
                    next_button.click()
                    pagina += 1
                except Exception as e:
                    print("❌ No hay más páginas o error al cambiar de página:", e)
                    fallos += 1
                    resultados.append({'name': name, 'link': 'fallo'})
                    break

        if found_card:
            pausa(short=True)
            driver.back()
            pausa(short=True)
            resultados.append({'name': name, 'link': card_url})
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        fallos += 1
        resultados.append({'name': name, 'link': 'fallo'})
        continue

pausa(short=False)
total = aciertos + fallos
print("\n=== RESULTADOS ===")
print(f"✅ Aciertos: {aciertos}")
print(f"❌ Fallos: {fallos}")
print(f"🎯 Porcentaje de éxito: {round(aciertos / total * 100, 2)}%")
print("📝 Fallos:")
for f in fallos_detalle:
    print(f" - {f}")

# Mostrar al final el resumen de cartas y sus links
print("\n=== RESULTADO DE BÚSQUEDA DE CARTAS ===")
for res in resultados:
    print(f"Carta: {res['name']}, Link: {res['link']}")

with open(output_log_path, "w", encoding="utf-8") as f:
    f.write("==== RESUMEN MAPACHE ====\n")
    f.write(f"Aciertos: {aciertos}\n")
    f.write(f"Fallos: {fallos}\n")
    f.write(f"Porcentaje: {round(aciertos / total * 100, 2)}%\n")
    f.write("\n=== RESULTADO DE BÚSQUEDA DE CARTAS ===\n")
    for res in resultados:
        f.write(f"Carta: {res['name']}, Link: {res['link']}\n")
    f.write("\nFallos:\n")
    for fail in fallos_detalle:
        f.write(f"{fail}\n")

driver.quit()

# === EXTRAER DATOS ===
print("\n📊 WALLE está extrayendo los precios con precisión...")
data = {}

# 1. Extraer secciones de precios ($ para TCGplayer, € para Cardmarket)
blocks = driver.find_elements(By.CLASS_NAME, "MuiGrid2-direction-xs-row")

for block in blocks:
    try:
        label = block.find_element(By.TAG_NAME, "p").text.strip()
        value = block.find_element(By.TAG_NAME, "span").text.strip()

        if "$" in value:
            data[f"TCG_{label}"] = value
        elif "€" in value:
            data[f"CM_{label}"] = value
    except:
        continue

# === MOSTRAR DATOS ===
print(f"\n🧾 Precios de la carta {card_id}:")
for label, value in data.items():
    print(f"{label}: {value}")

# === TERMINAR ===
print("\n👋 WALLE ha terminado por ahora.")
time.sleep(5)
driver.quit()
