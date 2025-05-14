import pandas as pd
import ast
import re
import time
import random
import string
import json
import csv
import os
import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from equivalencias_set import cardmarket_set_map

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
csv_path = r"C:\Users\vqv10\Downloads\Csv Cartas Completo\pokemon_tcg_full_data.csv"
chromedriver_path = r"C:\WebDriver\bin\chromedriver.exe"
output_log_path = r"C:\Users\vqv10\Downloads\mapache_log.txt"
# Ruta para guardar los datos de gráficos de todas las cartas en un único CSV
chart_csv_path = r"C:\Users\vqv10\Downloads\Csv Cartas Completo\charts_data.csv"

invalid_sets = ["shining-revelry", "triumphant-light", "space-time-smackdown",
                "mythical-island", "genetic-apex", "promo-a"]
captcha_xpath = '//div[contains(@class, "captcha-container")]//button'


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def normalize_text(text):
    return "".join(ch for ch in text if ch not in string.punctuation).lower().strip()


def limpiar_numero(localid):
    # Permite un guion o espacio opcional entre el prefijo y los dígitos.
    match = re.search(r'([A-Za-z]*)(?:[-\s])?(\d+)([a-z]*)', localid)
    if not match:
        return [], ""
    prefix, number, suffix = match.groups()
    stripped = number.lstrip("0")
    numeric_variants = [stripped, number.zfill(2), number.zfill(3)]
    if prefix:
        variant_with_prefix   = prefix + stripped
        variant_with_prefix2  = prefix + number.zfill(2)
        variant_with_prefix3  = prefix + number.zfill(3)
        full_variants = [prefix + number + suffix, variant_with_prefix, variant_with_prefix2, variant_with_prefix3] + numeric_variants
    else:
        full_variants = [number] + numeric_variants
    seen = set()
    result = []
    for v in full_variants:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result, stripped



def pausa(short=True):
    time.sleep(random.uniform(1, 1.5) if short else random.uniform(2, 3))


def cerrar_popups(driver):
    for xpath in ['//*[@id="CookiesConsent"]//button', '//div[contains(@class, "popover-header")]//button']:
        try:
            WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.XPATH, xpath))).click()
        except Exception:
            continue


def check_and_close_captcha(driver):
    try:
        WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.XPATH, captcha_xpath))).click()
        print("✅ Captcha resuelto.")
    except Exception:
        pass


def extraer_datos_grafica(driver):
    try:
        print("Intentando localizar el script de la gráfica...")
        script_element = driver.find_element(By.XPATH,
                                             '/html/body/main/div[3]/section[2]/div/div[2]/div[1]/div/div[3]/div[1]/script')
        script_text = script_element.get_attribute('innerHTML')
        print("Longitud del script:", len(script_text))
        if not script_text:
            print("El script está vacío.")
            return None
        start = script_text.find('{')
        if start == -1:
            print("No se encontró '{' en el script.")
            return None
        count = 0
        end = start
        for i, ch in enumerate(script_text[start:], start=start):
            if ch == '{':
                count += 1
            elif ch == '}':
                count -= 1
                if count == 0:
                    end = i
                    break
        json_text = script_text[start:end + 1].strip()
        print("JSON extraído (balanceado) (primeros 200 caracteres):", json_text[:200])
        json_text = re.sub(r'function\s*\([^)]*\)\s*\{[^}]*\}', 'null', json_text)
        print("JSON tras eliminar funciones (primeros 200 caracteres):", json_text[:200])
        try:
            chart_config = json.loads(json_text)
            print("JSON parseado correctamente.")
        except Exception as e:
            print("Error al parsear el JSON:", e)
            return None
        labels = chart_config["data"].get("labels", [])
        data = chart_config["data"].get("datasets", [])[0].get("data", [])
        print("Labels extraídas:", labels)
        print("Datos extraídos:", data)
        return [labels, data]
    except Exception as e:
        print("❌ Error al extraer datos de la gráfica:", e)
        return None


def guardar_chart_a_csv(card_name, card_set, card_number, set_id, chart_data, output_csv):
    file_exists = os.path.exists(output_csv)
    chart_dict = {"labels": chart_data[0], "data": chart_data[1]}
    chart_json = json.dumps(chart_dict, ensure_ascii=False)
    with open(output_csv, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Carta", "Set", "Número", "Set_ID", "ChartData"])
        writer.writerow([card_name, card_set, card_number, set_id, chart_json])
    print(f"Datos de la gráfica de '{card_name}' guardados en: {output_csv}")


# =============================================================================
# OBTENER DATOS DE LA CARTA DESDE EL CSV (número y set)
# =============================================================================
# Aquí usamos directamente los datos del CSV
# 'card_number_csv' se obtiene de limpiar_numero y 'expansion' del mapeo.
# =============================================================================

# =============================================================================
# CARGA Y PREPARACIÓN DEL CSV
# =============================================================================
df = pd.read_csv(csv_path)
df.columns = df.columns.str.lower()
df = df[df['name'].notna() & df['set'].notna() & df['localid'].notna()]


# =============================================================================
# CONFIGURACIÓN DEL DRIVER
# =============================================================================
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)

wait = WebDriverWait(driver, 4)

# =============================================================================
# SELECCIÓN CARTAS
# =============================================================================
# Función auxiliar para extraer el id del set desde el string
def get_set_id(set_str):
    try:
        set_data = ast.literal_eval(set_str)
        return set_data.get('id', '')
    except Exception:
        return ''

# Agregar una nueva columna 'set_id' extraída de la columna 'set'
df['set_id'] = df['set'].apply(get_set_id)

# Obtener la lista de set_ids en el orden definido en el diccionario
ordered_sets = list(cardmarket_set_map.keys())

# Filtrar solo las cartas cuyos set_id estén en cardmarket_set_map
df = df[df['set_id'].isin(ordered_sets)]

# Asignar a cada carta un índice basado en la posición del set_id en el diccionario
df['set_order'] = df['set_id'].apply(lambda x: ordered_sets.index(x))

# Función auxiliar para extraer el número de carta (como entero) a partir de 'localid'
def extract_card_number(localid):
    variantes, card_number_csv = limpiar_numero(localid)
    try:
        return int(card_number_csv)
    except Exception:
        return 0

# Agregar la columna 'card_number'
df['card_number'] = df['localid'].apply(extract_card_number)

# Ordenar el DataFrame por el orden del set y luego por el número de carta
df = df.sort_values(['set_order', 'card_number'])


# =============================================================================
# PROCESO REPETIDO 5 VECES
# =============================================================================
num_iteraciones = 1
for iteracion in range(num_iteraciones):
    print(f"\n===== EJECUCIÓN {iteracion + 1} =====")
    aciertos = 0
    fallos = 0
    resultados = []

    for index, row in df.iterrows():
        try:
            name = row['name']
            localid = str(row['localid'])
            # Usamos directamente el localid para obtener variantes y el número (card_number_csv)
            variantes, card_number_csv = limpiar_numero(localid)
            print(f"\n🦝 Buscando carta '{name}' ({localid}) en el set del CSV")
            print(f"   Variantes a buscar: {variantes}")

            try:
                set_data = ast.literal_eval(row['set'])
                set_id = set_data.get('id', '')
            except Exception:
                continue

            if set_id.lower() in invalid_sets:
                print(f"⚠️ Set prohibido: {set_id}")
                resultados.append({'name': name, 'link': 'error: set incorrecto'})
                continue

            set_info = cardmarket_set_map.get(set_id)
            if not set_info:
                print(f"⚠️ Set no encontrado: {set_id}")
                resultados.append({'name': name, 'link': 'fallo'})
                continue

            expansion = set_info['name']
            print(f"   Set: {expansion}, Set_ID: {set_id}, Número (CSV): {card_number_csv}")

            # Ir a la página de búsqueda
            driver.get("https://www.cardmarket.com/en/Pokemon/Products/Search")
            cerrar_popups(driver)
            check_and_close_captcha(driver)

            # Seleccionar la expansión
            try:
                select = wait.until(EC.presence_of_element_located((By.NAME, 'idExpansion')))
                opciones = select.find_elements(By.TAG_NAME, 'option')
                seleccionado = False
                for option in opciones:
                    if option.text.strip() == expansion:
                        option.click()
                        seleccionado = True
                        print(f"✅ Set seleccionado: {option.text}")
                        break
                if not seleccionado:
                    print(f"❌ No se pudo seleccionar el set: {expansion}")
                    resultados.append({'name': name, 'link': 'fallo: set incorrecto'})
                    continue
            except Exception as e:
                print(f"❌ Error al seleccionar set: {e}")
                resultados.append({'name': name, 'link': 'fallo: set incorrecto'})
                continue

            # Rellenar el campo "Name"
            try:
                name_input = wait.until(EC.presence_of_element_located((
                    By.XPATH, '/html/body/main/section/div[1]/form/div/div[3]/input'
                )))
                name_input.clear()
                name_input.send_keys(name)
                print(f"✅ Se ha introducido en el campo 'Name': {name}")
            except Exception as e:
                print(f"❌ Error al introducir el nombre: {e}")
                resultados.append({'name': name, 'link': 'fallo: no se pudo introducir nombre'})
                continue

            # Hacer clic en "Search"
            try:
                search_button = driver.find_element(By.XPATH, '//input[@type="submit" and @value="Search"]')
                search_button.click()
                print("🔍 Buscando...")
                check_and_close_captcha(driver)
            except Exception as e:
                print("❌ Botón de búsqueda no encontrado")
                resultados.append({'name': name, 'link': 'fallo'})
                continue

            pausa(short=True)
            cerrar_popups(driver)
            check_and_close_captcha(driver)

            # Caso 1: Resultado directo (URL ya no contiene "Products/Search")
            if "Products/Search" not in driver.current_url:
                print("🟢 Resultado directo: solo hay una carta para este Pokémon en el set.")
                found_card = True
                card_url = driver.current_url
                pausa(short=True)  # Mantener la carta abierta 1 segundo
            else:
                # Caso 2: Lista de resultados; filtrar por número (#)
                found_card = False
                card_url = ""
                pagina = 1
                while True:
                    print(f"🔄 Página {pagina}...")
                    pausa(short=True)
                    cerrar_popups(driver)
                    check_and_close_captcha(driver)
                    try:
                        filas = wait.until(
                            EC.presence_of_all_elements_located((By.XPATH, '//div[contains(@id, "productRow")]')))
                    except TimeoutException:
                        filas = driver.find_elements(By.XPATH, '//div[contains(@id, "productRow")]')
                    for fila in filas:
                        try:
                            raw_text = fila.find_element(By.XPATH,
                                                         './/div[contains(@class, "col-number")]//span[last()]').text.strip()
                            if not raw_text:
                                continue
                            numero_fila = raw_text.lstrip("#").lstrip("0")
                            if numero_fila in variantes or raw_text in variantes:
                                # Hacer clic en la carta candidata
                                link = fila.find_element(By.XPATH, './/div[contains(@class, "col-md-8")]//a')
                                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
                                pausa(short=True)
                                try:
                                    link.click()
                                except ElementClickInterceptedException:
                                    driver.execute_script("arguments[0].click();", link)
                                pausa(short=True)
                                print(
                                    f"Verificación: Se usará el número '{card_number_csv}' y la expansión '{expansion}' (del CSV)")
                                found_card = True
                                card_url = driver.current_url
                                print(f"↩️ Carta encontrada: {card_url}")
                                break
                        except Exception:
                            continue
                    if found_card:
                        break
                    else:
                        try:
                            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            pausa(short=True)
                            next_button = wait.until(
                                EC.element_to_be_clickable((By.XPATH, '//a[@aria-label="Next page"]')))
                            next_button.click()
                            pagina += 1
                        except Exception as e:
                            print(f"❌ Error cambiando página: {e}")
                            fallos += 1
                            resultados.append({'name': name, 'link': 'fallo'})
                            break

            if found_card:
                pausa(short=True)
                # Extraer datos del gráfico
                chart_data = extraer_datos_grafica(driver)
                if chart_data:
                    guardar_chart_a_csv(name, expansion, card_number_csv, set_id, chart_data, chart_csv_path)
                else:
                    print("No se extrajeron datos de la gráfica.")
                pausa(short=True)
                driver.back()
                pausa(short=True)
                resultados.append({'name': name, 'link': card_url})
                aciertos += 1
            else:
                resultados.append({'name': name, 'link': 'fallo'})
                fallos += 1

        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            fallos += 1
            resultados.append({'name': name, 'link': 'fallo'})
            continue

    # Fin de la iteración: mostrar resumen y guardar log
    pausa(short=False)
    total = aciertos + fallos if (aciertos + fallos) > 0 else 1
    print("\n=== RESULTADOS ===")
    print(f"Aciertos: {aciertos} - Fallos: {fallos} - Éxito: {round(aciertos / total * 100, 2)}%")
    print("\n=== RESULTADO DE CARTAS ===")
    for res in resultados:
        print(f"Carta: {res['name']} -> {res['link']}")

    with open(output_log_path, "a", encoding="utf-8") as f:
        f.write("==== RESUMEN MAPACHE ====\n")
        f.write(f"Ejecución {iteracion + 1}\n")
        f.write(f"Aciertos: {aciertos}\n")
        f.write(f"Fallos: {fallos}\n")
        f.write(f"Porcentaje: {round(aciertos / total * 100, 2)}%\n")
        f.write("=== RESULTADO DE CARTAS ===\n")
        for res in resultados:
            f.write(f"Carta: {res['name']} -> {res['link']}\n")
        f.write("\n")

# Fin del proceso
driver.quit()
sys.exit(0)
