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
from selenium.common.exceptions import TimeoutException
from equivalencias_set import cardmarket_set_map

# === CONFIGURACIÓN ===
csv_path = r"C:\Users\vqv10\Downloads\Csv Cartas Completo\pokemon_tcg_full_data.csv"
chromedriver_path = r"C:\WebDriver\bin\chromedriver.exe"
output_log_path = r"C:\Users\vqv10\Downloads\firulai_log.txt"

# === AJUSTES DE PAUSAS SIMULANDO HUMANO ===
def pausa_entre_cartas():
    time.sleep(random.uniform(5, 10))

def abrir_link_como_humano(driver, url):
    print(f"🌐 FIRULAI abre: {url}")
    time.sleep(random.uniform(3, 6))
    driver.get(url)

# === LIMPIEZA DE NOMBRE ===
def normalize_name(name):
    replacements = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "á": "a", "à": "a", "ä": "a", "â": "a",
        "í": "i", "ï": "i", "î": "i",
        "ó": "o", "ö": "o", "ô": "o",
        "ú": "u", "ü": "u", "û": "u",
        "ñ": "n", "ç": "c", "’": "", "'": "", ".": "", ":": ""
    }
    for char, repl in replacements.items():
        name = name.replace(char, repl)
    name = name.replace("Lv.", "Lv").replace("GX", "-GX").replace("EX", "-EX").replace("V", "-V")
    name = name.replace(" ", "-").replace("--", "-").strip("-")
    return name

# === CARGAR CSV Y FILTRAR ===
df = pd.read_csv(csv_path)
df.columns = df.columns.str.lower()
df = df[df['name'].notna() & df['set'].notna() & df['localid'].notna()]

invalid_sets = ["shining-revelry", "triumphant-light", "space-time-smackdown", "mythical-island", "genetic-apex", "promo-a"]
def es_valido(set_str):
    try:
        s = ast.literal_eval(set_str)
        return s.get("id", "").lower() not in invalid_sets
    except:
        return False

df = df[df['set'].apply(es_valido)]
df = df.sample(100)

# === INICIAR DRIVER ===
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)

# === VARIABLES LOG Y MÉTRICAS ===
aciertos = 0
fallos = 0
fallos_por_set = {}
log = []

# === BUCLE PRINCIPAL ===
for index, row in df.iterrows():
    name = row['name']
    local_id = str(row['localid'])

    try:
        set_data = ast.literal_eval(row['set'])
        set_id = set_data.get('id', '')
    except:
        continue

    if set_id.lower() in invalid_sets:
        continue

    set_info = cardmarket_set_map.get(set_id, {"abbreviation": set_id, "name": set_id})
    abbreviation = set_info.get("abbreviation", set_id)
    set_name = set_info.get("name", set_id).replace(" ", "-")
    local_id_digits = re.search(r"\d+", local_id)
    local_id_int = int(local_id_digits.group()) if local_id_digits else 0
    final_id = f"{abbreviation}{str(local_id_int).zfill(3)}"

    name_normalized = normalize_name(name)
    objetivo = f"{name} ({abbreviation} {local_id_int})"

    urls_a_probar = [
        f"https://www.cardmarket.com/es/Pokemon/Products/Singles/{set_name}/{name_normalized}",
        f"https://www.cardmarket.com/es/Pokemon/Products/Singles/{set_name}/{name_normalized}-{final_id}",
        f"https://www.cardmarket.com/es/Pokemon/Products/Singles/{set_name}/{name_normalized}-V1-{final_id}",
        f"https://www.cardmarket.com/es/Pokemon/Products/Singles/{set_name}/{name_normalized}-V2-{final_id}",
        f"https://www.cardmarket.com/es/Pokemon/Products/Singles/{set_name}/{name_normalized}-V3-{final_id}",

        f"https://www.cardmarket.com/es/Pokemon/Products/Singles/{set_name}/{name_normalized}-{abbreviation}{local_id_int}",
        f"https://www.cardmarket.com/es/Pokemon/Products/Singles/{set_name}/{name_normalized}-V1-{abbreviation}{local_id_int}",
        f"https://www.cardmarket.com/es/Pokemon/Products/Singles/{set_name}/{name_normalized}-V2-{abbreviation}{local_id_int}",
        f"https://www.cardmarket.com/es/Pokemon/Products/Singles/{set_name}/{name_normalized}-V3-{abbreviation}{local_id_int}",        f"https://www.cardmarket.com/es/Pokemon/Products/Singles/{set_name}/{name_normalized}-V3-{abbreviation}{local_id_int}",

    ]

    encontrado = False
    for url in urls_a_probar:
        abrir_link_como_humano(driver, url)

        try:
            # Cerrar cookies
            try:
                cookies_btn = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="CookiesConsent"]/div/div/form/div/button'))
                )
                cookies_btn.click()
            except: pass

            # Cerrar popover
            try:
                popover_close = WebDriverWait(driver, 2).until(
                    EC.element_to_be_clickable((By.XPATH, '//div[contains(@class, "popover-header")]//button[contains(@class, "btn-link")]'))
                )
                popover_close.click()
            except: pass

            # Verificar título
            WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.XPATH, '//h1')))
            titulo = driver.find_element(By.XPATH, '//h1').text.strip()

            if str(local_id_int) in titulo and abbreviation in titulo:
                print(f"✅ ENCONTRADO: {titulo}")
                aciertos += 1
                encontrado = True
                break
        except:
            continue

    if not encontrado:
        fallos += 1
        fallos_por_set.setdefault(set_id, 0)
        fallos_por_set[set_id] += 1
        log.append(f"❌ FALLÓ: {objetivo} — Último intento: {url}")

    print(f"📊 Progreso: {aciertos+fallos}/100 | ✅ {aciertos} | ❌ {fallos} | Éxito: {round((aciertos/(aciertos+fallos))*100, 2)}%")
    pausa_entre_cartas()

# === LOG FINAL ===
with open(output_log_path, "w", encoding="utf-8") as f:
    f.write("==== RESUMEN DE FIRULAI ====\n")
    f.write(f"Total cartas: {aciertos+fallos}\n")
    f.write(f"Aciertos: {aciertos}\n")
    f.write(f"Fallos: {fallos}\n")
    f.write(f"Porcentaje de éxito: {round((aciertos/(aciertos+fallos))*100, 2)}%\n\n")
    f.write("FALLOS POR SET:\n")
    for s, n in fallos_por_set.items():
        f.write(f" - {s}: {n} fallos\n")
    f.write("\n=== DETALLES DE FALLOS ===\n")
    for line in log:
        f.write(line + "\n")

driver.quit()

