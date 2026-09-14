import os
import random
import torch
import torch.nn.functional as F
import flet as ft
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ID_HF = "mick2332-q/semantle-es-vectors"
NOMBRE_ARCHIVO_CACHE = "vectores_es_fp16.pt"
RUTA_CACHE_BINARIO = os.path.join(BASE_DIR, NOMBRE_ARCHIVO_CACHE)

def obtener_cache():
    if not os.path.exists(RUTA_CACHE_BINARIO):
        hf_hub_download(
            repo_id=REPO_ID_HF, 
            filename=NOMBRE_ARCHIVO_CACHE,
            local_dir=BASE_DIR
        )
    datos = torch.load(RUTA_CACHE_BINARIO, map_location=device)
    return datos["palabras"], datos["tensor_vectores"], datos["dict_vocab"]

palabras, tensor_vectores, dict_vocab = obtener_cache()
TOTAL_PALABRAS = len(palabras)

PALABRA_SECRETA = "guitarra"
vector_secreto = None
ranking_dict = {}
orden_palabras_ranking = []

def actualizar_palabra_secreta(nueva_palabra=None):
    global PALABRA_SECRETA, vector_secreto, ranking_dict, orden_palabras_ranking
    if nueva_palabra and nueva_palabra in dict_vocab:
        PALABRA_SECRETA = nueva_palabra
    else:
        PALABRA_SECRETA = random.choice(palabras[:50000])

    idx_secreta = dict_vocab[PALABRA_SECRETA]
    vector_secreto = tensor_vectores[idx_secreta]

    similitudes_todas = torch.mv(tensor_vectores, vector_secreto)
    orden_indices = torch.argsort(similitudes_todas, descending=True).cpu().tolist()

    orden_palabras_ranking = [palabras[idx] for idx in orden_indices]
    ranking_dict = {palabras[idx]: pos + 1 for pos, idx in enumerate(orden_indices)}

actualizar_palabra_secreta(PALABRA_SECRETA)

def obtener_color_temperatura(puesto):
    if puesto == 1:
        return "greenAccent400"
    elif puesto <= 100:
        return "green400"
    elif puesto <= 1000:
        return "lime400"
    elif puesto <= 5000:
        return "amber400"
    elif puesto <= 20000:
        return "orange400"
    else:
        return "red400"

def main(page: ft.Page):
    page.title = f"Semantle Español ({device.type.upper()} FP16) 🎯"
    page.window_width = 540
    page.window_height = 850
    page.padding = 24
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(color_scheme_seed="cyan")

    intentos = []
    contador = 0
    pistas_usadas = 0

    header = ft.Column([
        ft.Text("Semantle Español 🎯", size=28, weight=ft.FontWeight.BOLD, color="cyan200"),
        ft.Text(f"Motor: PyTorch FP16 Tensor Matrix en {device.type.upper()} | Vocabulario: {TOTAL_PALABRAS:,}", size=11, color="grey500"),
    ], spacing=2)

    input_palabra = ft.TextField(
        hint_text="Escribe una palabra...",
        expand=True,
        autofocus=True,
        border_radius=10,
        on_submit=lambda e: evaluar_palabra(e)
    )

    mensaje_estado = ft.Text("", size=13, weight=ft.FontWeight.W_500)

    resultado_destacado = ft.Container(
        content=ft.Column([
            ft.Text("Último intento", size=12, color="grey400"),
            ft.Text("Esperando primera palabra...", size=18, weight=ft.FontWeight.BOLD),
            ft.ProgressBar(value=0, color="cyan", bgcolor="grey800", height=8),
            ft.Text("", size=12)
        ], spacing=6),
        padding=16,
        border_radius=12,
        bgcolor="surfaceVariant",
        visible=False
    )

    tabla_intentos = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("#")),
            ft.DataColumn(ft.Text("Palabra")),
            ft.DataColumn(ft.Text("Similitud")),
            ft.DataColumn(ft.Text("Ranking")),
        ],
        rows=[],
        border_radius=8
    )

    def reiniciar_interfaz(nueva_palabra=False):
        nonlocal contador, intentos, pistas_usadas
        contador = 0
        pistas_usadas = 0
        intentos.clear()
        tabla_intentos.rows = []
        resultado_destacado.visible = False
        mensaje_estado.value = ""
        input_palabra.value = ""
        input_palabra.disabled = False
        
        if nueva_palabra:
            actualizar_palabra_secreta()
            mensaje_estado.value = "🔄 ¡Nueva palabra secreta seleccionada!"
            mensaje_estado.color = "cyan400"
        
        page.update()

    def dar_pista_semantica(e):
        nonlocal pistas_usadas
        
        if pistas_usadas == 0:
            puesto_pista = 1000
            palabra_pista = orden_palabras_ranking[puesto_pista - 1]
            mensaje_estado.value = f"💡 Pista #1: La palabra '{palabra_pista}' está cerca, en el puesto #{puesto_pista}."
            pistas_usadas += 1
        elif pistas_usadas == 1:
            puesto_pista = 300
            palabra_pista = orden_palabras_ranking[puesto_pista - 1]
            mensaje_estado.value = f"💡 Pista #2: La palabra '{palabra_pista}' está muy cerca, en el puesto #{puesto_pista}."
            pistas_usadas += 1
        elif pistas_usadas == 2:
            inicial = PALABRA_SECRETA[0].upper()
            longitud = len(PALABRA_SECRETA)
            mensaje_estado.value = f"💡 Pista #3: Empieza por '{inicial}' y tiene {longitud} letras."
            pistas_usadas += 1
        else:
            mensaje_estado.value = "⚠️ No hay más pistas disponibles para esta partida."
            
        mensaje_estado.color = "amber300"
        page.update()

    def evaluar_palabra(e):
        nonlocal contador
        palabra_user = input_palabra.value.strip().lower()
        if not palabra_user:
            return

        if palabra_user not in dict_vocab:
            mensaje_estado.value = f"⚠️ '{palabra_user}' no está en el vocabulario."
            mensaje_estado.color = "orange400"
            page.update()
            return

        idx_user = dict_vocab[palabra_user]
        vec_user = tensor_vectores[idx_user]
        
        similitud = float(torch.dot(vector_secreto, vec_user).item())
        puesto = ranking_dict[palabra_user]
        
        contador += 1
        es_correcta = (palabra_user == PALABRA_SECRETA)
        color_puesto = obtener_color_temperatura(puesto)

        porcentaje_cercania = max(0.0, 1.0 - (puesto / TOTAL_PALABRAS))
        resultado_destacado.visible = True
        
        col_content = resultado_destacado.content.controls
        col_content[1].value = palabra_user.capitalize()
        col_content[1].color = color_puesto
        col_content[2].value = 1.0 if es_correcta else porcentaje_cercania
        col_content[2].color = color_puesto
        col_content[3].value = f"Puesto {puesto:,} de {TOTAL_PALABRAS:,} (Similitud: {similitud:.4f})"

        intentos.append({
            "num": contador,
            "palabra": palabra_user,
            "similitud": round(similitud, 4),
            "puesto": puesto,
            "color": color_puesto
        })

        intentos.sort(key=lambda x: x["puesto"])

        tabla_intentos.rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(str(item["num"]))),
                    ft.DataCell(ft.Text(item["palabra"], weight=ft.FontWeight.BOLD, color=item["color"])),
                    ft.DataCell(ft.Text(str(item["similitud"]))),
                    ft.DataCell(ft.Text(f"#{item['puesto']:,}", color=item["color"], weight=ft.FontWeight.BOLD)),
                ]
            ) for item in intentos
        ]

        if es_correcta:
            mensaje_estado.value = f"🎉 ¡Correcto! La palabra era '{PALABRA_SECRETA}' en {contador} intentos."
            mensaje_estado.color = "green400"
            input_palabra.disabled = True
        else:
            mensaje_estado.value = ""

        input_palabra.value = ""
        page.update()

    btn_probar = ft.ElevatedButton("Probar", on_click=evaluar_palabra)
    btn_pista = ft.OutlinedButton("Pista 💡", on_click=dar_pista_semantica)
    btn_reiniciar = ft.OutlinedButton("Reiniciar", icon="refresh", on_click=lambda e: reiniciar_interfaz(nueva_palabra=False))
    btn_nueva = ft.ElevatedButton("Nueva Palabra 🎲", icon="casino", on_click=lambda e: reiniciar_interfaz(nueva_palabra=True))

    page.add(
        header,
        ft.Divider(height=10, color="transparent"),
        ft.Row([input_palabra, btn_probar]),
        ft.Row([btn_pista, btn_nueva, btn_reiniciar], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        mensaje_estado,
        resultado_destacado,
        ft.Divider(height=10),
        ft.Text("Historial de intentos (ordenado por cercanía):", size=12, color="grey400"),
        ft.Column([tabla_intentos], scroll=ft.ScrollMode.AUTO, expand=True)
    )

ft.app(target=main)