import math
import os
import random
import threading

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURACIÓN DE ARCHIVOS Y MODELO
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

REPO_ID_HF = (
    "mick2332-q/semantle-es-vectors"
)

NOMBRE_ARCHIVO_CACHE = (
    "vectores_es_fp16.pt"
)

RUTA_CACHE_BINARIO = os.path.join(
    BASE_DIR,
    NOMBRE_ARCHIVO_CACHE,
)


# Render no tiene GPU.
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CATEGORÍAS Y PISTAS MANUALES (PALABRAS CLAVE + PISTA FINAL)
# ============================================================

CATEGORIAS = {
    "Animales": [
        {
            "palabra": "caballo",
            "pistas": ["yegua", "jinete", "Empieza por 'C' y tiene 7 letras."]
        },
        {
            "palabra": "pescado",
            "pistas": ["atún", "mar", "Empieza por 'P' y tiene 7 letras."]
        }
    ],

    "Lugares": [
        {
            "palabra": "hospital",
            "pistas": ["medico", "enfermera", "Empieza por 'H' y tiene 8 letras."]
        },
        {
            "palabra": "escuela",
            "pistas": ["profesor", "estudiante", "Empieza por 'E' y tiene 7 letras."]
        },
        {
            "palabra": "mercado",
            "pistas": ["tienda", "comida", "Empieza por 'M' y tiene 7 letras."]
        },
        {
            "palabra": "playa",
            "pistas": ["arena", "mar", "Empieza por 'P' y tiene 5 letras."]
        },
        {
            "palabra": "castillo",
            "pistas": ["reina", "muralla", "Empieza por 'C' y tiene 8 letras."]
        },
        {
            "palabra": "aeropuerto",
            "pistas": ["avion", "viaje", "Empieza por 'A' y tiene 10 letras."]
        },
        {
            "palabra": "bosque",
            "pistas": ["arbol", "naturaleza", "Empieza por 'B' y tiene 6 letras."]
        }
    ],

    "Objetos": [
        {
            "palabra": "guitarra",
            "pistas": ["musica", "cuerdas", "Empieza por 'G' y tiene 8 letras."]
        },
        {
            "palabra": "bicicleta",
            "pistas": ["rueda", "pedal", "Empieza por 'B' y tiene 9 letras."]
        },
        {
            "palabra": "ventana",
            "pistas": ["vidrio", "casa", "Empieza por 'V' y tiene 7 letras."]
        },
        {
            "palabra": "revista",
            "pistas": ["papel", "lectura", "Empieza por 'R' y tiene 7 letras."]
        },
        {
            "palabra": "zapato",
            "pistas": ["pie", "bota", "Empieza por 'Z' y tiene 6 letras."]
        },
        {
            "palabra": "juguete",
            "pistas": ["muneco", "nino", "Empieza por 'J' y tiene 7 letras."]
        }
    ],

    "Personas y sociedad": [
        {
            "palabra": "soldado",
            "pistas": ["ejercito", "guerra", "Empieza por 'S' y tiene 7 letras."]
        },
        {
            "palabra": "familia",
            "pistas": ["padres", "casa", "Empieza por 'F' y tiene 7 letras."]
        },
        {
            "palabra": "frontera",
            "pistas": ["pais", "limite", "Empieza por 'F' y tiene 8 letras."]
        }
    ],

    "Naturaleza": [
        {
            "palabra": "invierno",
            "pistas": ["frio", "lluvia", "Empieza por 'I' y tiene 8 letras."]
        },
        {
            "palabra": "tormenta",
            "pistas": ["rayo", "viento", "Empieza por 'T' y tiene 8 letras."]
        },
        {
            "palabra": "naranja",
            "pistas": ["fruta", "jugo", "Empieza por 'N' y tiene 7 letras."]
        },
        {
            "palabra": "desierto",
            "pistas": ["arena", "calor", "Empieza por 'D' y tiene 8 letras."]
        },
        {
            "palabra": "silencio",
            "pistas": ["calma", "ruido", "Empieza por 'S' y tiene 8 letras."]
        }
    ],

    "Salud y cuerpo": [
        {
            "palabra": "cocina",
            "pistas": ["comida", "fuego", "Empieza por 'C' y tiene 6 letras."]
        },
        {
            "palabra": "medicina",
            "pistas": ["doctor", "pastilla", "Empieza por 'M' y tiene 8 letras."]
        },
        {
            "palabra": "cerebro",
            "pistas": ["mente", "cabeza", "Empieza por 'C' y tiene 7 letras."]
        }
    ],

    "Transportes": [
        
        {
            "palabra": "carro",
            "pistas": ["electricidad", "chasis", "Empieza por 'C' y tiene 5 letras."]
        },
        {
            "palabra": "moto",
            "pistas": ["pato", "dos", "Empieza por 'M' y tiene 4 letras."]
        },
        {
            "palabra": "avion",
            "pistas": ["aeropuerto", "vuelo", "Empieza por 'A' y tiene 5 letras."]
        },
        {
            "palabra": "barco",
            "pistas": ["nave", "puerto", "Empieza por 'B' y tiene 5 letras."]
        },
        {
            "palabra": "bicicleta",
            "pistas": ["pedal", "rueda", "Empieza por 'B' y tiene 9 letras."]
        }
    ],
}


# ============================================================
# CARGA DIFERIDA DE VECTORES
# ============================================================

PALABRAS = None
TENSOR_VECTORES = None
DICT_VOCAB = None

TOTAL_PALABRAS = 0

TENSOR_VECTORES_NORMALIZADOS = None

_CACHE_CARGADA = False

_CACHE_LOCK = threading.Lock()


def cargar_cache():
    if not os.path.exists(
        RUTA_CACHE_BINARIO
    ):
        print(
            "Descargando vectores desde "
            "Hugging Face...",
            flush=True,
        )

        hf_hub_download(
            repo_id=REPO_ID_HF,
            filename=NOMBRE_ARCHIVO_CACHE,
            local_dir=BASE_DIR,
        )

    print(
        "Cargando vectores...",
        flush=True,
    )

    datos = torch.load(
        RUTA_CACHE_BINARIO,
        map_location=DEVICE,
        weights_only=False,
    )

    palabras = datos["palabras"]

    tensor_vectores = datos[
        "tensor_vectores"
    ].to(DEVICE)

    dict_vocab = datos["dict_vocab"]

    return (
        palabras,
        tensor_vectores,
        dict_vocab,
    )


def asegurar_cache():
    global PALABRAS
    global TENSOR_VECTORES
    global DICT_VOCAB
    global TOTAL_PALABRAS
    global TENSOR_VECTORES_NORMALIZADOS
    global _CACHE_CARGADA

    if _CACHE_CARGADA:
        return

    with _CACHE_LOCK:
        if _CACHE_CARGADA:
            return

        print(
            "Cargando vectores semánticos...",
            flush=True,
        )

        (
            PALABRAS,
            TENSOR_VECTORES,
            DICT_VOCAB,
        ) = cargar_cache()

        TOTAL_PALABRAS = len(
            PALABRAS
        )

        TENSOR_VECTORES_NORMALIZADOS = (
            F.normalize(
                TENSOR_VECTORES,
                p=2,
                dim=1,
            )
        )

        _CACHE_CARGADA = True

        print(
            "Vectores semánticos cargados.",
            flush=True,
        )


# ============================================================
# CATEGORÍAS Y VALIDACIONES
# ============================================================

def obtener_categorias() -> dict[str, list[str]]:
    asegurar_cache()

    resultado = {}

    for categoria, items in CATEGORIAS.items():
        palabras_validas = []
        for item in items:
            palabra = item["palabra"]
            if palabra in DICT_VOCAB:
                palabras_validas.append(palabra)

        if palabras_validas:
            resultado[categoria] = sorted(palabras_validas)

    return resultado


def obtener_palabras_jugables() -> list[str]:
    asegurar_cache()

    palabras_validas = set()
    for items in CATEGORIAS.values():
        for item in items:
            palabra = item["palabra"]
            if palabra in DICT_VOCAB:
                palabras_validas.add(palabra)

    return sorted(list(palabras_validas))


def validar_palabra_jugable(
    palabra: str,
    categoria: str,
) -> bool:
    cat_validas = obtener_categorias()

    palabra = palabra.strip().lower()
    categoria = categoria.strip()

    return (
        categoria in cat_validas
        and palabra in cat_validas[categoria]
    )


# ============================================================
# COLORES SEGÚN LA CERCANÍA
# ============================================================

def obtener_color_temperatura(
    puesto: int,
) -> str:
    if puesto == 1:
        return "#00E676"

    if puesto <= 100:
        return "#66BB6A"

    if puesto <= 1000:
        return "#D4E157"

    if puesto <= 5000:
        return "#FFCA28"

    if puesto <= 20000:
        return "#FFA726"

    return "#EF5350"


# ============================================================
# MOTOR DEL JUEGO
# ============================================================

class GameEngine:
    def __init__(
        self,
        palabra_secreta: str,
    ):
        asegurar_cache()

        palabra_secreta = (
            palabra_secreta.strip().lower()
        )

        if palabra_secreta not in DICT_VOCAB:
            raise ValueError(
                "La palabra secreta no existe "
                "en el archivo .pt."
            )

        self.palabra_secreta = (
            palabra_secreta
        )

        indice = DICT_VOCAB[
            palabra_secreta
        ]

        self.vector_secreto = (
            TENSOR_VECTORES_NORMALIZADOS[
                indice
            ]
        )

        self.ranking_dict = {}
        self.orden_palabras_ranking = []

        self.calcular_ranking()

    def calcular_ranking(self):
        similitudes = torch.matmul(
            TENSOR_VECTORES_NORMALIZADOS,
            self.vector_secreto,
        )

        indices = torch.argsort(
            similitudes,
            descending=True,
        ).cpu().tolist()

        self.orden_palabras_ranking = [
            PALABRAS[indice]
            for indice in indices
        ]

        self.ranking_dict = {
            PALABRAS[indice]: posicion + 1
            for posicion, indice in enumerate(
                indices
            )
        }

    def evaluar(
        self,
        palabra: str,
    ) -> dict:
        palabra = palabra.strip().lower()

        if not palabra:
            raise ValueError(
                "Debe escribir una palabra."
            )

        if palabra not in DICT_VOCAB:
            raise ValueError(
                f"La palabra '{palabra}' "
                "no existe en el vocabulario."
            )

        indice = DICT_VOCAB[palabra]

        vector_palabra = (
            TENSOR_VECTORES_NORMALIZADOS[
                indice
            ]
        )

        similitud = float(
            torch.dot(
                self.vector_secreto,
                vector_palabra,
            ).item()
        )

        puesto = self.ranking_dict[
            palabra
        ]

        es_correcta = (
            palabra == self.palabra_secreta
        )

        distancia = max(
            0.0,
            1.0 - max(0.0, similitud),
        )

        angulo = random.uniform(
            0,
            2 * math.pi,
        )

        posicion_x = (
            distancia * math.cos(angulo)
            if not es_correcta
            else 0.0
        )

        posicion_y = (
            distancia * math.sin(angulo)
            if not es_correcta
            else 0.0
        )

        porcentaje_cercania = max(
            0.0,
            1.0 - (
                puesto / TOTAL_PALABRAS
            ),
        )

        return {
            "palabra": palabra,
            "similitud": round(
                similitud,
                4,
            ),
            "puesto": puesto,
            "porcentaje_cercania": round(
                porcentaje_cercania,
                4,
            ),
            "es_correcta": es_correcta,
            "color": obtener_color_temperatura(
                puesto
            ),
            "x": round(
                posicion_x,
                5,
            ),
            "y": round(
                posicion_y,
                5,
            ),
        }

    def obtener_pista(
        self,
        numero_pista: int,
    ) -> str:
        # Buscar las pistas personalizadas de esta palabra secreta
        pistas_lista = None
        for cat_items in CATEGORIAS.values():
            for item in cat_items:
                if item["palabra"] == self.palabra_secreta:
                    pistas_lista = item["pistas"]
                    break
            if pistas_lista:
                break

        # Si se encontraron pistas definidas manualmente para este número (1, 2 o 3)
        if pistas_lista and 0 <= numero_pista - 1 < len(pistas_lista):
            pista_texto = pistas_lista[numero_pista - 1]
            
            # Si las primeras dos pistas son palabras, evaluamos su cercanía automáticamente en el juego
            if numero_pista in [1, 2] and pista_texto in self.ranking_dict:
                puesto_pista = self.ranking_dict[pista_texto]
                return f"Pista {numero_pista}: La palabra relacionada '{pista_texto}' está en el puesto #{puesto_pista}."
            else:
                return f"Pista {numero_pista}: {pista_texto}"

        # Respaldo por defecto si faltara alguna pista
        if numero_pista == 1:
            pos = min(1000, TOTAL_PALABRAS)
            return f"Pista 1: Una palabra cercana está en el puesto #{pos}: '{self.orden_palabras_ranking[pos - 1]}'."
        elif numero_pista == 2:
            pos = min(300, TOTAL_PALABRAS)
            return f"Pista 2: Una palabra muy cercana está en el puesto #{pos}: '{self.orden_palabras_ranking[pos - 1]}'."
        elif numero_pista == 3:
            return f"Pista 3: La palabra empieza por '{self.palabra_secreta[0].upper()}' y tiene {len(self.palabra_secreta)} letras."

        return "No hay más pistas disponibles."