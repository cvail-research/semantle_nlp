import math
import os
import random

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_ID_HF = "mick2332-q/semantle-es-vectors"
NOMBRE_ARCHIVO_CACHE = "vectores_es_fp16.pt"
RUTA_CACHE_BINARIO = os.path.join(
    BASE_DIR,
    NOMBRE_ARCHIVO_CACHE,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


PALABRAS_JUGABLES = [
    "guitarra",
    "soldado",
    "cocina",
    "hospital",
    "bicicleta",
    "invierno",
    "familia",
    "escuela",
    "pescado",
    "ventana",
    "camino",
    "silencio",
    "tormenta",
    "mercado",
    "caballo",
    "pintura",
    "medicina",
    "revista",
    "frontera",
    "naranja",
    "playa",
    "castillo",
    "orquesta",
    "desierto",
    "zapato",
    "cerebro",
    "bosque",
    "cerveza",
    "aeropuerto",
    "juguete",
]


CATEGORIAS = {
    "Animales": [
        "caballo",
        "pescado",
    ],
    "Lugares": [
        "hospital",
        "escuela",
        "mercado",
        "playa",
        "castillo",
        "aeropuerto",
        "bosque",
    ],
    "Objetos": [
        "guitarra",
        "bicicleta",
        "ventana",
        "revista",
        "zapato",
        "juguete",
    ],
    "Personas y sociedad": [
        "soldado",
        "familia",
        "frontera",
    ],
    "Naturaleza": [
        "invierno",
        "tormenta",
        "naranja",
        "desierto",
        "silencio",
    ],
    "Salud y cuerpo": [
        "cocina",
        "medicina",
        "cerebro",
    ],
    "Cultura y entretenimiento": [
        "pintura",
        "orquesta",
        "cerveza",
    ],
    "Transporte y caminos": [
        "camino",
    ],
}


def cargar_cache():
    if not os.path.exists(RUTA_CACHE_BINARIO):
        print("Descargando vectores desde Hugging Face...")

        hf_hub_download(
            repo_id=REPO_ID_HF,
            filename=NOMBRE_ARCHIVO_CACHE,
            local_dir=BASE_DIR,
        )

    print("Cargando vectores...")

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


PALABRAS, TENSOR_VECTORES, DICT_VOCAB = (
    cargar_cache()
)

TOTAL_PALABRAS = len(PALABRAS)


# Se normalizan una sola vez para poder calcular
# correctamente la similitud de coseno.
TENSOR_VECTORES_NORMALIZADOS = F.normalize(
    TENSOR_VECTORES,
    p=2,
    dim=1,
)


def obtener_categorias() -> dict[str, list[str]]:
    resultado = {}

    for categoria, palabras in CATEGORIAS.items():
        palabras_validas = sorted(
            palabra
            for palabra in palabras
            if palabra in DICT_VOCAB
        )

        if palabras_validas:
            resultado[categoria] = palabras_validas

    return resultado


def obtener_palabras_jugables() -> list[str]:
    palabras_validas = []

    for palabra in PALABRAS_JUGABLES:
        if palabra in DICT_VOCAB:
            palabras_validas.append(palabra)

    return sorted(palabras_validas)


def validar_palabra_jugable(
    palabra: str,
    categoria: str,
) -> bool:
    categorias = obtener_categorias()

    return (
        categoria in categorias
        and palabra in categorias[categoria]
    )


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


class GameEngine:
    def __init__(
        self,
        palabra_secreta: str,
    ):
        palabra_secreta = (
            palabra_secreta.strip().lower()
        )

        if palabra_secreta not in DICT_VOCAB:
            raise ValueError(
                "La palabra secreta no existe "
                "en el archivo .pt."
            )

        self.palabra_secreta = palabra_secreta

        indice = DICT_VOCAB[
            palabra_secreta
        ]

        self.vector_secreto = (
            TENSOR_VECTORES_NORMALIZADOS[indice]
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

    def evaluar(self, palabra: str) -> dict:
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
            TENSOR_VECTORES_NORMALIZADOS[indice]
        )

        similitud = float(
            torch.dot(
                self.vector_secreto,
                vector_palabra,
            ).item()
        )

        puesto = self.ranking_dict[palabra]

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
            "x": round(posicion_x, 5),
            "y": round(posicion_y, 5),
        }

    def obtener_pista(
        self,
        numero_pista: int,
    ) -> str:
        if numero_pista == 1:
            posicion = min(
                1000,
                TOTAL_PALABRAS,
            )

            palabra = (
                self.orden_palabras_ranking[
                    posicion - 1
                ]
            )

            return (
                f"La palabra '{palabra}' está cerca. "
                f"Está en el puesto #{posicion}."
            )

        if numero_pista == 2:
            posicion = min(
                300,
                TOTAL_PALABRAS,
            )

            palabra = (
                self.orden_palabras_ranking[
                    posicion - 1
                ]
            )

            return (
                f"La palabra '{palabra}' está muy "
                f"cerca. Está en el puesto #{posicion}."
            )

        if numero_pista == 3:
            inicial = (
                self.palabra_secreta[0].upper()
            )

            longitud = len(
                self.palabra_secreta
            )

            return (
                f"La palabra empieza por '{inicial}' "
                f"y tiene {longitud} letras."
            )

        return "No hay más pistas disponibles."