import math
import os
import random
import re
import threading
import unicodedata

import torch
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

# Archivo original de Hugging Face: 2.000.000 de entradas en fp16 (1,2 GB).
ARCHIVO_ORIGEN = "vectores_es_fp16.pt"

# Archivo derivado que realmente usa el juego: solo palabras limpias en fp32.
ARCHIVO_VOCABULARIO = "vocabulario_es.pt"

RUTA_ORIGEN = os.path.join(
    BASE_DIR,
    ARCHIVO_ORIGEN,
)

RUTA_VOCABULARIO = os.path.join(
    BASE_DIR,
    ARCHIVO_VOCABULARIO,
)


# El archivo original viene ordenado por frecuencia de uso. Más allá de las
# primeras 200.000 entradas solo quedan nombres propios raros, fechas, URLs y
# fragmentos rotos ("valorDato.value", "Orbitz.com134"), que arruinaban el
# ranking del juego.
LIMITE_ORIGEN = 200_000

LONGITUD_MINIMA = 3
LONGITUD_MAXIMA = 16

PATRON_PALABRA = re.compile(
    r"^[a-záéíóúüñ]+$"
)


# ============================================================
# CATEGORÍAS JUGABLES
# ============================================================
#
# Todas las pistas están escritas a mano. NO se generan desde el corpus: el
# texto del que salen los vectores viene de la web y contiene vocabulario
# adulto, así que "la palabra vecina número 41" puede ser cualquier cosa.

CATEGORIAS = {
    "Animales de granja": [
        {"palabra": "vaca", "pistas": ["establo", "leche"]},
        {"palabra": "caballo", "pistas": ["yegua", "jinete"]},
        {"palabra": "oveja", "pistas": ["pastor", "lana"]},
        {"palabra": "gallina", "pistas": ["gallo", "huevo"]},
        {"palabra": "pato", "pistas": ["estanque", "plumas"]},
        {"palabra": "conejo", "pistas": ["zanahoria", "madriguera"]},
        {"palabra": "burro", "pistas": ["establo", "mula"]},
        {"palabra": "cabra", "pistas": ["monte", "leche"]},
    ],
    "Animales del mar": [
        {"palabra": "ballena", "pistas": ["gigante", "océano"]},
        {"palabra": "delfín", "pistas": ["marino", "ballena"]},
        {"palabra": "pulpo", "pistas": ["molusco", "calamar"]},
        {"palabra": "cangrejo", "pistas": ["marisco", "langosta"]},
        {"palabra": "tortuga", "pistas": ["lenta", "caparazón"]},
        {"palabra": "foca", "pistas": ["nadar", "hielo"]},
        {"palabra": "estrella", "pistas": ["cielo", "brillar"]},
        {"palabra": "coral", "pistas": ["colores", "arrecife"]},
    ],
    "Animales de la selva": [
        {"palabra": "león", "pistas": ["melena", "rugido"]},
        {"palabra": "tigre", "pistas": ["rayas", "felino"]},
        {"palabra": "elefante", "pistas": ["colmillos", "trompa"]},
        {"palabra": "jirafa", "pistas": ["sabana", "cebra"]},
        {"palabra": "mono", "pistas": ["plátano", "árbol"]},
        {"palabra": "cocodrilo", "pistas": ["mandíbula", "río"]},
        {"palabra": "serpiente", "pistas": ["escamas", "reptil"]},
        {"palabra": "loro", "pistas": ["hablar", "plumas"]},
    ],
    "Insectos": [
        {"palabra": "abeja", "pistas": ["miel", "panal"]},
        {"palabra": "mariposa", "pistas": ["alas", "flor"]},
        {"palabra": "araña", "pistas": ["patas", "telaraña"]},
        {"palabra": "hormiga", "pistas": ["bicho", "insecto"]},
        {"palabra": "mosca", "pistas": ["zumbido", "insecto"]},
        {"palabra": "grillo", "pistas": ["chirrido", "saltamontes"]},
        {"palabra": "escarabajo", "pistas": ["caparazón", "insecto"]},
        {"palabra": "oruga", "pistas": ["hojas", "mariposa"]},
    ],
    "Instrumentos musicales": [
        {"palabra": "guitarra", "pistas": ["música", "cuerdas"]},
        {"palabra": "piano", "pistas": ["teclas", "melodía"]},
        {"palabra": "violín", "pistas": ["arco", "orquesta"]},
        {"palabra": "tambor", "pistas": ["ritmo", "percusión"]},
        {"palabra": "flauta", "pistas": ["melodía", "oboe"]},
        {"palabra": "trompeta", "pistas": ["metal", "banda"]},
        {"palabra": "arpa", "pistas": ["ángel", "cuerdas"]},
        {"palabra": "acordeón", "pistas": ["folclor", "fuelle"]},
    ],
    "En la cocina": [
        {"palabra": "cuchara", "pistas": ["cubiertos", "sopa"]},
        {"palabra": "sartén", "pistas": ["aceite", "freír"]},
        {"palabra": "nevera", "pistas": ["conservar", "frío"]},
        {"palabra": "horno", "pistas": ["calor", "hornear"]},
        {"palabra": "plato", "pistas": ["mesa", "comida"]},
        {"palabra": "taza", "pistas": ["beber", "café"]},
        {"palabra": "olla", "pistas": ["hervir", "guiso"]},
        {"palabra": "tenedor", "pistas": ["cubiertos", "pinchar"]},
    ],
    "Útiles de la escuela": [
        {"palabra": "lápiz", "pistas": ["escribir", "punta"]},
        {"palabra": "cuaderno", "pistas": ["hojas", "apuntes"]},
        {"palabra": "mochila", "pistas": ["llevar", "espalda"]},
        {"palabra": "tijeras", "pistas": ["papel", "cortar"]},
        {"palabra": "pizarra", "pistas": ["aula", "tiza"]},
        {"palabra": "libro", "pistas": ["páginas", "leer"]},
        {"palabra": "pegamento", "pistas": ["pegar", "goma"]},
        {"palabra": "diccionario", "pistas": ["palabras", "significado"]},
    ],
    "Vehículos": [
        {"palabra": "tren", "pistas": ["estación", "vagón"]},
        {"palabra": "barco", "pistas": ["navegar", "puerto"]},
        {"palabra": "bicicleta", "pistas": ["pedal", "ruedas"]},
        {"palabra": "camión", "pistas": ["carga", "carretera"]},
        {"palabra": "tractor", "pistas": ["campo", "arar"]},
        {"palabra": "moto", "pistas": ["velocidad", "casco"]},
        {"palabra": "autobús", "pistas": ["pasajeros", "parada"]},
        {"palabra": "ambulancia", "pistas": ["sirena", "hospital"]},
    ],
    "Frutas y verduras": [
        {"palabra": "naranja", "pistas": ["jugo", "cítrico"]},
        {"palabra": "manzana", "pistas": ["árbol", "roja"]},
        {"palabra": "plátano", "pistas": ["amarillo", "cáscara"]},
        {"palabra": "fresa", "pistas": ["roja", "dulce"]},
        {"palabra": "zanahoria", "pistas": ["conejo", "naranja"]},
        {"palabra": "tomate", "pistas": ["ensalada", "salsa"]},
        {"palabra": "lechuga", "pistas": ["verde", "ensalada"]},
        {"palabra": "sandía", "pistas": ["verano", "semillas"]},
    ],
    "El clima": [
        {"palabra": "lluvia", "pistas": ["nubes", "paraguas"]},
        {"palabra": "nieve", "pistas": ["frío", "invierno"]},
        {"palabra": "tormenta", "pistas": ["rayo", "viento"]},
        {"palabra": "viento", "pistas": ["aire", "soplar"]},
        {"palabra": "niebla", "pistas": ["visibilidad", "humedad"]},
        {"palabra": "relámpago", "pistas": ["luz", "trueno"]},
        {"palabra": "granizo", "pistas": ["piedras", "hielo"]},
        {"palabra": "arcoíris", "pistas": ["lluvia", "colores"]},
    ],
    "El espacio": [
        {"palabra": "luna", "pistas": ["satélite", "noche"]},
        {"palabra": "planeta", "pistas": ["sistema", "órbita"]},
        {"palabra": "cohete", "pistas": ["espacio", "despegue"]},
        {"palabra": "galaxia", "pistas": ["estrellas", "universo"]},
        {"palabra": "telescopio", "pistas": ["observar", "lentes"]},
        {"palabra": "astronauta", "pistas": ["gravedad", "traje"]},
        {"palabra": "meteorito", "pistas": ["roca", "cráter"]},
        {"palabra": "eclipse", "pistas": ["sombra", "sol"]},
    ],
    "Deportes": [
        {"palabra": "natación", "pistas": ["piscina", "nadar"]},
        {"palabra": "ciclismo", "pistas": ["carrera", "bicicleta"]},
        {"palabra": "tenis", "pistas": ["pelota", "raqueta"]},
        {"palabra": "baloncesto", "pistas": ["canasta", "balón"]},
        {"palabra": "patinaje", "pistas": ["hielo", "patines"]},
        {"palabra": "gimnasia", "pistas": ["entrenamiento", "acrobacia"]},
        {"palabra": "atletismo", "pistas": ["correr", "pista"]},
        {"palabra": "ajedrez", "pistas": ["estrategia", "tablero"]},
    ],
    "Oficios": [
        {"palabra": "bombero", "pistas": ["manguera", "incendio"]},
        {"palabra": "panadero", "pistas": ["horno", "pan"]},
        {"palabra": "cocinero", "pistas": ["recetas", "cocina"]},
        {"palabra": "carpintero", "pistas": ["muebles", "madera"]},
        {"palabra": "jardinero", "pistas": ["plantas", "jardín"]},
        {"palabra": "veterinario", "pistas": ["animales", "mascotas"]},
        {"palabra": "astrónomo", "pistas": ["estrellas", "observatorio"]},
        {"palabra": "arquitecto", "pistas": ["planos", "edificios"]},
    ],
    "Lugares de la ciudad": [
        {"palabra": "hospital", "pistas": ["enfermera", "médico"]},
        {"palabra": "escuela", "pistas": ["profesor", "alumnos"]},
        {"palabra": "mercado", "pistas": ["tienda", "comprar"]},
        {"palabra": "biblioteca", "pistas": ["lectura", "libros"]},
        {"palabra": "museo", "pistas": ["exposición", "arte"]},
        {"palabra": "parque", "pistas": ["árboles", "banco"]},
        {"palabra": "estadio", "pistas": ["público", "cancha"]},
        {"palabra": "panadería", "pistas": ["pastel", "pan"]},
    ],
    "En la casa": [
        {"palabra": "ventana", "pistas": ["luz", "vidrio"]},
        {"palabra": "escalera", "pistas": ["subir", "peldaños"]},
        {"palabra": "jardín", "pistas": ["plantas", "flores"]},
        {"palabra": "espejo", "pistas": ["vidrio", "reflejo"]},
        {"palabra": "alfombra", "pistas": ["tejido", "suelo"]},
        {"palabra": "almohada", "pistas": ["cabeza", "dormir"]},
        {"palabra": "chimenea", "pistas": ["humo", "fuego"]},
        {"palabra": "balcón", "pistas": ["barandilla", "terraza"]},
    ],
    "Cuentos y fantasía": [
        {"palabra": "dragón", "pistas": ["leyenda", "fuego"]},
        {"palabra": "castillo", "pistas": ["muralla", "torre"]},
        {"palabra": "princesa", "pistas": ["reino", "corona"]},
        {"palabra": "mago", "pistas": ["varita", "hechizo"]},
        {"palabra": "hada", "pistas": ["alas", "magia"]},
        {"palabra": "gigante", "pistas": ["ogro", "enano"]},
        {"palabra": "tesoro", "pistas": ["oro", "cofre"]},
        {"palabra": "duende", "pistas": ["bosque", "travieso"]},
    ],
    "Paisajes de la naturaleza": [
        {"palabra": "bosque", "pistas": ["naturaleza", "árboles"]},
        {"palabra": "montaña", "pistas": ["escalar", "cima"]},
        {"palabra": "playa", "pistas": ["mar", "arena"]},
        {"palabra": "desierto", "pistas": ["calor", "arena"]},
        {"palabra": "isla", "pistas": ["mar", "costa"]},
        {"palabra": "volcán", "pistas": ["lava", "erupción"]},
        {"palabra": "cascada", "pistas": ["agua", "caída"]},
        {"palabra": "selva", "pistas": ["húmeda", "vegetación"]},
    ],
}

# ============================================================
# CARGA DEL VOCABULARIO
# ============================================================

PALABRAS = None
VECTORES = None
DICT_VOCAB = None
ALIAS_SIN_ACENTOS = None

# Variantes mal acentuadas que apuntan a la palabra correcta.
REDIRECCIONES = {}

# Cuántas veces más rara debe ser una variante para considerarla un error de
# tipeo y no una palabra distinta.
FACTOR_VARIANTE_RARA = 5

TOTAL_PALABRAS = 0
LOG_TOTAL = 1.0

_CACHE_CARGADA = False
_CACHE_LOCK = threading.Lock()


def quitar_acentos(texto: str) -> str:
    descompuesto = unicodedata.normalize(
        "NFD",
        texto,
    )

    return "".join(
        caracter
        for caracter in descompuesto
        if unicodedata.category(caracter) != "Mn"
    )


def _formas_familia(palabra: str) -> set[str]:
    """
    Singular y plural posibles de una palabra, sin acentos.

    Sirve para no mostrar "cráter" y "cráteres" como si fueran dos ideas
    distintas en la pantalla educativa. En español el plural se forma con "s"
    tras vocal y con "es" tras consonante, y desde la forma plural no se sabe
    cuál de las dos fue ("jinetes" podría venir de "jinete" o de "jinet"), así
    que comparamos todas las variantes en vez de intentar adivinar una sola.
    """

    simple = quitar_acentos(palabra)

    formas = {simple, simple + "s", simple + "es"}

    for terminacion in ("es", "s"):
        if (
            simple.endswith(terminacion)
            and len(simple) - len(terminacion) >= 3
        ):
            formas.add(simple[: -len(terminacion)])

    return formas


def _palabras_jugables_configuradas() -> list[str]:
    palabras = []

    for items in CATEGORIAS.values():
        for item in items:
            palabras.append(
                item["palabra"]
            )

    return palabras


def _construir_vocabulario():
    """
    Genera el archivo reducido a partir del original de Hugging Face.

    Solo corre la primera vez. Después el servidor únicamente lee el archivo
    pequeño, lo que baja el arranque de ~15 s a menos de 1 s y el uso de
    memoria de ~3 GB a ~250 MB.
    """

    if not os.path.exists(RUTA_ORIGEN):
        print(
            "Descargando vectores desde "
            "Hugging Face (1,2 GB, solo la "
            "primera vez)...",
            flush=True,
        )

        hf_hub_download(
            repo_id=REPO_ID_HF,
            filename=ARCHIVO_ORIGEN,
            local_dir=BASE_DIR,
        )

    print(
        "Construyendo vocabulario limpio "
        "(solo la primera vez)...",
        flush=True,
    )

    datos = torch.load(
        RUTA_ORIGEN,
        map_location="cpu",
        weights_only=False,
    )

    palabras_origen = datos["palabras"]
    tensor_origen = datos["tensor_vectores"]

    limite = min(
        LIMITE_ORIGEN,
        len(palabras_origen),
    )

    indices = []
    palabras = []
    vistas = set()

    for indice in range(limite):
        palabra = palabras_origen[indice]

        if not (
            LONGITUD_MINIMA
            <= len(palabra)
            <= LONGITUD_MAXIMA
        ):
            continue

        if not PATRON_PALABRA.match(palabra):
            continue

        indices.append(indice)
        palabras.append(palabra)
        vistas.add(palabra)

    # Las palabras jugables entran siempre, incluso si quedaron fuera del corte
    # de frecuencia o si el filtro las descartó.
    indice_origen = {
        palabra: posicion
        for posicion, palabra in enumerate(
            palabras_origen
        )
    }

    for palabra in _palabras_jugables_configuradas():
        if palabra in vistas:
            continue

        posicion = indice_origen.get(palabra)

        if posicion is None:
            print(
                f"  Aviso: '{palabra}' no existe "
                "en los vectores y no será jugable.",
                flush=True,
            )
            continue

        indices.append(posicion)
        palabras.append(palabra)
        vistas.add(palabra)

    vectores = tensor_origen[
        torch.tensor(indices)
    ].to(torch.float32)

    vectores /= (
        vectores
        .norm(dim=1, keepdim=True)
        .clamp_min(1e-8)
    )

    torch.save(
        {
            "palabras": palabras,
            "vectores": vectores,
        },
        RUTA_VOCABULARIO,
    )

    print(
        f"Vocabulario listo: {len(palabras)} "
        "palabras.",
        flush=True,
    )


def asegurar_cache():
    global PALABRAS
    global VECTORES
    global DICT_VOCAB
    global ALIAS_SIN_ACENTOS
    global TOTAL_PALABRAS
    global LOG_TOTAL
    global _CACHE_CARGADA

    if _CACHE_CARGADA:
        return

    with _CACHE_LOCK:
        if _CACHE_CARGADA:
            return

        if not os.path.exists(
            RUTA_VOCABULARIO
        ):
            _construir_vocabulario()

        datos = torch.load(
            RUTA_VOCABULARIO,
            map_location="cpu",
            weights_only=False,
        )

        PALABRAS = datos["palabras"]
        VECTORES = datos["vectores"]

        DICT_VOCAB = {
            palabra: indice
            for indice, palabra in enumerate(
                PALABRAS
            )
        }

        # Permite que un niño escriba "corazon" y el juego entienda "corazón".
        #
        # El corpus también contiene la variante sin tilde como token aparte,
        # pero con un vector mucho peor (sale de textos mal escritos). Cuando
        # una variante es bastante más rara que la otra la tratamos como error
        # de tipeo y la redirigimos a la frecuente. Si las dos son comunes
        # ("papa" y "papá") las dejamos separadas, porque ahí el acento sí
        # cambia el significado.
        variantes = {}

        for palabra, indice in DICT_VOCAB.items():
            variantes.setdefault(
                quitar_acentos(palabra),
                [],
            ).append(indice)

        ALIAS_SIN_ACENTOS = {}

        for simple, indices in variantes.items():
            indices.sort()

            canonico = indices[0]

            ALIAS_SIN_ACENTOS[simple] = canonico

            for otro in indices[1:]:
                if otro > canonico * FACTOR_VARIANTE_RARA:
                    REDIRECCIONES[otro] = canonico

        TOTAL_PALABRAS = len(PALABRAS)
        LOG_TOTAL = math.log(TOTAL_PALABRAS)

        _CACHE_CARGADA = True

        print(
            f"Vocabulario cargado: "
            f"{TOTAL_PALABRAS} palabras.",
            flush=True,
        )


# ============================================================
# BÚSQUEDA DE PALABRAS
# ============================================================

def buscar_indice(palabra: str):
    """
    Devuelve el índice de la palabra, tolerando acentos y mayúsculas.
    """

    asegurar_cache()

    palabra = palabra.strip().lower()

    if not palabra:
        return None

    indice = DICT_VOCAB.get(palabra)

    if indice is not None:
        return REDIRECCIONES.get(indice, indice)

    return ALIAS_SIN_ACENTOS.get(
        quitar_acentos(palabra)
    )


# ============================================================
# CATEGORÍAS Y VALIDACIONES
# ============================================================

_CATEGORIAS_VALIDAS = None


def obtener_categorias() -> dict[str, list[str]]:
    global _CATEGORIAS_VALIDAS

    if _CATEGORIAS_VALIDAS is not None:
        return _CATEGORIAS_VALIDAS

    asegurar_cache()

    resultado = {}

    for categoria, items in CATEGORIAS.items():
        palabras_validas = []

        for item in items:
            palabra = item["palabra"]

            if palabra in DICT_VOCAB:
                palabras_validas.append(
                    palabra
                )

        if palabras_validas:
            resultado[categoria] = sorted(
                palabras_validas
            )

    _CATEGORIAS_VALIDAS = resultado

    return resultado


def obtener_palabras_jugables() -> list[str]:
    palabras = set()

    for lista in obtener_categorias().values():
        palabras.update(lista)

    return sorted(palabras)


def validar_palabra_jugable(
    palabra: str,
    categoria: str,
) -> bool:
    categorias = obtener_categorias()

    palabra = palabra.strip().lower()
    categoria = categoria.strip()

    return (
        categoria in categorias
        and palabra in categorias[categoria]
    )


def elegir_palabra_al_azar(
    excluir: set[str] | None = None,
) -> tuple[str, str]:
    """
    Devuelve (categoría, palabra) al azar entre las jugables.
    """

    categorias = obtener_categorias()

    opciones = [
        (categoria, palabra)
        for categoria, palabras in categorias.items()
        for palabra in palabras
    ]

    if excluir:
        disponibles = [
            opcion
            for opcion in opciones
            if opcion[1] not in excluir
        ]

        if disponibles:
            opciones = disponibles

    return random.choice(opciones)


# ============================================================
# TEMPERATURA Y COLOR SEGÚN LA CERCANÍA
# ============================================================

def calcular_temperatura(puesto: int) -> float:
    """
    Convierte un puesto (1 = la palabra secreta) en un número de 0 a 100.

    Usa escala logarítmica porque la lineal era inútil: con 95.000 palabras,
    el puesto 1 y el puesto 500 daban ambos "99%". Así el puesto 10 da 80,
    el 100 da 60 y el 1.000 da 40, que es lo que un niño percibe como
    "me estoy acercando".
    """

    if puesto <= 1:
        return 100.0

    valor = 100.0 * (
        1.0 - math.log(puesto) / LOG_TOTAL
    )

    return round(
        max(0.0, min(100.0, valor)),
        1,
    )


def obtener_color_temperatura(
    puesto: int,
) -> str:
    if puesto == 1:
        return "#00E676"

    if puesto <= 10:
        return "#76FF03"

    if puesto <= 100:
        return "#C6FF00"

    if puesto <= 1000:
        return "#FFD600"

    if puesto <= 10000:
        return "#FF9100"

    return "#FF5252"


def describir_temperatura(
    puesto: int,
) -> str:
    if puesto == 1:
        return "¡Es la palabra!"

    if puesto <= 10:
        return "🔥 Quemando"

    if puesto <= 100:
        return "🌡️ Muy caliente"

    if puesto <= 1000:
        return "☀️ Caliente"

    if puesto <= 10000:
        return "❄️ Frío"

    return "🧊 Congelado"



# ============================================================
# FILTRO DE CONTENIDO
# ============================================================
#
# El vocabulario sale de texto web, así que incluye lenguaje adulto. Se
# comparan palabras COMPLETAS, no prefijos: si no, "armario" caería por "arma"
# y "matemático" por "matar".
#
# OJO: el texto se parte por espacios, así que aquí dentro SOLO pueden ir
# términos a bloquear. Un comentario suelto convertiría cada una de sus
# palabras ("de", "en", "el"...) en término bloqueado.

_TERMINOS_BLOQUEADOS = """
sexo sexual sexualidad erotico erotica porno pornografia semen esperma
pene vagina vulva clitoris testiculo escroto pezon seno senos teta tetas
culo nalga nalgas ano anal coito orgasmo eyaculacion masturbacion
desnudo desnuda desnudez nudista prostituta prostituto prostitucion burdel
proxeneta condon preservativo anticonceptivo aborto abortar menstruacion
utero ovario ovulo virgen virginidad libido violacion violador pedofilia
pedofilo incesto sodomia afrodisiaco

puta puto perra zorra cabron cabrona pendejo pendeja gilipollas imbecil
idiota estupido estupida mierda joder follar coño carajo maricon marica
bastardo pervertido

droga drogas drogadicto cocaina marihuana heroina cannabis porro crack
narcotrafico narcotraficante adiccion sobredosis alcohol alcoholico
borracho borracha ebrio embriaguez cerveza whisky vodka ron licor tequila
tabaco cigarrillo cigarro fumar fumador nicotina

arma armas armamento pistola pistolas revolver fusil escopeta rifle
metralleta bala balas municion disparo disparar tiroteo bomba bombas
bombardero bombardeo granada explosivo explosion misil torpedo canon
cuchillo cuchillos navaja punal machete espada daga

asesino asesina asesinato asesinar homicidio matar mata matanza masacre
suicidio suicidarse cadaver cadaveres muerto muerta muerte morir agonia
sangre sangriento sangrienta herida heridas tortura torturar mutilacion
decapitar secuestro secuestrar rehen violencia violento violenta
terrorismo terrorista atentado guerra guerras genocidio

demonio demonios satan satanas diablo infierno brujeria exorcismo
maldito maldita
cancer tumor sida leucemia

gonorrea malparido malparida hijueputa jueputa chimba picha verga chucha
culicagado gueva guevon huevon boludo pelotudo pija poronga concha forro
panocha pinche chingar chinga capullo polla nabo rabo culear mamada
pajero paja putear zorrear ramera fufurufa

sicario sicarios narco narcos bazuco perico marimba coca
guerrilla guerrillero guerrilleros guerrillera paramilitar paramilitares
paraco paracos autodefensas milicia miliciano milicianos combatiente
combatientes insurgente insurgentes secuestrado secuestrados
desaparecido desaparecidos fosa fosas masacrar degollar degollado
apunalar apunalado balacera balazo tiroteos exterminio

violado violada abusar abusada acoso acosar acosada maltrato maltratar
golpiza paliza agredir agresion agresor vejacion

drogado drogada drogadicto drogadicta alcoholismo borrachera embriagado
embriagada ebriedad basuco

suicidarse ahorcarse ahorcado 
eyacular masturbar masturbarse penetrar penetracion desvirgar fornicar
promiscuo promiscua farc sexy trasero traseros
lascivo lasciva obsceno obscena vulgaridad morbo morboso
asesinado asesinada ejecutado ejecutada fallecido difunto
"""

PALABRAS_BLOQUEADAS = frozenset(
    _TERMINOS_BLOQUEADOS.split()
)


# Comparar palabras completas no alcanza: el español cambia género y conjuga,
# así que "asesinaron", "torturado", "secuestrada" y "suicida" se escapaban de
# una lista que solo tenía "asesinato", "tortura", "secuestro" y "suicidio".
#
# Estas raíces sí se comparan por prefijo. Solo entran aquí las que no tienen
# ninguna palabra inocente debajo: "violen" atrapa violento y violencia pero no
# toca violín ni violeta, y "asesin" no alcanza a asesor ni asesoría.
_RAICES_BLOQUEADAS = """
asesin tortur secuestr suicid violen violacion violad
masturb eyacul prostitu proxenet pedofil vagin sexual pornogra
drogadic alcoholi narcotrafic
degoll apunal masacr genocid mutil decapit
maltrat golpiz agresion agresor
estupid imbecil gilipoll
fallecid cadaver
"""

RAICES_BLOQUEADAS = tuple(
    _RAICES_BLOQUEADAS.split()
)


def es_apropiada(palabra: str) -> bool:
    """
    False si la palabra no debería mostrársele a un niño.

    Dos pasadas: palabra completa (con su plural) contra PALABRAS_BLOQUEADAS,
    y prefijo contra RAICES_BLOQUEADAS.
    """

    simple = quitar_acentos(palabra).lower()

    if simple.startswith(RAICES_BLOQUEADAS):
        return False

    return not (
        _formas_familia(palabra)
        & PALABRAS_BLOQUEADAS
    )


# ============================================================
# MOTOR DEL JUEGO
# ============================================================

# Cuántos vecinos guardamos para pistas y para la pantalla educativa final.
VECINOS_GUARDADOS = 400

# El vocabulario conserva el orden por frecuencia del corpus, así que un índice
# bajo significa "palabra común". Las pistas y los vecinos que se le muestran a
# un niño salen de aquí: sin este filtro aparecían cosas como "fulgurantes" o
# "rutilante", que no ayudan a nadie.
LIMITE_PALABRA_COMUN = 25_000


class GameEngine:
    def __init__(
        self,
        palabra_secreta: str,
    ):
        asegurar_cache()

        palabra_secreta = (
            palabra_secreta.strip().lower()
        )

        indice = DICT_VOCAB.get(
            palabra_secreta
        )

        if indice is None:
            raise ValueError(
                "La palabra secreta no existe "
                "en el vocabulario."
            )

        self.palabra_secreta = palabra_secreta
        self.indice_secreto = indice

        self.vector_secreto = VECTORES[indice]

        self.similitudes = torch.matmul(
            VECTORES,
            self.vector_secreto,
        )

        orden = torch.argsort(
            self.similitudes,
            descending=True,
        )

        # rangos[i] = puesto de la palabra i.
        # Un tensor int32 de 95.000 posiciones ocupa 380 KB. La versión
        # anterior construía una lista y un diccionario de Python de 2.000.000
        # de entradas por ronda: ~280 MB y 4 segundos bloqueando el servidor.
        self.rangos = torch.empty(
            TOTAL_PALABRAS,
            dtype=torch.int32,
        )

        self.rangos[orden] = torch.arange(
            1,
            TOTAL_PALABRAS + 1,
            dtype=torch.int32,
        )

        self.vecinos = orden[
            :VECINOS_GUARDADOS
        ].clone()

        self.ejes = self._calcular_ejes()

    def obtener_puesto(
        self,
        indice: int,
    ) -> int:
        return int(self.rangos[indice])

    def _comparte_raiz(
        self,
        palabra: str,
    ) -> bool:
        """
        Evita que una pista automática regale la respuesta con un plural o un
        diminutivo ("caballos", "caballito").
        """

        a = quitar_acentos(palabra)
        b = quitar_acentos(self.palabra_secreta)

        if a in b or b in a:
            return True

        minimo = min(len(a), len(b))

        prefijo = min(5, minimo)

        return (
            prefijo >= 4
            and a[:prefijo] == b[:prefijo]
        )

    def obtener_vecinos(
        self,
        cantidad: int = 12,
        omitir_familia: bool = True,
    ) -> list[dict]:
        """
        Palabras que el modelo considera más parecidas a la secreta.

        Es la pantalla educativa del final: deja ver que el modelo agrupa
        "caballo" con "jinete", "potro" y "galope" porque aparecen en textos
        parecidos.
        """

        comunes = []
        raras = []
        vistas = set()

        for indice in self.vecinos.tolist():
            if indice == self.indice_secreto:
                continue

            palabra = PALABRAS[indice]

            if (
                omitir_familia
                and self._comparte_raiz(palabra)
            ):
                continue

            if not es_apropiada(palabra):
                continue

            # "autobús"/"autobus" y "cráter"/"cráteres" son el mismo concepto:
            # mostrar las dos formas confunde y gasta espacio.
            formas = _formas_familia(palabra)

            if formas & vistas:
                continue

            vistas |= formas

            vecino = {
                "palabra": palabra,
                "similitud": round(
                    float(
                        self.similitudes[indice]
                    ),
                    3,
                ),
                "puesto": self.obtener_puesto(
                    indice
                ),
            }

            if indice < LIMITE_PALABRA_COMUN:
                comunes.append(vecino)
            else:
                raras.append(vecino)

        # Las palabras raras solo entran si no hay suficientes conocidas.
        return (comunes + raras)[:cantidad]

    def evaluar(
        self,
        palabra: str,
    ) -> dict:
        texto = palabra.strip().lower()

        if not texto:
            raise ValueError(
                "Debe escribir una palabra."
            )

        indice = buscar_indice(texto)

        # Un niño va a probar groserías: se responde igual que con una palabra
        # inexistente, sin señalarla ni repetirla en pantalla.
        if indice is not None and not es_apropiada(
            PALABRAS[indice]
        ):
            indice = None

        if indice is None or not es_apropiada(texto):
            raise ValueError(
                f"La palabra '{texto}' no está "
                "en el diccionario del juego. "
                "Prueba con otra."
            )

        palabra_real = PALABRAS[indice]

        puesto = self.obtener_puesto(indice)

        similitud = float(
            self.similitudes[indice]
        )

        es_correcta = (
            indice == self.indice_secreto
        )

        posicion = self.calcular_mapa(
            [indice]
        )[0]

        return {
            "palabra": palabra_real,
            "indice": indice,
            "x": posicion["x"],
            "y": posicion["y"],
            "similitud": round(similitud, 4),
            "puesto": puesto,
            "temperatura": calcular_temperatura(
                puesto
            ),
            "descripcion": describir_temperatura(
                puesto
            ),
            "es_correcta": es_correcta,
            "color": obtener_color_temperatura(
                puesto
            ),
            "total_palabras": TOTAL_PALABRAS,
        }

    # --------------------------------------------------------
    # MAPA SEMÁNTICO
    # --------------------------------------------------------

    def calcular_mapa(
        self,
        indices: list[int],
    ) -> list[dict]:
        """
        Proyecta los intentos a 2D de forma que el dibujo signifique algo.

        - El radio depende del puesto: cerca del centro = cerca de la secreta.
        - El ángulo sale de una proyección real de los embeddings, así que dos
          palabras parecidas entre sí (aunque ambas estén lejos de la secreta)
          caen en la misma zona del mapa.

        La versión anterior sorteaba el ángulo al azar, lo que hacía que
        "perro" y "gato" pudieran aparecer en lados opuestos del círculo.
        """

        if not indices:
            return []

        seleccion = torch.tensor(
            indices,
            dtype=torch.long,
        )

        residuos = self._residuos(seleccion)

        coordenadas = torch.matmul(
            residuos,
            self.ejes.T,
        )

        resultado = []

        for posicion, indice in enumerate(indices):
            puesto = self.obtener_puesto(indice)

            radio = self._radio_desde_puesto(
                puesto
            )

            x = float(coordenadas[posicion, 0])
            y = float(coordenadas[posicion, 1])

            magnitud = math.hypot(x, y)

            if magnitud < 1e-6:
                # Palabra alineada con la secreta: el ángulo real no existe,
                # así que usamos uno estable derivado de la propia palabra.
                angulo = (
                    (indice % 360)
                    * math.pi
                    / 180.0
                )

                x = math.cos(angulo)
                y = math.sin(angulo)
            else:
                x /= magnitud
                y /= magnitud

            resultado.append(
                {
                    "x": round(x * radio, 4),
                    "y": round(y * radio, 4),
                }
            )

        return resultado

    def _radio_desde_puesto(
        self,
        puesto: int,
    ) -> float:
        if puesto <= 1:
            return 0.0

        radio = (
            math.log(puesto) / LOG_TOTAL
        )

        # Un mínimo para que no se encime con el marcador del centro.
        return max(0.12, min(1.0, radio))

    def _residuos(
        self,
        seleccion: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quita de cada vector la componente que apunta hacia la palabra secreta.

        Esa información ya la transmite el radio del mapa. Lo que queda es "en
        qué dirección se equivocó" cada intento, que es lo que hace que dos
        palabras parecidas entre sí caigan en la misma zona.
        """

        vectores = VECTORES[seleccion]

        proyeccion = torch.matmul(
            vectores,
            self.vector_secreto,
        )

        return vectores - torch.outer(
            proyeccion,
            self.vector_secreto,
        )

    def _calcular_ejes(self) -> torch.Tensor:
        """
        Fija los dos ejes del mapa una sola vez por ronda.

        Se calculan sobre un conjunto de referencia constante (las palabras
        jugables más los vecinos de la secreta), nunca sobre los intentos del
        jugador. Si dependieran de los intentos, cada palabra nueva rotaría
        todo el dibujo y los puntos anteriores saltarían de lugar.
        """

        referencia = set(
            self.vecinos.tolist()
        )

        for palabra in _palabras_jugables_configuradas():
            indice = DICT_VOCAB.get(palabra)

            if indice is not None:
                referencia.add(indice)

        referencia.discard(
            self.indice_secreto
        )

        seleccion = torch.tensor(
            sorted(referencia),
            dtype=torch.long,
        )

        residuos = self._residuos(seleccion)

        dimensiones = residuos.shape[1]

        ejes = torch.zeros(2, dimensiones)

        try:
            _, _, componentes = torch.linalg.svd(
                residuos,
                full_matrices=False,
            )
        except Exception:
            componentes = torch.eye(
                min(2, dimensiones),
                dimensiones,
            )

        disponibles = min(
            2,
            componentes.shape[0],
        )

        ejes[:disponibles] = componentes[
            :disponibles
        ]

        for fila in range(2):
            eje = ejes[fila]

            if eje.abs().sum() == 0:
                continue

            dominante = int(eje.abs().argmax())

            if eje[dominante] < 0:
                ejes[fila] = -eje

        return ejes

    # --------------------------------------------------------
    # PISTAS
    # --------------------------------------------------------

    def _pistas_configuradas(self) -> list[str]:
        for items in CATEGORIAS.values():
            for item in items:
                if (
                    item["palabra"]
                    == self.palabra_secreta
                ):
                    return item.get("pistas", [])

        return []

    def obtener_pista(
        self,
        numero_pista: int,
    ) -> str:
        """
        Pistas 1 y 2: una palabra relacionada, escrita a mano, y su puesto.
        Pista 3: la inicial y el largo.

        Nunca se saca una pista del corpus. Antes sí se hacía y acabó
        ofreciéndole a un niño la palabra vecina número 41, que resultó ser
        vocabulario adulto.
        """

        pistas = self._pistas_configuradas()

        if numero_pista in (1, 2) and len(pistas) >= numero_pista:
            texto = pistas[numero_pista - 1]

            indice = buscar_indice(texto)

            if indice is not None:
                puesto = self.obtener_puesto(indice)

                return (
                    f"Pista {numero_pista}: "
                    f"la palabra '{texto}' está "
                    f"en el puesto #{puesto}."
                )

            return f"Pista {numero_pista}: {texto}"

        letra = self.palabra_secreta[0].upper()
        letras = len(self.palabra_secreta)

        if numero_pista == 1:
            return f"Pista 1: tiene {letras} letras."

        if numero_pista == 2:
            return f"Pista 2: empieza por '{letra}'."

        return (
            f"Pista 3: empieza por '{letra}' "
            f"y tiene {letras} letras."
        )


# ============================================================
# LABORATORIO DE PALABRAS
# ============================================================
#
# Funciones sueltas para la página educativa. No las usa la partida: si algo
# de aquí falla, el juego sigue intacto.

# Los resultados que se le enseñan a un niño salen de la zona frecuente del
# vocabulario. Más allá aparecen tecnicismos y erratas que no explican nada.
LIMITE_RESULTADO_LABORATORIO = 40_000

# Letras iniciales que comparten una palabra y sus derivados. Con 5 se agrupan
# "volcán"/"volcánico" pero siguen separadas "reina"/"reino".
LARGO_RAIZ = 5


def _indices_de(palabras: list[str]) -> list[int]:
    indices = []

    for palabra in palabras:
        indice = buscar_indice(palabra)

        if indice is None:
            raise ValueError(
                f"La palabra '{palabra}' no está "
                "en el diccionario del juego."
            )

        if not es_apropiada(PALABRAS[indice]):
            raise ValueError(
                f"La palabra '{palabra}' no está "
                "en el diccionario del juego."
            )

        indices.append(indice)

    return indices


def _mejores_resultados(
    direccion: torch.Tensor,
    excluir: list[int],
    cantidad: int,
) -> list[dict]:
    """
    Palabras más cercanas a una dirección del espacio, ya filtradas.
    """

    direccion = direccion / direccion.norm().clamp_min(1e-8)

    similitudes = torch.matmul(VECTORES, direccion)

    # Se piden de más porque el filtrado descarta bastantes.
    cuantas = min(
        len(PALABRAS),
        (cantidad + len(excluir) + 1) * 40,
    )

    candidatos = torch.topk(
        similitudes,
        cuantas,
    ).indices.tolist()

    familias_excluidas = set()
    raices_excluidas = set()

    for indice in excluir:
        original = PALABRAS[indice]

        familias_excluidas |= _formas_familia(
            original
        )

        simple = quitar_acentos(original)

        if len(simple) >= LARGO_RAIZ:
            raices_excluidas.add(
                simple[:LARGO_RAIZ]
            )

    resultado = []
    vistas = set()

    for indice in candidatos:
        if indice in excluir:
            continue

        if indice >= LIMITE_RESULTADO_LABORATORIO:
            continue

        palabra = PALABRAS[indice]

        if not es_apropiada(palabra):
            continue

        formas = _formas_familia(palabra)

        # Ni una variante de las palabras que entraron, ni repetir concepto.
        if formas & familias_excluidas:
            continue

        # Un derivado de la propia palabra no enseña nada: al explorar
        # "volcán" sobran "volcánico" y "volcánica".
        simple = quitar_acentos(palabra)

        if (
            len(simple) >= LARGO_RAIZ
            and simple[:LARGO_RAIZ] in raices_excluidas
        ):
            continue

        if formas & vistas:
            continue

        vistas |= formas

        resultado.append(
            {
                "palabra": palabra,
                "similitud": round(
                    float(similitudes[indice]),
                    3,
                ),
            }
        )

        if len(resultado) >= cantidad:
            break

    return resultado


def resolver_analogia(
    positiva_a: str,
    negativa: str,
    positiva_b: str,
    cantidad: int = 5,
) -> list[dict]:
    """
    Calcula  A - B + C  y devuelve las palabras más cercanas al resultado.

    Es la demostración clásica: "rey - hombre + mujer" cae junto a "reina",
    porque el modelo aprendió que la diferencia entre ambas parejas apunta en
    la misma dirección.
    """

    indices = _indices_de(
        [positiva_a, negativa, positiva_b]
    )

    direccion = (
        VECTORES[indices[0]]
        - VECTORES[indices[1]]
        + VECTORES[indices[2]]
    )

    return _mejores_resultados(
        direccion,
        excluir=indices,
        cantidad=cantidad,
    )


def vecinos_de(
    palabra: str,
    cantidad: int = 12,
) -> list[dict]:
    """
    Palabras que el modelo considera más parecidas a una dada.
    """

    indices = _indices_de([palabra])

    return _mejores_resultados(
        VECTORES[indices[0]],
        excluir=indices,
        cantidad=cantidad,
    )


def encontrar_intruso(
    palabras: list[str],
) -> dict:
    """
    Señala cuál de las palabras encaja menos con el grupo.

    Se compara cada una con el centro del grupo: la que queda más lejos es la
    intrusa.
    """

    if len(palabras) < 3:
        raise ValueError(
            "Hacen falta al menos 3 palabras."
        )

    indices = _indices_de(palabras)

    seleccion = torch.tensor(
        indices,
        dtype=torch.long,
    )

    vectores = VECTORES[seleccion]

    centro = vectores.mean(dim=0)
    centro = centro / centro.norm().clamp_min(1e-8)

    parecidos = torch.matmul(vectores, centro)

    posicion_intruso = int(parecidos.argmin())

    return {
        "intruso": palabras[posicion_intruso],
        "posicion": posicion_intruso,
        "parecidos": [
            {
                "palabra": palabras[i],
                "encaje": round(
                    float(parecidos[i]),
                    3,
                ),
            }
            for i in range(len(palabras))
        ],
    }


def generar_ronda_intruso() -> dict:
    """
    Arma un grupo con tres palabras de una categoría y una de otra.

    Las palabras salen de CATEGORIAS, que está revisada a mano, así que nunca
    puede aparecer vocabulario inapropiado.
    """

    categorias = obtener_categorias()

    nombres = [
        nombre
        for nombre, palabras in categorias.items()
        if len(palabras) >= 3
    ]

    familia, ajena = random.sample(nombres, 2)

    elegidas = random.sample(
        categorias[familia],
        3,
    )

    intrusa = random.choice(
        categorias[ajena]
    )

    opciones = elegidas + [intrusa]
    random.shuffle(opciones)

    return {
        "palabras": opciones,
        "intruso": intrusa,
        "categoria_grupo": familia,
        "categoria_intruso": ajena,
    }
