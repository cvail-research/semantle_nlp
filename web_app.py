import math
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from game_engine import (
    GameEngine,
    asegurar_cache,
    elegir_palabra_al_azar,
    encontrar_intruso,
    generar_ronda_intruso,
    obtener_categorias,
    resolver_analogia,
    validar_palabra_jugable,
    vecinos_de,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="Semantle Español Multijugador"
)


@app.on_event("startup")
def cargar_modelo_al_iniciar():
    asegurar_cache()


app.mount(
    "/static",
    StaticFiles(
        directory=str(STATIC_DIR)
    ),
    name="static",
)


templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


def ajuste(nombre: str, valor_por_defecto: int) -> int:
    """
    Lee un número del .env para poder cambiar el ritmo del juego en el sitio,
    sin tocar el código ni reinstalar nada.
    """

    texto = os.getenv(nombre)

    if not texto:
        return valor_por_defecto

    try:
        return max(1, int(texto))
    except ValueError:
        print(
            f"Aviso: {nombre}='{texto}' no es un número. "
            f"Se usa {valor_por_defecto}.",
            flush=True,
        )
        return valor_por_defecto


# Un curso entero debe caber en una sola sala. Medido en una máquina de 6,7 GB:
# 40 jugadores a la vez responden en ~3 ms y 80 en ~350 ms (percentil 95).
MAX_JUGADORES = ajuste("MAX_JUGADORES", 30)

MAX_INTENTOS = ajuste("MAX_INTENTOS", 5)
MAX_PISTAS = ajuste("MAX_PISTAS", 3)

TIEMPO_PARTIDA_SEGUNDOS = ajuste("SEGUNDOS_PARTIDA", 120)
CUENTA_REGRESIVA_SEGUNDOS = ajuste("SEGUNDOS_CUENTA_REGRESIVA", 3)

# Una sala se borra sola si nadie la toca durante este tiempo. Evita que las
# partidas abandonadas se acumulen en memoria.
INACTIVIDAD_MAXIMA_SEGUNDOS = 3600

ADMIN_PASSWORD = os.getenv(
    "ADMIN_PASSWORD",
)

ADMIN_TOKEN = os.getenv(
    "ADMIN_TOKEN"
)

if not ADMIN_PASSWORD:
    raise RuntimeError(
        "Falta configurar ADMIN_PASSWORD."
    )

if not ADMIN_TOKEN:
    raise RuntimeError(
        "Falta configurar ADMIN_TOKEN."
    )


# Registro de salas activas. Antes existía una única partida global, así que
# crear una sala nueva destruía la que estuviera en curso y era imposible jugar
# solo y acompañado al mismo tiempo.
SALAS: dict[str, "Partida"] = {}

# jugador_id -> código de sala.
JUGADORES: dict[str, str] = {}

# Protege SALAS y JUGADORES. Cada sala además tiene su propio candado, para que
# una partida no bloquee a las demás.
bloqueo_registro = threading.Lock()


class CrearPartidaRequest(BaseModel):
    categoria: str
    palabra: str


class UnirseRequest(BaseModel):
    codigo: str
    nombre: str


class IntentoRequest(BaseModel):
    jugador_id: str
    palabra: str


class PistaRequest(BaseModel):
    jugador_id: str


class AnalogiaRequest(BaseModel):
    a: str
    b: str
    c: str


class PalabraRequest(BaseModel):
    palabra: str


class IntrusoRequest(BaseModel):
    palabras: list[str]


class SoloRequest(BaseModel):
    nombre: str = "Jugador"


class JugadorRequest(BaseModel):
    jugador_id: str


class Partida:
    def __init__(
        self,
        categoria: str,
        palabra: str,
        modo: str = "sala",
    ):
        self.codigo = self.generar_codigo()
        self.modo = modo

        self.categoria = ""
        self.engine = None

        self.estado = "ESPERANDO_JUGADORES"

        self.jugadores = {}

        self.numero_ronda = 1
        self.historial_rondas = []
        self.estadisticas_acumuladas = {}

        self.ronda_actual_guardada = False

        self.cuenta_regresiva_hasta = None
        self.inicio_juego = None
        self.fin_juego = None

        self.ultima_actividad = time.monotonic()

        # Evita repetir palabra entre rondas de la misma sala.
        self.palabras_usadas = set()

        self.bloqueo = threading.RLock()

        self.configurar_ronda(
            categoria=categoria,
            palabra=palabra,
            reiniciar_jugadores=False,
        )

    @staticmethod
    def generar_codigo() -> str:
        return secrets.token_hex(3).upper()

    @property
    def max_jugadores(self) -> int:
        return (
            1
            if self.modo == "solo"
            else MAX_JUGADORES
        )

    def tocar(self):
        self.ultima_actividad = time.monotonic()

    def configurar_ronda(
        self,
        categoria: str,
        palabra: str,
        reiniciar_jugadores: bool = True,
    ):
        categoria = categoria.strip()
        palabra = palabra.strip().lower()

        if not categoria:
            raise ValueError(
                "Debe seleccionar una categoría."
            )

        if not palabra:
            raise ValueError(
                "Debe seleccionar una palabra."
            )

        if not validar_palabra_jugable(
            palabra,
            categoria,
        ):
            raise ValueError(
                "La palabra no pertenece a la "
                "categoría seleccionada."
            )

        self.categoria = categoria

        self.engine = GameEngine(
            palabra_secreta=palabra
        )

        self.palabras_usadas.add(palabra)

        self.estado = "ESPERANDO_JUGADORES"

        self.cuenta_regresiva_hasta = None
        self.inicio_juego = None
        self.fin_juego = None

        self.tocar()

        if reiniciar_jugadores:
            for jugador in self.jugadores.values():
                self.reiniciar_jugador(jugador)

    @staticmethod
    def reiniciar_jugador(jugador: dict):
        jugador["intentos"] = []
        jugador["pistas_usadas"] = 0
        jugador["pistas"] = []
        jugador["gano"] = False
        jugador["terminado"] = False
        jugador["gano_en"] = None

    def reiniciar_ronda(
        self,
        categoria: str,
        palabra: str,
    ):
        self.actualizar_estado()

        if self.estado not in (
            "ESPERANDO_JUGADORES",
            "FINALIZADA",
        ):
            raise ValueError(
                "No puedes cambiar la palabra mientras "
                "la partida está en curso."
            )

        if self.estado == "FINALIZADA":
            self.guardar_resultado_ronda()
            self.numero_ronda += 1

        self.configurar_ronda(
            categoria=categoria,
            palabra=palabra,
            reiniciar_jugadores=True,
        )

        self.ronda_actual_guardada = False

    def actualizar_estado(self):
        ahora = time.monotonic()

        if (
            self.estado == "INICIANDO"
            and self.cuenta_regresiva_hasta is not None
            and ahora >= self.cuenta_regresiva_hasta
        ):
            self.estado = "EN_CURSO"

            self.inicio_juego = ahora

            self.fin_juego = (
                ahora + TIEMPO_PARTIDA_SEGUNDOS
            )

        if (
            self.estado == "EN_CURSO"
            and self.fin_juego is not None
            and ahora >= self.fin_juego
        ):
            self.finalizar()

        # Si todos terminaron no tiene sentido esperar al reloj.
        if (
            self.estado == "EN_CURSO"
            and self.jugadores
            and all(
                jugador["terminado"]
                for jugador in self.jugadores.values()
            )
        ):
            self.finalizar()

    def agregar_jugador(
        self,
        nombre: str,
    ) -> dict:
        self.actualizar_estado()

        nombre = nombre.strip()

        if not nombre:
            raise ValueError(
                "El nombre es obligatorio."
            )

        if len(nombre) > 30:
            raise ValueError(
                "El nombre no puede superar "
                "los 30 caracteres."
            )

        if self.estado != "ESPERANDO_JUGADORES":
            raise ValueError(
                "La sala ya empezó. Pídele al "
                "profesor que cree una ronda nueva."
            )

        if len(self.jugadores) >= self.max_jugadores:
            raise ValueError(
                "La sala ya está llena "
                f"({self.max_jugadores} jugadores)."
            )

        nombres = [
            jugador["nombre"].lower()
            for jugador in self.jugadores.values()
        ]

        if nombre.lower() in nombres:
            raise ValueError(
                "Ya existe un jugador con ese nombre."
            )

        jugador_id = str(uuid.uuid4())

        jugador = {
            "id": jugador_id,
            "nombre": nombre,
        }

        self.reiniciar_jugador(jugador)

        self.jugadores[jugador_id] = jugador

        self.tocar()

        return jugador

    def iniciar(self):
        self.actualizar_estado()

        if self.estado != "ESPERANDO_JUGADORES":
            raise ValueError(
                "La sala no está esperando para comenzar."
            )

        if not self.jugadores:
            raise ValueError(
                "Debe haber al menos un jugador."
            )

        self.estado = "INICIANDO"

        self.cuenta_regresiva_hasta = (
            time.monotonic()
            + CUENTA_REGRESIVA_SEGUNDOS
        )

        self.inicio_juego = None
        self.fin_juego = None

        self.tocar()

    def finalizar(self):
        if self.estado == "FINALIZADA":
            return

        self.estado = "FINALIZADA"

        for jugador in self.jugadores.values():
            jugador["terminado"] = True

        self.guardar_resultado_ronda()

        self.tocar()

    def obtener_jugador(
        self,
        jugador_id: str,
    ) -> dict:
        jugador = self.jugadores.get(jugador_id)

        if jugador is None:
            raise ValueError(
                "Jugador no encontrado."
            )

        return jugador

    def segundos_restantes(self) -> int:
        ahora = time.monotonic()

        if self.estado == "INICIANDO":
            if self.cuenta_regresiva_hasta is None:
                return 0

            return max(
                0,
                math.ceil(
                    self.cuenta_regresiva_hasta - ahora
                ),
            )

        if self.estado == "EN_CURSO":
            if self.fin_juego is None:
                return 0

            return max(
                0,
                math.ceil(
                    self.fin_juego - ahora
                ),
            )

        return 0

    def registrar_intento(
        self,
        jugador_id: str,
        palabra: str,
    ) -> dict:
        self.actualizar_estado()

        if self.estado != "EN_CURSO":
            raise ValueError(
                "La partida todavía no está en curso."
            )

        jugador = self.obtener_jugador(
            jugador_id
        )

        if jugador["gano"]:
            raise ValueError(
                "¡Ya ganaste! Espera a que termine "
                "la ronda."
            )

        if jugador["terminado"]:
            raise ValueError(
                "Ya terminaste tu participación."
            )

        if len(jugador["intentos"]) >= MAX_INTENTOS:
            jugador["terminado"] = True

            raise ValueError(
                "Ya no tienes más intentos."
            )

        resultado = self.engine.evaluar(palabra)

        repetida = any(
            intento["palabra"]
            == resultado["palabra"]
            for intento in jugador["intentos"]
        )

        if repetida:
            raise ValueError(
                f"Ya probaste '{resultado['palabra']}'. "
                "Prueba con otra palabra."
            )

        resultado["numero_intento"] = (
            len(jugador["intentos"]) + 1
        )

        jugador["intentos"].append(resultado)

        if resultado["es_correcta"]:
            jugador["gano"] = True
            jugador["terminado"] = True

            if self.inicio_juego is not None:
                jugador["gano_en"] = (
                    time.monotonic()
                    - self.inicio_juego
                )

        elif len(jugador["intentos"]) >= MAX_INTENTOS:
            jugador["terminado"] = True

        self.tocar()

        self.actualizar_estado()

        return resultado

    def obtener_pista(
        self,
        jugador_id: str,
    ) -> str:
        self.actualizar_estado()

        if self.estado != "EN_CURSO":
            raise ValueError(
                "La partida todavía no está en curso."
            )

        jugador = self.obtener_jugador(
            jugador_id
        )

        if jugador["gano"]:
            raise ValueError(
                "Ya ganaste. Espera a que termine "
                "la ronda."
            )

        if jugador["terminado"]:
            raise ValueError(
                "Ya terminaste tu participación."
            )

        if jugador["pistas_usadas"] >= MAX_PISTAS:
            raise ValueError(
                "Ya usaste tus "
                f"{MAX_PISTAS} pistas."
            )

        jugador["pistas_usadas"] += 1

        pista = self.engine.obtener_pista(
            jugador["pistas_usadas"]
        )

        jugador["pistas"].append(pista)

        self.tocar()

        return pista

    def obtener_podio(self) -> list[dict]:
        jugadores = list(
            self.jugadores.values()
        )

        def criterio(jugador):
            tiempo = jugador["gano_en"]

            if tiempo is None:
                tiempo = float("inf")

            return (
                not jugador["gano"],
                len(jugador["intentos"]),
                tiempo,
                jugador["nombre"].lower(),
            )

        jugadores.sort(key=criterio)

        podio = []

        for posicion, jugador in enumerate(
            jugadores[:3],
            start=1,
        ):
            tiempo = jugador["gano_en"]

            podio.append(
                {
                    "posicion": posicion,
                    "jugador_id": jugador["id"],
                    "nombre": jugador["nombre"],
                    "intentos": len(
                        jugador["intentos"]
                    ),
                    "gano": jugador["gano"],
                    "tiempo_segundos": (
                        round(tiempo, 2)
                        if tiempo is not None
                        else None
                    ),
                }
            )

        return podio

    def guardar_resultado_ronda(self):
        if self.ronda_actual_guardada:
            return

        podio = self.obtener_podio()

        self.historial_rondas.append(
            {
                "ronda": self.numero_ronda,
                "categoria": self.categoria,
                "palabra_secreta": (
                    self.engine.palabra_secreta
                ),
                "podio": podio,
            }
        )

        puntos = {1: 3, 2: 2, 3: 1}

        for jugador_id, jugador in (
            self.jugadores.items()
        ):
            if jugador_id not in (
                self.estadisticas_acumuladas
            ):
                self.estadisticas_acumuladas[
                    jugador_id
                ] = {
                    "jugador_id": jugador_id,
                    "nombre": jugador["nombre"],
                    "puntos": 0,
                    "victorias": 0,
                    "podios": 0,
                    "rondas_jugadas": 0,
                    "intentos_totales": 0,
                }

            estadistica = (
                self.estadisticas_acumuladas[
                    jugador_id
                ]
            )

            estadistica["nombre"] = (
                jugador["nombre"]
            )

            estadistica["rondas_jugadas"] += 1

            estadistica["intentos_totales"] += len(
                jugador["intentos"]
            )

        for jugador in podio:
            jugador_id = jugador["jugador_id"]

            estadistica = (
                self.estadisticas_acumuladas[
                    jugador_id
                ]
            )

            posicion = jugador["posicion"]

            estadistica["puntos"] += puntos.get(
                posicion,
                0,
            )

            estadistica["podios"] += 1

            if posicion == 1 and jugador["gano"]:
                estadistica["victorias"] += 1

        self.ronda_actual_guardada = True

    def obtener_ranking_acumulado(self) -> list[dict]:
        ranking = list(
            self.estadisticas_acumuladas.values()
        )

        ranking.sort(
            key=lambda jugador: (
                -jugador["puntos"],
                -jugador["victorias"],
                -jugador["podios"],
                jugador["intentos_totales"],
                jugador["nombre"].lower(),
            )
        )

        return [
            {"posicion": posicion, **jugador}
            for posicion, jugador in enumerate(
                ranking,
                start=1,
            )
        ]

    def estado_admin(self) -> dict:
        self.actualizar_estado()

        respuesta = {
            "activa": True,
            "codigo": self.codigo,
            "modo": self.modo,
            "estado": self.estado,
            "numero_ronda": self.numero_ronda,
            "categoria": self.categoria,
            "palabra_secreta": (
                self.engine.palabra_secreta
            ),
            "total_jugadores": len(
                self.jugadores
            ),
            "max_jugadores": self.max_jugadores,
            "segundos_restantes": (
                self.segundos_restantes()
            ),
            "cuenta_regresiva": (
                self.segundos_restantes()
                if self.estado == "INICIANDO"
                else 0
            ),
            "jugadores": [
                {
                    "id": jugador["id"],
                    "nombre": jugador["nombre"],
                    "intentos": len(
                        jugador["intentos"]
                    ),
                    "pistas_usadas": jugador[
                        "pistas_usadas"
                    ],
                    "gano": jugador["gano"],
                    "terminado": jugador[
                        "terminado"
                    ],
                }
                for jugador in self.jugadores.values()
            ],
            "historial_rondas": (
                self.historial_rondas
            ),
            "ranking_acumulado": (
                self.obtener_ranking_acumulado()
            ),
        }

        if self.estado == "FINALIZADA":
            respuesta["podio"] = (
                self.obtener_podio()
            )

        return respuesta

    def estado_jugador(
        self,
        jugador_id: str,
    ) -> dict:
        self.actualizar_estado()

        jugador = self.obtener_jugador(
            jugador_id
        )

        respuesta = {
            "codigo": self.codigo,
            "modo": self.modo,
            "estado": self.estado,
            "numero_ronda": self.numero_ronda,
            "categoria": self.categoria,
            "nombre": jugador["nombre"],
            "jugadores_conectados": len(
                self.jugadores
            ),
            "max_jugadores": self.max_jugadores,
            "intentos_usados": len(
                jugador["intentos"]
            ),
            "intentos_restantes": max(
                0,
                MAX_INTENTOS
                - len(jugador["intentos"]),
            ),
            "pistas_usadas": jugador[
                "pistas_usadas"
            ],
            "pistas_restantes": max(
                0,
                MAX_PISTAS
                - jugador["pistas_usadas"],
            ),
            "pistas": jugador["pistas"],
            "gano": jugador["gano"],
            "terminado": jugador["terminado"],
            "intentos": jugador["intentos"],
            "segundos_restantes": (
                self.segundos_restantes()
            ),
            "cuenta_regresiva": (
                self.segundos_restantes()
                if self.estado == "INICIANDO"
                else 0
            ),
        }

        if self.estado == "FINALIZADA":
            respuesta["palabra_secreta"] = (
                self.engine.palabra_secreta
            )

            respuesta["podio"] = (
                self.obtener_podio()
            )

            # Pantalla educativa: qué palabras considera parecidas el modelo.
            respuesta["vecinos"] = (
                self.engine.obtener_vecinos(12)
            )

        return respuesta


# ============================================================
# REGISTRO DE SALAS
# ============================================================

def limpiar_salas_vencidas():
    ahora = time.monotonic()

    with bloqueo_registro:
        vencidas = [
            codigo
            for codigo, sala in SALAS.items()
            if ahora - sala.ultima_actividad
            > INACTIVIDAD_MAXIMA_SEGUNDOS
        ]

        for codigo in vencidas:
            sala = SALAS.pop(codigo, None)

            if sala is None:
                continue

            for jugador_id in sala.jugadores:
                JUGADORES.pop(jugador_id, None)


def registrar_sala(sala: "Partida"):
    with bloqueo_registro:
        SALAS[sala.codigo] = sala


def registrar_jugador(
    jugador_id: str,
    codigo: str,
):
    with bloqueo_registro:
        JUGADORES[jugador_id] = codigo


def obtener_sala_por_codigo(
    codigo: str,
) -> "Partida":
    with bloqueo_registro:
        sala = SALAS.get(codigo)

    if sala is None:
        raise HTTPException(
            status_code=404,
            detail="No existe una sala con ese código.",
        )

    return sala


def obtener_sala_de_jugador(
    jugador_id: str,
) -> "Partida":
    with bloqueo_registro:
        codigo = JUGADORES.get(jugador_id)

        sala = (
            SALAS.get(codigo)
            if codigo
            else None
        )

    if sala is None:
        raise HTTPException(
            status_code=404,
            detail="Tu partida ya no está disponible.",
        )

    return sala


def sala_administrada() -> "Partida":
    """
    Sala que controla el panel de administración.

    Es la última sala en modo "sala" que se creó. Las partidas en solitario no
    aparecen acá: cada niño tiene la suya y no las maneja el profesor.
    """

    with bloqueo_registro:
        salas = [
            sala
            for sala in SALAS.values()
            if sala.modo == "sala"
        ]

    if not salas:
        raise HTTPException(
            status_code=404,
            detail="No existe una sala activa.",
        )

    return max(
        salas,
        key=lambda sala: sala.ultima_actividad,
    )


def verificar_admin(
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Sesión de administrador inválida.",
        )

    return True


def error_400(error: ValueError):
    return HTTPException(
        status_code=400,
        detail=str(error),
    )


# ============================================================
# PÁGINAS
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def pagina_inicio(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="player.html",
    )


@app.get(
    "/jugar",
    response_class=HTMLResponse,
)
def pagina_jugador(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="player.html",
    )


@app.get(
    "/admin",
    response_class=HTMLResponse,
)
def pagina_admin(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
    )


# ============================================================
# MODO SOLITARIO
# ============================================================

@app.post("/api/solo/nueva")
def crear_partida_solo(datos: SoloRequest):
    """
    Crea una partida de un jugador y la arranca al instante.

    No necesita administrador ni código: el niño entra y juega.
    """

    limpiar_salas_vencidas()

    categoria, palabra = elegir_palabra_al_azar()

    sala = Partida(
        categoria=categoria,
        palabra=palabra,
        modo="solo",
    )

    nombre = (
        datos.nombre.strip()
        or "Jugador"
    )

    with sala.bloqueo:
        try:
            jugador = sala.agregar_jugador(
                nombre
            )
            sala.iniciar()
        except ValueError as error:
            raise error_400(error) from error

    registrar_sala(sala)

    registrar_jugador(
        jugador["id"],
        sala.codigo,
    )

    return {
        "codigo": sala.codigo,
        "jugador_id": jugador["id"],
        "nombre": jugador["nombre"],
        "modo": "solo",
    }


@app.post("/api/solo/revancha")
def revancha_solo(datos: JugadorRequest):
    """
    Otra palabra en la misma sala, conservando las estadísticas del jugador.
    """

    sala = obtener_sala_de_jugador(
        datos.jugador_id
    )

    if sala.modo != "solo":
        raise HTTPException(
            status_code=400,
            detail="Esta sala la controla el administrador.",
        )

    with sala.bloqueo:
        sala.actualizar_estado()

        if sala.estado != "FINALIZADA":
            sala.finalizar()

        categoria, palabra = elegir_palabra_al_azar(
            excluir=sala.palabras_usadas
        )

        try:
            sala.reiniciar_ronda(
                categoria=categoria,
                palabra=palabra,
            )
            sala.iniciar()
        except ValueError as error:
            raise error_400(error) from error

        return sala.estado_jugador(
            datos.jugador_id
        )


# ============================================================
# ADMINISTRACIÓN
# ============================================================

@app.post("/api/admin/login")
def iniciar_sesion_admin(password: str):
    if password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Contraseña incorrecta.",
        )

    return {"token": ADMIN_TOKEN}


@app.get("/api/admin/categorias")
def listar_categorias(
    _: bool = Depends(verificar_admin),
):
    return {
        "categorias": obtener_categorias()
    }


@app.post("/api/admin/partida")
def crear_partida(
    datos: CrearPartidaRequest,
    _: bool = Depends(verificar_admin),
):
    limpiar_salas_vencidas()

    try:
        sala = Partida(
            categoria=datos.categoria,
            palabra=datos.palabra,
        )
    except ValueError as error:
        raise error_400(error) from error

    registrar_sala(sala)

    return sala.estado_admin()


@app.post("/api/admin/partida/nueva-ronda")
def crear_nueva_ronda(
    datos: CrearPartidaRequest,
    _: bool = Depends(verificar_admin),
):
    sala = sala_administrada()

    with sala.bloqueo:
        try:
            sala.reiniciar_ronda(
                categoria=datos.categoria,
                palabra=datos.palabra,
            )
        except ValueError as error:
            raise error_400(error) from error

        return sala.estado_admin()


@app.get("/api/admin/partida")
def estado_partida_admin(
    _: bool = Depends(verificar_admin),
):
    try:
        sala = sala_administrada()
    except HTTPException:
        return {
            "activa": False,
            "codigo": "",
            "modo": "sala",
            "estado": "SIN_PARTIDA",
            "numero_ronda": 0,
            "categoria": "",
            "palabra_secreta": "",
            "total_jugadores": 0,
            "max_jugadores": MAX_JUGADORES,
            "segundos_restantes": 0,
            "cuenta_regresiva": 0,
            "jugadores": [],
            "historial_rondas": [],
            "ranking_acumulado": [],
        }

    with sala.bloqueo:
        return sala.estado_admin()


@app.post("/api/admin/partida/iniciar")
def iniciar_partida(
    _: bool = Depends(verificar_admin),
):
    sala = sala_administrada()

    with sala.bloqueo:
        try:
            sala.iniciar()
        except ValueError as error:
            raise error_400(error) from error

        return sala.estado_admin()


@app.post("/api/admin/partida/finalizar")
def finalizar_partida(
    _: bool = Depends(verificar_admin),
):
    sala = sala_administrada()

    with sala.bloqueo:
        sala.finalizar()

        return sala.estado_admin()


# ============================================================
# JUGADORES
# ============================================================

@app.post("/api/partida/unirse")
def unirse_a_partida(datos: UnirseRequest):
    limpiar_salas_vencidas()

    codigo = datos.codigo.strip().upper()

    sala = obtener_sala_por_codigo(codigo)

    if sala.modo == "solo":
        raise HTTPException(
            status_code=400,
            detail="Esa partida es individual.",
        )

    with sala.bloqueo:
        try:
            jugador = sala.agregar_jugador(
                datos.nombre
            )
        except ValueError as error:
            raise error_400(error) from error

    registrar_jugador(
        jugador["id"],
        sala.codigo,
    )

    return {
        "codigo": sala.codigo,
        "jugador_id": jugador["id"],
        "nombre": jugador["nombre"],
        "modo": sala.modo,
    }


@app.get("/api/partida/estado")
def estado_partida_jugador(jugador_id: str):
    sala = obtener_sala_de_jugador(jugador_id)

    with sala.bloqueo:
        try:
            return sala.estado_jugador(
                jugador_id
            )
        except ValueError as error:
            raise HTTPException(
                status_code=404,
                detail=str(error),
            ) from error


@app.post("/api/partida/intento")
def enviar_intento(datos: IntentoRequest):
    sala = obtener_sala_de_jugador(
        datos.jugador_id
    )

    with sala.bloqueo:
        try:
            return sala.registrar_intento(
                datos.jugador_id,
                datos.palabra,
            )
        except ValueError as error:
            raise error_400(error) from error


@app.post("/api/partida/pista")
def solicitar_pista(datos: PistaRequest):
    sala = obtener_sala_de_jugador(
        datos.jugador_id
    )

    with sala.bloqueo:
        try:
            pista = sala.obtener_pista(
                datos.jugador_id
            )
        except ValueError as error:
            raise error_400(error) from error

    return {"pista": pista}


# ============================================================
# LABORATORIO DE PALABRAS
# ============================================================
#
# Página educativa independiente. No comparte estado con las partidas: nada de
# lo que pase aquí puede afectar a una sala en curso.

@app.get(
    "/laboratorio",
    response_class=HTMLResponse,
)
def pagina_laboratorio(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="laboratorio.html",
    )


@app.post("/api/laboratorio/analogia")
def laboratorio_analogia(datos: AnalogiaRequest):
    try:
        resultados = resolver_analogia(
            datos.a,
            datos.b,
            datos.c,
        )
    except ValueError as error:
        raise error_400(error) from error

    if not resultados:
        raise HTTPException(
            status_code=400,
            detail="No se encontró ninguna palabra parecida.",
        )

    return {"resultados": resultados}


@app.post("/api/laboratorio/vecinos")
def laboratorio_vecinos(datos: PalabraRequest):
    try:
        vecinos = vecinos_de(datos.palabra)
    except ValueError as error:
        raise error_400(error) from error

    return {"vecinos": vecinos}


@app.get("/api/laboratorio/intruso")
def laboratorio_intruso_nuevo():
    ronda = generar_ronda_intruso()

    # El intruso correcto no se manda al navegador: si no, bastaría con mirar
    # la respuesta de red para hacer trampa.
    return {"palabras": ronda["palabras"]}


@app.post("/api/laboratorio/intruso")
def laboratorio_intruso_resolver(
    datos: IntrusoRequest,
):
    try:
        return encontrar_intruso(datos.palabras)
    except ValueError as error:
        raise error_400(error) from error
