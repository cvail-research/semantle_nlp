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
    obtener_categorias,
    validar_palabra_jugable,
)
load_dotenv()  # Carga las variables de entorno desde el archivo .env
BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="Semantle Español Multijugador"
)


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


MAX_JUGADORES = 10
MAX_INTENTOS = 5

TIEMPO_PARTIDA_SEGUNDOS = 120
CUENTA_REGRESIVA_SEGUNDOS = 3

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

partida_actual = None
bloqueo = threading.Lock()


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


class Partida:
    def __init__(
        self,
        categoria: str,
        palabra: str,
    ):
        self.codigo = self.generar_codigo()

        self.categoria = ""
        self.engine = None

        self.estado = "ESPERANDO_JUGADORES"

        self.jugadores = {}

        # Número de ronda actual.
        self.numero_ronda = 1

        # Historial de los podios anteriores.
        self.historial_rondas = []

        # Ranking acumulado de todos los jugadores.
        self.estadisticas_acumuladas = {}

        # Evita guardar dos veces una misma ronda.
        self.ronda_actual_guardada = False

        self.cuenta_regresiva_hasta = None
        self.inicio_juego = None
        self.fin_juego = None

        self.configurar_ronda(
            categoria=categoria,
            palabra=palabra,
            reiniciar_jugadores=False,
        )

    @staticmethod
    def generar_codigo() -> str:
        return secrets.token_hex(3).upper()

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

        self.estado = "ESPERANDO_JUGADORES"

        self.cuenta_regresiva_hasta = None
        self.inicio_juego = None
        self.fin_juego = None

        if reiniciar_jugadores:
            for jugador in self.jugadores.values():
                jugador["intentos"] = []
                jugador["pistas_usadas"] = 0
                jugador["gano"] = False
                jugador["terminado"] = False
                jugador["gano_en"] = None

    def reiniciar_ronda(
        self,
        categoria: str,
        palabra: str,
    ):
        """
        Permite cambiar la palabra:

        - Antes de iniciar una ronda.
        - Después de finalizar una ronda.

        No permite cambiarla mientras:
        - Está iniciando.
        - Está en curso.
        """

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
                "La sala no está aceptando jugadores."
            )

        if len(self.jugadores) >= MAX_JUGADORES:
            raise ValueError(
                "La sala ya tiene 10 jugadores."
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
            "intentos": [],
            "pistas_usadas": 0,
            "gano": False,
            "terminado": False,
            "gano_en": None,
        }

        self.jugadores[jugador_id] = jugador

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

    def finalizar(self):
        """
        Finaliza la ronda y guarda sus resultados.
        """

        if self.estado == "FINALIZADA":
            return

        self.estado = "FINALIZADA"

        for jugador in self.jugadores.values():
            jugador["terminado"] = True

        self.guardar_resultado_ronda()

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
        self.actualizar_estado()

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

        resultado = self.engine.evaluar(
            palabra
        )

        resultado["numero_intento"] = (
            len(jugador["intentos"]) + 1
        )

        jugador["intentos"].append(
            resultado
        )

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

        if jugador["pistas_usadas"] >= 3:
            return "No hay más pistas disponibles."

        jugador["pistas_usadas"] += 1

        return self.engine.obtener_pista(
            jugador["pistas_usadas"]
        )

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
        """
        Guarda el resultado de la ronda solamente una vez.
        """

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

        puntos = {
            1: 3,
            2: 2,
            3: 1,
        }

        # Primero registra a todos los jugadores
        # que participaron en la ronda.
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

        # Después agrega puntos según el podio.
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

        resultado = []

        for posicion, jugador in enumerate(
            ranking,
            start=1,
        ):
            resultado.append(
                {
                    "posicion": posicion,
                    **jugador,
                }
            )

        return resultado

    def estado_admin(self) -> dict:
        self.actualizar_estado()

        respuesta = {
            "codigo": self.codigo,
            "estado": self.estado,
            "numero_ronda": self.numero_ronda,
            "categoria": self.categoria,
            "palabra_secreta": (
                self.engine.palabra_secreta
            ),
            "total_jugadores": len(
                self.jugadores
            ),
            "max_jugadores": MAX_JUGADORES,
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
            "estado": self.estado,
            "numero_ronda": self.numero_ronda,
            "categoria": self.categoria,
            "nombre": jugador["nombre"],
            "jugadores_conectados": len(
                self.jugadores
            ),
            "max_jugadores": MAX_JUGADORES,
            "intentos_usados": len(
                jugador["intentos"]
            ),
            "intentos_restantes": max(
                0,
                MAX_INTENTOS - len(
                    jugador["intentos"]
                ),
            ),
            "gano": jugador["gano"],
            "terminado": jugador["terminado"],
            "intentos": jugador["intentos"],
            "pistas_usadas": jugador[
                "pistas_usadas"
            ],
            "segundos_restantes": (
                self.segundos_restantes()
            ),
            "cuenta_regresiva": (
                self.segundos_restantes()
                if self.estado == "INICIANDO"
                else 0
            ),
        }

        if jugador["gano"]:
            respuesta["mensaje"] = (
                "¡Felicitaciones! Encontraste "
                "la palabra secreta."
            )

        elif jugador["terminado"]:
            respuesta["mensaje"] = (
                "Ya terminaste tus intentos."
            )

        if self.estado == "FINALIZADA":
            respuesta["palabra_secreta"] = (
                self.engine.palabra_secreta
            )

            respuesta["podio"] = (
                self.obtener_podio()
            )

        return respuesta


def obtener_partida() -> Partida:
    if partida_actual is None:
        raise HTTPException(
            status_code=404,
            detail="No existe una sala activa.",
        )

    partida_actual.actualizar_estado()

    return partida_actual


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
    "/admin",
    response_class=HTMLResponse,
)
def pagina_admin(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
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


@app.post("/api/admin/login")
def iniciar_sesion_admin(
    password: str,
):
    if password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Contraseña incorrecta.",
        )

    return {
        "token": ADMIN_TOKEN,
    }


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
    global partida_actual

    with bloqueo:
        try:
            partida_actual = Partida(
                categoria=datos.categoria,
                palabra=datos.palabra,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

    return partida_actual.estado_admin()


@app.post("/api/admin/partida/nueva-ronda")
def crear_nueva_ronda(
    datos: CrearPartidaRequest,
    _: bool = Depends(verificar_admin),
):
    partida = obtener_partida()

    with bloqueo:
        try:
            partida.reiniciar_ronda(
                categoria=datos.categoria,
                palabra=datos.palabra,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

    return partida.estado_admin()


@app.get("/api/admin/partida")
def estado_partida_admin(
    _: bool = Depends(verificar_admin),
):
    partida = obtener_partida()

    return partida.estado_admin()


@app.post("/api/admin/partida/iniciar")
def iniciar_partida(
    _: bool = Depends(verificar_admin),
):
    partida = obtener_partida()

    with bloqueo:
        try:
            partida.iniciar()
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

    return partida.estado_admin()


@app.post("/api/admin/partida/finalizar")
def finalizar_partida(
    _: bool = Depends(verificar_admin),
):
    partida = obtener_partida()

    with bloqueo:
        partida.finalizar()

    return partida.estado_admin()


@app.post("/api/partida/unirse")
def unirse_a_partida(
    datos: UnirseRequest,
):
    partida = obtener_partida()

    codigo = datos.codigo.strip().upper()

    if codigo != partida.codigo:
        raise HTTPException(
            status_code=400,
            detail="El código de sala es incorrecto.",
        )

    with bloqueo:
        try:
            jugador = partida.agregar_jugador(
                datos.nombre
            )
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

    return {
        "codigo": partida.codigo,
        "jugador_id": jugador["id"],
        "nombre": jugador["nombre"],
    }


@app.get("/api/partida/estado")
def estado_partida_jugador(
    jugador_id: str,
):
    partida = obtener_partida()

    with bloqueo:
        try:
            return partida.estado_jugador(
                jugador_id
            )
        except ValueError as error:
            raise HTTPException(
                status_code=404,
                detail=str(error),
            ) from error


@app.post("/api/partida/intento")
def enviar_intento(
    datos: IntentoRequest,
):
    partida = obtener_partida()

    with bloqueo:
        try:
            return partida.registrar_intento(
                datos.jugador_id,
                datos.palabra,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error


@app.post("/api/partida/pista")
def solicitar_pista(
    datos: PistaRequest,
):
    partida = obtener_partida()

    with bloqueo:
        try:
            pista = partida.obtener_pista(
                datos.jugador_id
            )
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

    return {
        "pista": pista,
    }