"""
Levanta el juego y muestra TODAS las formas de entrar que estén disponibles.

Pensado para montar en un sitio desconocido: nunca se apaga por falta de
internet. Si el túnel no se puede abrir, lo dice y sigue funcionando en la red
local, que es lo que casi siempre va a hacer falta.

Uso:
    uv run compartir.py
"""

import re
import socket
import subprocess
import sys
import time
import urllib.request

import qrcode

PUERTO = 8000
URL_LOCAL = f"http://localhost:{PUERTO}"
ARCHIVO_LOG_TUNEL = "tunel.log"

PATRON_URL = re.compile(
    r"https://[a-zA-Z0-9-]+\.trycloudflare\.com"
)

SEGUNDOS_ESPERA_SERVIDOR = 300
SEGUNDOS_ESPERA_TUNEL = 25


def linea(caracter="=", ancho=64):
    print(caracter * ancho)


def direcciones_locales() -> list[tuple[str, str]]:
    """
    IPs IPv4 de la máquina, con el nombre de su interfaz.

    Se consultan con `ip` porque hay que mostrarlas todas: la del wifi de la
    escuela, la del cable y la del hotspot pueden existir a la vez y no hay
    forma de saber de antemano cuál van a poder usar los jugadores.
    """

    encontradas = []

    try:
        salida = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout

        for fila in salida.splitlines():
            partes = fila.split()

            if len(partes) >= 4:
                interfaz = partes[1]
                ip = partes[3].split("/")[0]
                encontradas.append((interfaz, ip))
    except Exception:
        pass

    if not encontradas:
        # Respaldo por si `ip` no existe: preguntamos por la ruta de salida.
        try:
            with socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM,
            ) as s:
                s.connect(("8.8.8.8", 80))
                encontradas.append(("red", s.getsockname()[0]))
        except Exception:
            pass

    return encontradas


def esperar_servidor(timeout=SEGUNDOS_ESPERA_SERVIDOR) -> bool:
    print(
        "Arrancando el servidor "
        "(la primera vez construye el vocabulario "
        "y tarda unos minutos)..."
    )

    inicio = time.time()

    while time.time() - inicio < timeout:
        try:
            urllib.request.urlopen(URL_LOCAL, timeout=2)
            print("Servidor listo.\n")
            return True
        except Exception:
            time.sleep(1)

    return False


def abrir_tunel():
    """
    Intenta publicar el juego en internet.

    Devuelve (url, proceso, log) o (None, proceso, log). Nunca lanza: que no
    haya internet es una posibilidad normal, no un error.
    """

    try:
        log = open(
            ARCHIVO_LOG_TUNEL,
            "w",
            encoding="utf-8",
        )

        proceso = subprocess.Popen(
            [
                "cloudflared",
                "tunnel",
                "--url",
                URL_LOCAL,
            ],
            stdout=log,
            stderr=log,
        )
    except FileNotFoundError:
        print(
            "  cloudflared no está instalado: "
            "se juega solo en red local.\n"
        )
        return None, None, None
    except Exception as error:
        print(f"  No se pudo abrir el túnel: {error}\n")
        return None, None, None

    inicio = time.time()

    while time.time() - inicio < SEGUNDOS_ESPERA_TUNEL:
        if proceso.poll() is not None:
            break

        try:
            with open(
                ARCHIVO_LOG_TUNEL,
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:
                encontrado = PATRON_URL.search(f.read())

            if encontrado:
                return encontrado.group(0), proceso, log
        except FileNotFoundError:
            pass

        time.sleep(0.5)

    return None, proceso, log


def mostrar_qr(url):
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


def main():
    # Se usa el intérprete actual en vez de "uv run": así no importa si uv
    # quedó fuera del PATH, que es un fallo desagradable de diagnosticar
    # cuando estás montando a las 6 de la mañana.
    proceso_servidor = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "web_app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(PUERTO),
        ]
    )

    if not esperar_servidor():
        print(
            "El servidor no respondió a tiempo. "
            "Revisa los errores de arriba."
        )
        proceso_servidor.terminate()
        sys.exit(1)

    print("Buscando salida a internet para el QR...")

    url_tunel, proceso_tunel, log_tunel = abrir_tunel()

    ips = direcciones_locales()

    print()
    linea()
    print("  OPCIÓN 1 - EN ESTA MISMA PC (siempre funciona)")
    linea("-")
    print(f"  Jugar:  {URL_LOCAL}")
    print(f"  Panel:  {URL_LOCAL}/admin")

    if ips:
        print()
        linea()
        print("  OPCIÓN 2 - MISMA WIFI (rápido, no necesita internet)")
        linea("-")

        for interfaz, ip in ips:
            print(f"  [{interfaz}]  http://{ip}:{PUERTO}")

        principal = ips[0][1]

        print()
        print(f"  QR de http://{principal}:{PUERTO}")
        print()
        mostrar_qr(f"http://{principal}:{PUERTO}")

    print()
    linea()

    if url_tunel:
        print("  OPCIÓN 3 - POR INTERNET (desde cualquier red)")
        linea("-")
        print(f"  Jugar:  {url_tunel}")
        print(f"  Panel:  {url_tunel}/admin")
        print()
        mostrar_qr(url_tunel)
    else:
        print("  OPCIÓN 3 - POR INTERNET: NO DISPONIBLE")
        linea("-")
        print("  No se pudo abrir el túnel (sin internet o bloqueado).")
        print("  No pasa nada: usa la opción 1 o la 2.")

    print()
    linea()
    print("  SI NADA FUNCIONA EN EL SITIO")
    linea("-")
    print("  Crea una wifi desde esta PC y que se conecten ahí:")
    print()
    print("    nmcli device wifi hotspot ifname wlp3s0 \\")
    print("      ssid SemantleClase password juguemos123")
    print()
    print("  Luego mira la IP nueva con:  ip addr show wlp3s0")
    print("  (para deshacerlo: nmcli connection down Hotspot)")
    linea()
    print("\n  Ctrl+C para apagar todo.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nApagando...")

        if proceso_tunel is not None:
            proceso_tunel.terminate()

        if log_tunel is not None:
            log_tunel.close()

        proceso_servidor.terminate()
        print("Listo.")


if __name__ == "__main__":
    main()
