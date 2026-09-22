"""
Levanta el servidor local, abre un túnel público con cloudflared,
y muestra un QR para que los jugadores entren + el link de admin para ti.

Uso:
    uv run --with qrcode compartir.py

Requiere tener cloudflared instalado (ver README).
"""

import re
import subprocess
import sys
import time
import urllib.request

import qrcode

PUERTO = 8000
URL_LOCAL = f"http://localhost:{PUERTO}"
ARCHIVO_LOG_TUNEL = "tunel.log"

PATRON_URL = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def esperar_servidor(timeout_segundos=300):
    print("Esperando a que el servidor arranque (puede tardar si descarga el modelo)...")
    inicio = time.time()
    while time.time() - inicio < timeout_segundos:
        try:
            urllib.request.urlopen(URL_LOCAL, timeout=2)
            print("Servidor listo.")
            return True
        except Exception:
            time.sleep(1)
    return False


def esperar_url_tunel(archivo_log, timeout_segundos=60):
    inicio = time.time()
    while time.time() - inicio < timeout_segundos:
        try:
            with open(archivo_log, "r", encoding="utf-8", errors="ignore") as f:
                contenido = f.read()
            match = PATRON_URL.search(contenido)
            if match:
                return match.group(0)
        except FileNotFoundError:
            pass
        time.sleep(0.5)
    return None


def mostrar_qr(url):
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


def main():
    print("Iniciando servidor local...\n")
    proceso_servidor = subprocess.Popen(
        ["uv", "run", "uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", str(PUERTO)]
    )

    if not esperar_servidor():
        print("El servidor no respondió a tiempo. Revisa los errores arriba.")
        proceso_servidor.terminate()
        sys.exit(1)

    print("\nAbriendo túnel público con cloudflared...\n")
    log_tunel = open(ARCHIVO_LOG_TUNEL, "w", encoding="utf-8")
    proceso_tunel = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", URL_LOCAL],
        stdout=log_tunel,
        stderr=log_tunel,
    )

    url_publica = esperar_url_tunel(ARCHIVO_LOG_TUNEL)

    if not url_publica:
        print("No se pudo obtener el link del túnel a tiempo. Revisa tunel.log")
        proceso_servidor.terminate()
        proceso_tunel.terminate()
        sys.exit(1)

    url_admin = url_publica + "/admin"

    print("\n" + "=" * 60)
    print("LINK PARA JUGADORES (compártelo o muestra el QR):")
    print(url_publica)
    print()
    mostrar_qr(url_publica)
    print()
    print("LINK PARA TI (panel de administrador):")
    print(url_admin)
    print("=" * 60)
    print("\nPresiona Ctrl+C para apagar todo.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nApagando servidor y túnel...")
        proceso_tunel.terminate()
        proceso_servidor.terminate()
        log_tunel.close()
        print("Listo.")


if __name__ == "__main__":
    main()
