# Semantle Español

## 1. Clonar el repositorio

```
git clone https://github.com/cvail-research/semantle_nlp.git
cd semantle_nlp
```

## 2. Instalar dependencias

```
uv sync
```

## 3. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto con:

```
ADMIN_PASSWORD=tu_contraseña
ADMIN_TOKEN=tu_token
```

Pide estos valores a quien te compartió el proyecto.

## 4. Instalar cloudflared (solo la primera vez)

**Windows:**
```
winget install --id Cloudflare.cloudflared
```

**Linux:**
```
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

**Mac:**
```
brew install cloudflared
```

Verifica con `cloudflared --version` (cierra y abre la terminal si acabas de instalarlo).

## 5. Correr y compartir

```
uv run compartir.py
```

Esto levanta el servidor (descarga los pesos la primera vez, puede tardar 1-2 min), abre el túnel, y muestra:

- Un **QR + link** para que los jugadores entren.
- El **link de admin** para ti.

`Ctrl+C` apaga todo.