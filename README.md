# Semantle Español

Juego de adivinar palabras por **significado**, no por letras. Pensado para que
niños descubran cómo una computadora representa el lenguaje con *embeddings*.

Tiene dos modos de juego más una página educativa aparte
(**Laboratorio de palabras**, en `/laboratorio`) donde se puede sumar y restar
significados (`rey − hombre + mujer = reina`), buscar al intruso de un grupo y
explorar los vecinos de cualquier palabra.

Modos de juego:

- **Jugar solo**: se entra y se juega al instante, sin código ni administrador.
- **Sala**: el profesor crea una sala, comparte el código y juegan todos juntos
  con podio y ranking acumulado.

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

## 4. Instalar cloudflared (solo si vas a jugar por internet)

Si solo vas a jugar en tu propia red, puedes saltarte este paso y usar
`uv run uvicorn web_app:app --host 0.0.0.0 --port 8000`.

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

Verifica con `cloudflared --version` (cierra y abre la terminal si acabas de
instalarlo).

## 5. Correr y compartir

```
uv run compartir.py
```

Esto levanta el servidor y te muestra **todas** las formas de entrar que estén
disponibles en ese momento:

1. **En esta misma PC** (`localhost`) — siempre funciona.
2. **Misma wifi** — la IP de cada interfaz, con QR. No necesita internet.
3. **Por internet** — link de Cloudflare con QR, si se pudo abrir el túnel.

Si no hay internet o el túnel está bloqueado, **el juego sigue funcionando**:
lo avisa y te deja las opciones 1 y 2. `Ctrl+C` apaga todo.

## Primer arranque

La primera vez el juego descarga los vectores de Hugging Face (1,2 GB) y
construye a partir de ellos `vocabulario_es.pt`, un archivo de ~110 MB con las
94.737 palabras limpias que usa la partida.

Ese paso tarda un par de minutos y **ocurre una sola vez**. Después el servidor
solo lee el archivo pequeño y arranca en segundos. Cuando termine puedes borrar
`vectores_es_fp16.pt` para recuperar 1,2 GB de disco; solo hará falta de nuevo
si borras también `vocabulario_es.pt`.

## Cuántos pueden jugar

Medido en una máquina con 6,7 GB de RAM y un solo proceso de Uvicorn:

| Jugadores a la vez | Respuesta (mediana) | Percentil 95 | RAM   |
|--------------------|---------------------|--------------|-------|
| 10                 | 3 ms                | 9 ms         | 485 MB |
| 40                 | 3 ms                | 49 ms        | 643 MB |
| 80                 | 8 ms                | 351 ms       | 778 MB |
| 120                | 8 ms                | 745 ms       | 898 MB |

Sin errores en ningún caso. En la práctica:

- Hasta **40 jugadores simultáneos** va instantáneo.
- Hasta **80** sigue siendo cómodo.
- Más allá de eso conviene levantar varios procesos
  (`uvicorn web_app:app --workers 4`), aunque entonces cada proceso tiene sus
  propias salas en memoria y haría falta compartir el estado.

Una sala admite hasta **30 jugadores** (`MAX_JUGADORES` en `web_app.py`). Las
partidas en solitario no ocupan cupo: cada niño tiene la suya y cuestan unos
5 MB cada una. Las salas inactivas se borran solas después de una hora.

## Cómo se obtiene el código de sala

1. Entra a `/admin` y escribe la contraseña.
2. Elige categoría y palabra secreta, y pulsa **Crear sala**.
3. El código aparece en el panel **Estado de la sala** (6 caracteres, por
   ejemplo `278F7E`).
4. Los jugadores entran a la raíz del sitio, eligen **Unirse a una sala** y
   escriben ese código junto con su nombre.

El código se genera al azar en cada sala nueva y no se guarda en disco: si
reinicias el servidor, se pierde.


## Ajustar el ritmo sin tocar código

Para un stand con grupos rotando conviene acortar las rondas. Se configura en
el `.env` y basta con reiniciar:

```
SEGUNDOS_PARTIDA=90
MAX_INTENTOS=5
MAX_PISTAS=3
MAX_JUGADORES=30
```

Si no pones nada, usa los valores por defecto (120 s, 5 intentos, 3 pistas,
30 jugadores).

## Guía para montar en un stand

Llega, enchufa y corre `uv run compartir.py`. Mira lo que imprime y prueba las
opciones **en este orden**, según lo que tengas delante:

**1. ¿Los niños traen teléfono y hay wifi que los conecte?**
Prueba la **opción 2** (IP local). Abre esa dirección en un celular cualquiera.
Si carga, listo: es la más rápida y no depende de internet.

**2. ¿La IP local no carga en el celular?**
Es aislamiento de clientes en el router de la escuela. Usa la **opción 3**
(QR del túnel). Necesita internet en tu PC y en los celulares.

**3. ¿No hay wifi utilizable?**
Crea la red desde tu propia PC:

```
nmcli device wifi hotspot ifname wlp3s0 ssid SemantleClase password juguemos123
```

Los niños se conectan a `SemantleClase` y entran a la IP que te muestre
`ip addr show wlp3s0` (normalmente `10.42.0.1:8000`). Sin internet, hasta unos
8–15 dispositivos. Para deshacerlo: `nmcli connection down Hotspot`.

**4. ¿Los niños no tienen teléfono, o no los dejan sacarlos?**
Usa la **opción 1**: tu PC es la estación de juego. Modo **Jugar solo**, y al
terminar el botón **Otra palabra** arranca una partida nueva sin escribir nada.
Pasan de a uno o en parejas. Si llevas un par de portátiles o tablets, conéctalos
por la opción 2 o 3 y tienes varias estaciones en paralelo.

### Consejo de organización

Con 30 niños por curso, en vez de que jueguen por turnos de a uno (una hora de
espera), ponlos en **equipos de 2–3 por dispositivo**. Resuelve el límite de
aparatos y funciona mejor: los niños discuten qué palabra probar, que es justo
donde se entiende la idea de significado.

Si vas a usar salas con código, recuerda que **una sala por grupo** mantiene los
podios separados. Crea una sala nueva cuando entre el siguiente curso.


## Seguridad del contenido

El vocabulario sale de texto web y contiene lenguaje adulto (~1.100 términos de
94.737). Por eso:

- **Ninguna pista se genera desde el corpus.** Las 136 palabras jugables tienen
  sus dos pistas escritas a mano en `CATEGORIAS` (`game_engine.py`). Si una
  palabra no las tuviera, el respaldo es la inicial y el número de letras, que
  siempre es seguro.
- La pantalla educativa de vecinos filtra por `PALABRAS_BLOQUEADAS`.
- Si un niño escribe una grosería, se responde igual que con una palabra
  inexistente, sin nombrarla ni mostrarla.

El filtro compara palabras **completas** (con su plural), no prefijos: así
"armario" no cae por "arma" ni "matemático" por "matar".

Después de tocar `CATEGORIAS` o `PALABRAS_BLOQUEADAS`, corre:

```
uv run verificar_contenido.py
```

Revisa las pistas y los 12 vecinos de cada palabra jugable, y además avisa si
alguna pista quedó tan lejos que no ayuda.


## Montarlo en otro computador

Git **no lleva** el modelo ni las contraseñas: `*.pt` y `.env` están en
`.gitignore`. El repo son 134 KB; el modelo son 111 MB aparte.

```
git clone https://github.com/cvail-research/semantle_nlp.git
cd semantle_nlp
uv sync
cp .env.example .env     # y edita ADMIN_PASSWORD y ADMIN_TOKEN
```

Después hace falta el vocabulario, y aquí hay dos caminos muy distintos:

### Opción A — copiarlo por USB (recomendado)

Lleva `vocabulario_es.pt` (111 MB) en una memoria USB y déjalo en la raíz del
proyecto, junto a `web_app.py`. Listo: **no descarga nada** y arranca en
segundos. Es la única opción si en el sitio no hay internet.

### Opción B — dejar que lo descargue

Si el archivo no está, el primer arranque baja `vectores_es_fp16.pt` de Hugging
Face (**1,2 GB**) y construye el vocabulario a partir de él. Tarda varios
minutos y necesita buena conexión. Una vez construido puedes borrar el archivo
de 1,2 GB.

No dejes esto para el día del evento.

### Comprobación antes de salir

```
uv run verificar_contenido.py   # debe decir "Todo limpio"
uv run compartir.py             # y abrir http://localhost:8000
```
