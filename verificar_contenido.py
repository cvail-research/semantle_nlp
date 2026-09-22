"""
Comprueba que nada de lo que ve un niño sea inapropiado.

Recorre todas las palabras jugables y revisa sus pistas y los vecinos que se
muestran en la pantalla educativa. Correr después de tocar CATEGORIAS o
PALABRAS_BLOQUEADAS:

    uv run verificar_contenido.py
"""

import sys

from game_engine import (
    GameEngine,
    asegurar_cache,
    buscar_indice,
    es_apropiada,
    obtener_categorias,
    obtener_palabras_jugables,
)

PUESTO_MAXIMO_PISTA = 3000


def main() -> int:
    asegurar_cache()

    categorias = obtener_categorias()
    palabras = obtener_palabras_jugables()

    print(
        f"Revisando {len(palabras)} palabras "
        f"en {len(categorias)} categorías..."
    )

    inapropiados = []
    inutiles = []

    for palabra in palabras:
        motor = GameEngine(palabra)

        for vecino in motor.obtener_vecinos(12):
            if not es_apropiada(vecino["palabra"]):
                inapropiados.append(
                    f"{palabra}: vecino "
                    f"'{vecino['palabra']}'"
                )

        for numero in (1, 2, 3):
            texto = motor.obtener_pista(numero)

            for pieza in (
                texto
                .replace("'", " ")
                .replace(".", " ")
                .split()
            ):
                if not es_apropiada(pieza):
                    inapropiados.append(
                        f"{palabra}: pista {numero} "
                        f"dice '{pieza}'"
                    )

        for numero, pista in enumerate(
            motor._pistas_configuradas(),
            start=1,
        ):
            indice = buscar_indice(pista)

            if indice is None:
                inutiles.append(
                    f"{palabra}: la pista "
                    f"'{pista}' no está en el "
                    "vocabulario"
                )
                continue

            puesto = motor.obtener_puesto(indice)

            if puesto > PUESTO_MAXIMO_PISTA:
                inutiles.append(
                    f"{palabra}: la pista "
                    f"'{pista}' está en el puesto "
                    f"#{puesto}, demasiado lejos "
                    "para ayudar"
                )

    for problema in inapropiados:
        print(f"  INAPROPIADO  {problema}")

    for problema in inutiles:
        print(f"  INÚTIL       {problema}")

    if inapropiados or inutiles:
        print(
            f"\n{len(inapropiados)} inapropiados, "
            f"{len(inutiles)} inútiles."
        )
        return 1

    print("\nTodo limpio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
