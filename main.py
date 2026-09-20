"""
Mini-script de prueba para el sistema RAG.

Corre dos casos:
1. Una pregunta cuya respuesta SÍ está en los documentos indexados
   (verifica que el sistema recupera y cita correctamente la fuente).
2. Una "pregunta trampa" cuya respuesta NO está en los documentos
   (verifica que el modelo no alucina y admite que no tiene la información).

Cada corrida guarda automáticamente los resultados reales (incluida la
respuesta a la pregunta trampa) en resultados/resultados_pruebas.md,
como evidencia de que el sistema fue probado de punta a punta.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from rag_chain import get_rag_response

load_dotenv()

RESULTADOS_DIR = Path(__file__).parent / "resultados"
RESULTADOS_PATH = RESULTADOS_DIR / "resultados_pruebas.md"

# Pregunta respondible: la respuesta está en 01_arquitectura.md
PREGUNTA_RESPONDIBLE = "¿Qué framework se usa para los endpoints REST y por qué se eligió?"

# Pregunta trampa: no hay ningún documento que hable de esto
PREGUNTA_TRAMPA = "¿Cuál es el presupuesto anual asignado al equipo de infraestructura?"


async def probar_pregunta(descripcion: str, pregunta: str, provider: str) -> dict:
    """
    Corre una pregunta por el pipeline RAG, muestra el resultado en
    consola y lo devuelve como dict (para poder guardarlo como evidencia).
    """
    print(f"\n{'='*70}")
    print(f"Caso: {descripcion}")
    print(f"{'='*70}")
    print(f"\nPregunta: {pregunta}\n")

    try:
        respuesta = await get_rag_response(pregunta, provider=provider)
        print("✓ Respuesta validada:\n")
        print(json.dumps(respuesta.model_dump(), indent=2, ensure_ascii=False))
        return {
            "descripcion": descripcion,
            "pregunta": pregunta,
            "ok": True,
            "respuesta": respuesta.respuesta,
            "referencias": respuesta.referencias,
        }
    except Exception as e:
        mensaje_error = f"{type(e).__name__} - {e}"
        print(f"❌ El pipeline falló: {mensaje_error}")
        return {
            "descripcion": descripcion,
            "pregunta": pregunta,
            "ok": False,
            "error": mensaje_error,
        }


def _guardar_resultados(provider: str, resultados: list) -> None:
    """
    Guarda los resultados reales de la corrida en un archivo Markdown,
    como evidencia de que el sistema fue probado de punta a punta (con
    la respuesta real a la pregunta trampa incluida).
    """
    RESULTADOS_DIR.mkdir(exist_ok=True)

    lineas = [
        "# Resultados de pruebas — Sistema RAG local",
        "",
        f"Última corrida: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Proveedor de generación usado: `{provider}`",
        "",
    ]

    for r in resultados:
        lineas.append(f"## {r['descripcion']}")
        lineas.append("")
        lineas.append(f"**Pregunta:** {r['pregunta']}")
        lineas.append("")
        if r["ok"]:
            lineas.append(f"**Respuesta:** {r['respuesta']}")
            lineas.append("")
            lineas.append(f"**Referencias:** {r['referencias']}")
        else:
            lineas.append(f"**❌ Error:** {r['error']}")
        lineas.append("")

    RESULTADOS_PATH.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\n✓ Resultados guardados en {RESULTADOS_PATH.relative_to(Path(__file__).parent)}")


async def main():
    """Punto de entrada: corre ambos casos de prueba con el proveedor configurado."""
    from rag_chain import GENERATION_PROVIDER

    print(f"\n{'='*70}")
    print("Sistema RAG local — Suite de Pruebas")
    print(f"{'='*70}")
    print(f"Proveedor de generación: {GENERATION_PROVIDER}")
    print("(los embeddings corren 100% local, ver embeddings.py — no usan API key)")

    if GENERATION_PROVIDER == "ollama":
        print(
            "\n⚠ Asegurate de tener Ollama corriendo localmente y el modelo "
            "descargado (ver README.md, sección Ollama)."
        )
    elif GENERATION_PROVIDER == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n❌ GENERATION_PROVIDER=anthropic pero falta ANTHROPIC_API_KEY en .env")
        return
    elif GENERATION_PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("\n❌ GENERATION_PROVIDER=openai pero falta OPENAI_API_KEY en .env")
        return

    resultados = []

    resultados.append(
        await probar_pregunta(
            "Pregunta respondible (la info está en los documentos)",
            PREGUNTA_RESPONDIBLE,
            GENERATION_PROVIDER,
        )
    )

    resultados.append(
        await probar_pregunta(
            "Pregunta trampa (la info NO está en los documentos)",
            PREGUNTA_TRAMPA,
            GENERATION_PROVIDER,
        )
    )

    _guardar_resultados(GENERATION_PROVIDER, resultados)

    print(f"\n{'='*70}")
    print("Suite de pruebas completada")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
