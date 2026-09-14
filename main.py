"""
Mini-script de prueba para el sistema RAG.

Corre dos casos:
1. Una pregunta cuya respuesta SÍ está en los documentos indexados
   (verifica que el sistema recupera y cita correctamente la fuente).
2. Una "pregunta trampa" cuya respuesta NO está en los documentos
   (verifica que el modelo no alucina y admite que no tiene la información).
"""

import asyncio
import json
import os

from dotenv import load_dotenv

from rag_chain import get_rag_response

load_dotenv()

# Pregunta respondible: la respuesta está en 01_arquitectura.md
PREGUNTA_RESPONDIBLE = "¿Qué framework se usa para los endpoints REST y por qué se eligió?"

# Pregunta trampa: no hay ningún documento que hable de esto
PREGUNTA_TRAMPA = "¿Cuál es el presupuesto anual asignado al equipo de infraestructura?"


async def probar_pregunta(descripcion: str, pregunta: str, provider: str) -> None:
    """Corre una pregunta por el pipeline RAG y muestra el resultado validado."""
    print(f"\n{'='*70}")
    print(f"Caso: {descripcion}")
    print(f"{'='*70}")
    print(f"\nPregunta: {pregunta}\n")

    try:
        respuesta = await get_rag_response(pregunta, provider=provider)
        print("✓ Respuesta validada:\n")
        print(json.dumps(respuesta.model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ El pipeline falló: {type(e).__name__} - {e}")


async def main():
    """Punto de entrada: corre ambos casos de prueba."""
    if os.getenv("ANTHROPIC_API_KEY"):
        provider = "anthropic"
    elif os.getenv("OPENAI_API_KEY"):
        provider = "openai"
    else:
        print(
            "❌ No se encontró ninguna API key configurada.\n"
            "   Copiá .env.example a .env y completá ANTHROPIC_API_KEY "
            "o OPENAI_API_KEY (los embeddings son locales y no necesitan key)."
        )
        return

    print(f"\n{'='*70}")
    print("Sistema RAG local — Suite de Pruebas")
    print(f"{'='*70}")
    print(f"Proveedor de generación: {provider}")
    print("(los embeddings corren 100% local, ver embeddings.py — no usan API key)")

    await probar_pregunta(
        "Pregunta respondible (la info está en los documentos)",
        PREGUNTA_RESPONDIBLE,
        provider,
    )

    await probar_pregunta(
        "Pregunta trampa (la info NO está en los documentos)",
        PREGUNTA_TRAMPA,
        provider,
    )

    print(f"\n{'='*70}")
    print("Suite de pruebas completada")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
