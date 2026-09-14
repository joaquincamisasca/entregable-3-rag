"""
Esquema de salida para el sistema RAG: la respuesta final debe incluir
tanto el texto generado como las referencias (documentos fuente) que lo
sustentan.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class RAGResponse(BaseModel):
    """
    Respuesta validada del sistema RAG.

    El campo `referencias` debe listar únicamente los documentos fuente
    que efectivamente respaldan la respuesta. Si el sistema no encuentra
    la información en el contexto recuperado, `referencias` debe quedar
    vacío y `respuesta` debe indicar explícitamente que no tiene esa
    información (nunca inventar un dato para "completar" el campo).
    """

    respuesta: str = Field(
        ...,
        min_length=3,
        description=(
            "Respuesta generada, basada exclusivamente en el CONTEXTO "
            "recuperado de la base vectorial. Si la información no está "
            "en el contexto, debe indicar explícitamente que no se cuenta "
            "con esa información."
        ),
    )
    referencias: List[str] = Field(
        default_factory=list,
        description=(
            "Nombres de los documentos fuente (ej: '01_arquitectura.md') "
            "que efectivamente respaldan la respuesta. Vacío si la "
            "respuesta es 'no lo sé'."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "respuesta": (
                    "El framework utilizado para los endpoints REST es "
                    "FastAPI, elegido por su soporte nativo de async/await "
                    "y su integración con Pydantic."
                ),
                "referencias": ["01_arquitectura.md"],
            }
        }
    )
