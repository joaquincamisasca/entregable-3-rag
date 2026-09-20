"""
Tests del modelo Pydantic RAGResponse: validan que el schema de salida
del sistema RAG se comporte como corresponde (campos requeridos,
defaults, y rechazo de datos inválidos).
"""

import pytest
from pydantic import ValidationError

from schemas import RAGResponse


def test_respuesta_valida_con_referencias():
    """Un RAGResponse con respuesta y referencias válidas se crea sin problemas."""
    r = RAGResponse(
        respuesta="El framework usado para los endpoints REST es FastAPI.",
        referencias=["01_arquitectura.md"],
    )
    assert r.respuesta == "El framework usado para los endpoints REST es FastAPI."
    assert r.referencias == ["01_arquitectura.md"]


def test_referencias_default_vacio_si_no_se_pasa():
    """Si no se especifica 'referencias', debe quedar como lista vacía (caso 'no lo sé')."""
    r = RAGResponse(respuesta="No tengo acceso a esa información en la documentación disponible.")
    assert r.referencias == []


def test_respuesta_muy_corta_es_rechazada():
    """respuesta tiene min_length=3: un texto más corto debe fallar la validación."""
    with pytest.raises(ValidationError):
        RAGResponse(respuesta="ok", referencias=[])


def test_respuesta_vacia_o_solo_espacios_es_rechazada():
    """El validador de respuesta_no_vacia debe rechazar strings en blanco."""
    with pytest.raises(ValidationError):
        RAGResponse(respuesta="   ", referencias=[])


def test_model_config_usa_configdict_pydantic_v2():
    """
    Confirma que el modelo usa la sintaxis moderna de Pydantic v2
    (model_config = ConfigDict(...)) en vez de la vieja `class Config`,
    que genera warnings de deprecación.
    """
    assert hasattr(RAGResponse, "model_config")
    assert "json_schema_extra" in RAGResponse.model_config
