"""
Tests de la cadena RAG con el LLM MOCKEADO (sin llamar a ninguna API
real). Esto permite verificar el comportamiento anti-alucinación del
sistema (incluyendo el caso de contexto vacío / "pregunta trampa") en
cualquier entorno, sin necesitar una API key configurada.

Cómo funciona el mock: se reemplaza get_llm() por una función que
devuelve un RunnableLambda -- un "modelo falso" que, en vez de llamar a
una API, simplemente devuelve un texto JSON fijo que simula lo que
respondería un LLM real. Ese texto pasa igual por el PydanticOutputParser
real, así que también se está probando el parseo real, no solo el mock.
"""

import asyncio
import json

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import rag_chain


class _RetrieverFalso:
    """Simula un retriever de LangChain: expone .ainvoke() y devuelve documentos fijos."""

    def __init__(self, documentos):
        self._documentos = documentos

    async def ainvoke(self, query):
        return self._documentos


class _VectorstoreFalso:
    """Simula un vectorstore de Chroma: expone .as_retriever()."""

    def __init__(self, documentos):
        self._documentos = documentos

    def as_retriever(self, search_kwargs=None):
        return _RetrieverFalso(self._documentos)


def _llm_falso_que_devuelve(respuesta_dict: dict) -> RunnableLambda:
    """
    Crea un LLM falso (Runnable) que siempre devuelve el JSON indicado
    como texto, simulando la salida ya formateada de un modelo real.
    """
    texto_json = json.dumps(respuesta_dict, ensure_ascii=False)
    return RunnableLambda(lambda _entrada: texto_json)


def test_formatear_contexto_incluye_etiqueta_de_fuente():
    docs = [
        Document(
            page_content="FastAPI es el framework usado para los endpoints REST.",
            metadata={"source": "01_arquitectura.md"},
        )
    ]
    contexto = rag_chain._formatear_contexto(docs)
    assert "[Fuente: 01_arquitectura.md]" in contexto
    assert "FastAPI" in contexto


def test_formatear_contexto_vacio_no_rompe():
    """Con una lista de documentos vacía, debe devolver un string vacío sin lanzar excepción."""
    contexto = rag_chain._formatear_contexto([])
    assert contexto == ""


class _MensajeFalso:
    """Simula un AIMessage con solo el atributo .content, para probar _extraer_json."""

    def __init__(self, content: str):
        self.content = content


def test_extraer_json_con_json_bien_formado():
    """Caso normal: el LLM ya devolvió JSON válido, se usa tal cual."""
    mensaje = _MensajeFalso('{"respuesta": "FastAPI.", "referencias": ["01_arquitectura.md"]}')
    resultado = rag_chain.parser.parse(rag_chain._extraer_json(mensaje))
    assert resultado.respuesta == "FastAPI."
    assert resultado.referencias == ["01_arquitectura.md"]


def test_extraer_json_con_json_envuelto_en_markdown():
    """El LLM envuelve el JSON en un bloque de código Markdown; igual debe parsear."""
    mensaje = _MensajeFalso('Acá está:\n```json\n{"respuesta": "FastAPI.", "referencias": []}\n```')
    resultado = rag_chain.parser.parse(rag_chain._extraer_json(mensaje))
    assert resultado.respuesta == "FastAPI."


def test_extraer_json_con_formato_texto_plano_de_modelo_local():
    """
    Caso real observado con llama3.2 (Ollama): el modelo ignora el
    formato JSON pedido y responde en texto plano tipo
    "Respuesta: ... \n\n Referencias: [...]". _extraer_json debe
    reconstruir el JSON a partir de ese formato.
    """
    texto_real = (
        'Respuesta:\n"El presupuesto anual asignado al equipo de infraestructura '
        'no se menciona explícitamente en el contexto proporcionado."\n\n'
        "Referencias:\n[]"
    )
    mensaje = _MensajeFalso(texto_real)
    resultado = rag_chain.parser.parse(rag_chain._extraer_json(mensaje))
    assert resultado.referencias == []
    assert "presupuesto" in resultado.respuesta.lower()


def test_get_llm_rechaza_proveedor_invalido():
    with pytest.raises(ValueError):
        rag_chain.get_llm(provider="gemini")


def test_get_rag_response_con_contexto_relevante(monkeypatch):
    """
    Caso 'pregunta respondible': el retriever encuentra un fragmento
    relevante, y el LLM (mockeado) devuelve una respuesta grounded en
    ese fragmento. Verifica que el pipeline entero funciona end-to-end
    sin necesitar ninguna API real.
    """
    docs_relevantes = [
        Document(
            page_content="FastAPI es el framework usado para los endpoints REST.",
            metadata={"source": "01_arquitectura.md"},
        )
    ]

    monkeypatch.setattr(rag_chain, "_get_vectorstore", lambda: _VectorstoreFalso(docs_relevantes))
    monkeypatch.setattr(
        rag_chain,
        "get_llm",
        lambda provider=None, **kwargs: _llm_falso_que_devuelve(
            {
                "respuesta": "El framework utilizado es FastAPI.",
                "referencias": ["01_arquitectura.md"],
            }
        ),
    )

    resultado = asyncio.run(rag_chain.get_rag_response("¿Qué framework se usa para los endpoints REST?"))

    assert resultado.respuesta == "El framework utilizado es FastAPI."
    assert resultado.referencias == ["01_arquitectura.md"]


def test_get_rag_response_sin_contexto_no_alucina(monkeypatch):
    """
    Caso 'pregunta trampa': el retriever NO encuentra ningún documento
    relevante (contexto vacío). Verifica que el resultado final no tenga
    referencias inventadas -- el sistema debe admitir que no tiene la
    información, en vez de alucinar una respuesta.
    """
    monkeypatch.setattr(rag_chain, "_get_vectorstore", lambda: _VectorstoreFalso([]))
    monkeypatch.setattr(
        rag_chain,
        "get_llm",
        lambda provider=None, **kwargs: _llm_falso_que_devuelve(
            {
                "respuesta": "No tengo acceso a esa información en la documentación disponible.",
                "referencias": [],
            }
        ),
    )

    resultado = asyncio.run(
        rag_chain.get_rag_response("¿Cuál es el presupuesto anual del área de infraestructura?")
    )

    assert resultado.referencias == []
    assert "no tengo" in resultado.respuesta.lower()


def test_get_rag_response_rechaza_consulta_vacia():
    with pytest.raises(ValueError):
        asyncio.run(rag_chain.get_rag_response("   "))
