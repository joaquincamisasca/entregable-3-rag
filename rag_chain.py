"""
Cadena RAG (Retrieval-Augmented Generation) end-to-end.

Flujo: consulta del usuario -> embedding -> búsqueda de similitud en
ChromaDB -> construcción del prompt con los fragmentos recuperados ->
LLM -> PydanticOutputParser -> RAGResponse validado.

El prompt está diseñado para actuar como un "filtro de veracidad": el
modelo solo puede responder con información presente en el contexto
recuperado, y debe decir explícitamente que no tiene la información si
no está ahí (grounded generation, sin alucinaciones).
"""

import logging
import os
from typing import List, Literal, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ingest import ingestar
from schemas import RAGResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# top_k entre 3 y 5, tal como recomienda la consigna, para evitar el
# "contexto infinito" (degradación de atención / límite de tokens si se
# pasan demasiados fragmentos al LLM).
TOP_K = int(os.getenv("RAG_TOP_K", "4"))

parser = PydanticOutputParser(pydantic_object=RAGResponse)

SYSTEM_PROMPT = (
    "Sos un asistente técnico interno. Tu única fuente de verdad es el "
    "CONTEXTO que se te proporciona a continuación, extraído de la "
    "documentación oficial de la empresa.\n\n"
    "Reglas estrictas:\n"
    "1. Respondé ÚNICAMENTE con información que esté explícitamente en "
    "el CONTEXTO. Nunca uses conocimiento general ni supuestos externos.\n"
    "2. Si la respuesta a la pregunta NO se encuentra en el CONTEXTO, "
    "respondé exactamente: 'No tengo acceso a esa información en la "
    "documentación disponible.' y dejá el campo 'referencias' vacío.\n"
    "3. Si respondés en base al CONTEXTO, indicá en 'referencias' "
    "únicamente los nombres de archivo que aparecen en las etiquetas "
    "[Fuente: ...] que realmente usaste para construir la respuesta.\n\n"
    "{formato_instrucciones}"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ]
)


# ---------------------------------------------------------------------------
# Cliente de LLM (mismo patrón intercambiable de entregables anteriores)
# ---------------------------------------------------------------------------

def get_llm(
    provider: Literal["openai", "anthropic"] = "openai",
    model: Optional[str] = None,
    temperature: float = 0.0,
):
    """Crea el chat model de LangChain para el proveedor pedido."""
    if provider == "openai":
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        return ChatOpenAI(
            model=model, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "anthropic":
        model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return ChatAnthropic(
            model=model, temperature=temperature, api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    else:
        raise ValueError(f"Proveedor no soportado: {provider}")


# ---------------------------------------------------------------------------
# Retriever y formateo de contexto
# ---------------------------------------------------------------------------

def _formatear_contexto(docs: List[Document]) -> str:
    """
    Concatena los fragmentos recuperados en un bloque de texto, etiquetando
    cada uno con su archivo de origen para que el modelo pueda citarlo
    correctamente en el campo 'referencias'.
    """
    partes = []
    for doc in docs:
        fuente = doc.metadata.get("source", "desconocido")
        partes.append(f"[Fuente: {fuente}]\n{doc.page_content}")
    return "\n\n---\n\n".join(partes)


_vectorstore = None  # cacheado en memoria para no reabrir la conexión en cada consulta


def _get_vectorstore():
    """Obtiene (o abre una vez) la base vectorial persistida por ingest.py."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = ingestar()
    return _vectorstore


# ---------------------------------------------------------------------------
# Función principal: get_rag_response()
# ---------------------------------------------------------------------------

async def get_rag_response(
    query: str,
    provider: Literal["openai", "anthropic"] = "openai",
    top_k: int = TOP_K,
) -> RAGResponse:
    """
    Ejecuta el flujo RAG completo de forma asíncrona:
      1. Convierte la consulta en un embedding y busca los fragmentos más
         relevantes en ChromaDB (búsqueda de similitud).
      2. Construye el prompt incluyendo esos fragmentos como contexto.
      3. Llama al LLM de forma asíncrona (.ainvoke()).
      4. Parsea la respuesta con PydanticOutputParser hacia un RAGResponse
         validado, que incluye el texto y las referencias usadas.

    Args:
        query: pregunta del usuario en lenguaje natural
        provider: "openai" o "anthropic"
        top_k: cantidad de fragmentos a recuperar (recomendado: 3 a 5)

    Returns:
        RAGResponse con la respuesta generada y las referencias citadas

    Raises:
        ValueError: si la consulta está vacía
        Exception: si falla la generación o el parseo de la respuesta
    """
    if not query or not query.strip():
        raise ValueError("La consulta no puede estar vacía.")

    logger.info(f"Consulta recibida: '{query}' (top_k={top_k})")

    # 1. Búsqueda de similitud en ChromaDB
    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = await retriever.ainvoke(query)
    logger.info(
        f"Se recuperaron {len(docs)} fragmentos relevantes: "
        f"{[d.metadata.get('source') for d in docs]}"
    )

    # 2. Construcción del contexto para el prompt
    contexto = _formatear_contexto(docs)

    # 3. Cadena LCEL: prompt -> LLM -> parser (generación asíncrona)
    llm = get_llm(provider=provider)
    chain = PROMPT | llm | parser

    try:
        resultado = await chain.ainvoke(
            {
                "contexto": contexto,
                "pregunta": query,
                "formato_instrucciones": parser.get_format_instructions(),
            }
        )
        logger.info(f"Respuesta validada generada: {resultado.model_dump()}")
        return resultado

    except Exception as e:
        logger.error(
            f"Error al generar o validar la respuesta RAG: "
            f"{type(e).__name__} - {e}"
        )
        raise
