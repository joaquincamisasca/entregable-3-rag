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

import json
import logging
import os
import re
from typing import List, Literal, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

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

# Proveedor de generación por default. "ollama" es 100% gratis y corre
# local (requiere tener Ollama instalado y corriendo en la compu, ver
# README.md). Se puede cambiar a "anthropic" u "openai" seteando
# GENERATION_PROVIDER en el .env, sin tocar código.
GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "ollama")

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
    provider: Literal["openai", "anthropic", "ollama"] = "ollama",
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
    elif provider == "ollama":
        # Import diferido: quien use OpenAI/Anthropic no necesita tener
        # instalado langchain-ollama ni Ollama corriendo localmente.
        from langchain_ollama import ChatOllama

        model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        return ChatOllama(model=model, temperature=temperature)
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


def _extraer_json(mensaje) -> str:
    """
    Intenta convertir la salida cruda del LLM en un JSON válido para el
    parser, con dos estrategias en cascada:

    1. Si la salida ya contiene un bloque {...}, lo usa directamente
       (caso normal: el modelo respetó el formato pedido).
    2. Si no hay ningún {...} -- como pasó en la práctica con llama3.2,
       que a veces responde en texto plano tipo "Respuesta: ... \n\n
       Referencias: [...]" en vez del JSON pedido -- reconstruye el JSON
       a mano parseando ese formato con regex.

    Si ninguna de las dos estrategias reconoce el texto, lo devuelve tal
    cual y deja que el parser (y el reintento de .with_retry()) se
    encarguen de la falla.
    """
    texto = mensaje.content if hasattr(mensaje, "content") else str(mensaje)

    # Estrategia 1: ya viene como JSON (envuelto en texto o no)
    match_json = re.search(r"\{.*\}", texto, re.DOTALL)
    if match_json:
        return match_json.group(0)

    # Estrategia 2: formato "Respuesta: ... \n\n Referencias: [...]"
    match_respuesta = re.search(
        r"respuesta:?\s*[\"']?(.+?)[\"']?\s*\n\s*\n", texto, re.IGNORECASE | re.DOTALL
    )
    if match_respuesta:
        respuesta_texto = match_respuesta.group(1).strip().strip('"').strip("'")

        match_referencias = re.search(r"referencias:?\s*(\[.*?\])", texto, re.IGNORECASE | re.DOTALL)
        try:
            referencias = json.loads(match_referencias.group(1)) if match_referencias else []
        except json.JSONDecodeError:
            referencias = []

        return json.dumps({"respuesta": respuesta_texto, "referencias": referencias}, ensure_ascii=False)

    # No se reconoció ningún formato: se devuelve tal cual
    return texto


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
    provider: Literal["openai", "anthropic", "ollama"] = GENERATION_PROVIDER,
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

    # 3. Cadena LCEL: prompt -> LLM -> limpieza de JSON -> parser
    #
    # Modelos locales chicos (como llama3.2 vía Ollama) a veces no
    # devuelven el JSON "puro" que pide el prompt, y lo envuelven en
    # texto adicional (ej: 'Respuesta: "..." \n\nReferencias: []' en vez
    # de '{"respuesta": "...", "referencias": []}'). _extraer_json()
    # busca el primer bloque que parece un objeto JSON antes de
    # parsearlo, en vez de asumir que la salida ya viene perfecta.
    #
    # Como capa extra de resiliencia, .with_retry() reintenta toda la
    # cadena (nueva llamada al LLM incluida) si el parseo sigue
    # fallando -- el mismo patrón de reintento con backoff exponencial
    # que se usó en los entregables anteriores.
    llm = get_llm(provider=provider)
    base_chain = PROMPT | llm | RunnableLambda(_extraer_json) | parser
    chain = base_chain.with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError),
        wait_exponential_jitter=True,
        stop_after_attempt=3,
    )

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
