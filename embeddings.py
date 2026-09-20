"""
Módulo centralizado para el modelo de embeddings.

Este es un archivo deliberadamente chico: existe para garantizar que
ingest.py (que indexa) y rag_chain.py (que consulta) usen exactamente el
mismo modelo de embeddings. Si se indexara con un modelo y se consultara
con otro, la distancia vectorial entre las consultas y los documentos ya
no tendría sentido y los resultados de búsqueda serían básicamente
aleatorios — es el error #1 que advierte la consigna de este entregable.

Por defecto usamos un modelo de embeddings LOCAL con **fastembed** (de
Qdrant): corre en CPU vía ONNX Runtime, no requiere ninguna API key y no
depende de PyTorch — esto último es importante en Windows, donde
instalar PyTorch (como hace sentence-transformers) puede fallar por el
límite de longitud de rutas de archivo del sistema operativo. fastembed
evita ese problema por completo y además es una descarga mucho más
liviana.

Opcionalmente, si se cuenta con crédito de OpenAI, se puede cambiar a
OpenAIEmbeddings con una sola variable de entorno (EMBEDDINGS_PROVIDER),
sin tocar ni ingest.py ni rag_chain.py — el mismo patrón de proveedor
intercambiable que se usó en los entregables anteriores.
"""

import os
from typing import List

from langchain_core.embeddings import Embeddings

# "local" (default, sin costo, sin API key) o "openai" (requiere
# OPENAI_API_KEY y crédito cargado en la cuenta)
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "local")

# BAAI/bge-small-en-v1.5 es un modelo chico y rápido soportado nativamente
# por fastembed, con muy buen resultado para búsqueda semántica.
LOCAL_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


class FastEmbedEmbeddings(Embeddings):
    """
    Wrapper mínimo que adapta la librería fastembed a la interfaz
    Embeddings de LangChain (embed_documents / embed_query), para poder
    usarla directamente en Chroma como cualquier otro embedding de
    LangChain.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self._model.embed([text]))).tolist()


def get_embeddings():
    """
    Devuelve siempre la misma instancia de modelo de embeddings, según
    EMBEDDINGS_PROVIDER ("local" por default, o "openai").

    Tanto ingest.py como rag_chain.py deben importar esta función en vez
    de instanciar el modelo de embeddings por su cuenta, para que
    indexación y consulta queden garantizadas en el mismo espacio
    vectorial.

    Los imports de cada proveedor son diferidos (dentro de la función):
    así, si usás "local" nunca hace falta tener instalado ni configurado
    nada de OpenAI, y viceversa.

    Raises:
        ValueError: si EMBEDDINGS_PROVIDER tiene un valor no soportado
    """
    if EMBEDDINGS_PROVIDER == "local":
        return FastEmbedEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)

    elif EMBEDDINGS_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    else:
        raise ValueError(
            f"EMBEDDINGS_PROVIDER no soportado: '{EMBEDDINGS_PROVIDER}'. "
            f"Usá 'local' o 'openai'."
        )
