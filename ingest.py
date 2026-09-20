"""
Módulo de Ingesta (Setup) del sistema RAG.

Lee documentos de /data, los fragmenta (chunking) respetando límites de
tokens, y los persiste en una colección local de ChromaDB. Si la base
vectorial ya existe, no vuelve a indexar todo desde cero (evita gastar
tiempo y costo de embeddings innecesariamente).
"""

import logging
from pathlib import Path
from typing import List

import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import get_embeddings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "documentos_tecnicos"

# Chunking: mínimo 500 tokens con 50 de overlap, medido en tokens reales
# (no en caracteres), usando el mismo tokenizador que usan los modelos de
# OpenAI, para que el tamaño de cada fragmento sea predecible.
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

_encoding = tiktoken.get_encoding("cl100k_base")


def _contar_tokens(texto: str) -> int:
    """Cuenta tokens reales (no caracteres ni palabras) usando tiktoken."""
    return len(_encoding.encode(texto))


# ---------------------------------------------------------------------------
# Carga de documentos
# ---------------------------------------------------------------------------

def cargar_documentos(data_dir: Path = DATA_DIR) -> List[Document]:
    """
    Lee todos los archivos .txt y .md de la carpeta indicada.

    Args:
        data_dir: carpeta con los documentos fuente

    Returns:
        Lista de Document de LangChain, con el nombre de archivo en
        metadata["source"] para poder citarlo después como referencia.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta de datos: {data_dir}. "
            f"Creala y poné ahí tus archivos .txt/.md."
        )

    documentos: List[Document] = []
    archivos = sorted(
        [*data_dir.glob("*.txt"), *data_dir.glob("*.md")]
    )

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos .txt o .md en {data_dir}."
        )

    for archivo in archivos:
        contenido = archivo.read_text(encoding="utf-8")
        doc = Document(page_content=contenido, metadata={"source": archivo.name})
        documentos.append(doc)
        logger.info(f"Cargado: {archivo.name} ({len(contenido)} caracteres)")

    logger.info(f"Total de documentos cargados: {len(documentos)}")
    return documentos


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def fragmentar_documentos(documentos: List[Document]) -> List[Document]:
    """
    Fragmenta los documentos usando RecursiveCharacterTextSplitter, con el
    tamaño medido en tokens reales (no caracteres).

    Args:
        documentos: documentos ya cargados

    Returns:
        Lista de fragmentos (chunks), cada uno conservando el metadata
        original (incluyendo "source") para poder citarlo como referencia.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_contar_tokens,
    )
    chunks = splitter.split_documents(documentos)
    logger.info(
        f"Documentos fragmentados en {len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE_TOKENS} tokens, overlap={CHUNK_OVERLAP_TOKENS} tokens)"
    )
    return chunks


# ---------------------------------------------------------------------------
# Persistencia en ChromaDB
# ---------------------------------------------------------------------------

def _base_vectorial_existe() -> bool:
    """Verifica si ya hay una base vectorial persistida en disco."""
    return PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())


def ingestar(forzar_reindexado: bool = False) -> Chroma:
    """
    Punto de entrada principal del módulo de ingesta.

    Si la base vectorial ya existe en disco, la reutiliza directamente sin
    volver a leer/fragmentar/embeber los documentos (optimiza tiempo y
    costo). Para forzar una re-indexación completa, pasar forzar_reindexado=True.

    Args:
        forzar_reindexado: si es True, ignora la base existente y reindexa
                            todo desde cero

    Returns:
        Instancia de Chroma lista para usarse como vectorstore/retriever
    """
    embeddings = get_embeddings()

    if _base_vectorial_existe() and not forzar_reindexado:
        logger.info(
            f"Base vectorial ya existente en '{PERSIST_DIR}' — se reutiliza "
            f"sin volver a indexar. Usá forzar_reindexado=True si necesitás "
            f"reindexar desde cero."
        )
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings,
        )

    logger.info("No hay base vectorial previa (o se forzó reindexado). Indexando desde cero...")
    documentos = cargar_documentos()
    chunks = fragmentar_documentos(documentos)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )
    logger.info(f"Indexación completa: {len(chunks)} chunks persistidos en '{PERSIST_DIR}'")
    return vectorstore


if __name__ == "__main__":
    ingestar()
