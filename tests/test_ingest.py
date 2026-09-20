"""
Tests del módulo de ingesta: validan que la carga de documentos y el
chunking se comporten como corresponde.
"""

from ingest import cargar_documentos, fragmentar_documentos


def test_cargar_documentos_carga_los_4_archivos_de_ejemplo():
    """El dataset de ejemplo tiene exactamente 4 archivos .md en /data."""
    docs = cargar_documentos()
    assert len(docs) == 4

    fuentes = {d.metadata["source"] for d in docs}
    assert fuentes == {
        "01_arquitectura.md",
        "02_seguridad.md",
        "03_testing.md",
        "04_deployment.md",
    }


def test_fragmentar_documentos_preserva_metadata_source():
    """Cada chunk generado debe conservar el archivo de origen en su metadata."""
    docs = cargar_documentos()
    chunks = fragmentar_documentos(docs)

    assert len(chunks) >= len(docs)
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] in {
            "01_arquitectura.md",
            "02_seguridad.md",
            "03_testing.md",
            "04_deployment.md",
        }


def test_chunks_no_estan_vacios():
    """Ningún chunk generado debe quedar con contenido vacío."""
    docs = cargar_documentos()
    chunks = fragmentar_documentos(docs)

    for chunk in chunks:
        assert chunk.page_content.strip() != ""
