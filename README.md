# Sistema de Recuperación Semántica Local (RAG)

Sistema RAG (Retrieval-Augmented Generation) end-to-end: recibe una
pregunta, busca los fragmentos más relevantes en una base vectorial local
(ChromaDB) y genera una respuesta usando **exclusivamente** esa
información — sin alucinar datos que no estén en los documentos.

**Módulo 3, Pre-entrega 3** - Programa de AI Engineering @ CodeHouse

---

## 🎯 Qué hace

```
Pregunta del usuario
        │
        ▼
  Embedding de la pregunta
        │
        ▼
  Búsqueda de similitud en ChromaDB  (top_k = 4 fragmentos)
        │
        ▼
  Prompt con el contexto recuperado + "filtro de veracidad"
        │
        ▼
  LLM (OpenAI o Anthropic)
        │
        ▼
  PydanticOutputParser  →  RAGResponse { respuesta, referencias }
```

Si la pregunta no puede responderse con los documentos indexados, el
sistema dice explícitamente que no tiene esa información, en vez de
inventar una respuesta.

---

## 📋 Estructura del Proyecto

```
entregable_3_rag/
├── data/                     # Dataset de ejemplo (documentación técnica interna)
│   ├── 01_arquitectura.md
│   ├── 02_seguridad.md
│   ├── 03_testing.md
│   └── 04_deployment.md
├── tests/                     # Tests automatizados (LLM mockeado, sin API key)
│   ├── test_schemas.py
│   ├── test_ingest.py
│   └── test_rag_chain.py
├── resultados/                 # Se genera al correr main.py (evidencia real de pruebas)
│   └── resultados_pruebas.md
├── embeddings.py             # Fuente única de verdad del modelo de embeddings
├── ingest.py                  # Módulo de Ingesta: carga, chunking y persistencia en ChromaDB
├── schemas.py                 # Modelo Pydantic de la respuesta (RAGResponse)
├── rag_chain.py                # Cadena RAG: retriever + prompt + LLM + parser
├── main.py                      # Mini-script de prueba (pregunta normal + pregunta trampa)
├── pytest.ini                    # Config de pytest
├── requirements.txt               # Dependencias
├── .env.example                    # Plantilla de variables de entorno
└── README.md                         # Este archivo
```

El dataset de ejemplo es documentación técnica interna ficticia de una
empresa (arquitectura, seguridad, testing y despliegue), pero podés
reemplazar los archivos de `/data` por cualquier .txt/.md sobre el tema
que quieras — el pipeline no depende del contenido específico.

---

## 🏗️ Cómo está armado

### 1. `embeddings.py` — Un solo modelo de embeddings, siempre

```python
def get_embeddings():
    if EMBEDDINGS_PROVIDER == "local":
        return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    elif EMBEDDINGS_PROVIDER == "openai":
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, ...)
```

Este archivo existe por una razón puntual: la consigna advierte que el
**error #1** en sistemas RAG es indexar con un modelo de embeddings y
consultar con otro distinto — la distancia vectorial deja de tener
sentido y los resultados de búsqueda se vuelven básicamente aleatorios.
Tanto `ingest.py` como `rag_chain.py` importan `get_embeddings()` desde
acá, así que es imposible que queden desincronizados, sin importar qué
proveedor se elija.

Por defecto (`EMBEDDINGS_PROVIDER=local`), los embeddings corren **100%
local** usando **fastembed** (de Qdrant) — no necesitan ninguna API key
ni generan costo, y a diferencia de `sentence-transformers` no dependen
de PyTorch (lo cual evita un problema común en Windows: PyTorch instala
archivos con rutas internas tan largas que superan el límite de 260
caracteres de Windows y la instalación falla). El modelo (`BAAI/bge-small-en-v1.5`,
~130 MB) se descarga una sola vez la primera vez que corrés `ingest.py`,
y después queda cacheado en tu compu. Si en algún momento contás con
crédito de OpenAI, podés cambiar a `EMBEDDINGS_PROVIDER=openai` en tu
`.env` sin tocar una sola línea de código.

### 2. `ingest.py` — Módulo de Ingesta

- Lee todos los `.txt`/`.md` de `/data`
- Los fragmenta con `RecursiveCharacterTextSplitter`, midiendo el tamaño
  en **tokens reales** (con `tiktoken`, el mismo tokenizador que usan los
  modelos de OpenAI) — no en caracteres ni palabras. Chunk size: 500
  tokens, overlap: 50 tokens, tal como pide la consigna.
- Cada chunk conserva en su metadata el archivo de origen (`source`), para
  poder citarlo como referencia más adelante.
- **Verifica si la base vectorial ya existe** antes de reindexar: si ya
  hay una carpeta `vectorstore/` con datos, la reutiliza directamente sin
  volver a generar embeddings (ahorra tiempo y costo de API).

```python
def ingestar(forzar_reindexado: bool = False) -> Chroma:
    if _base_vectorial_existe() and not forzar_reindexado:
        return Chroma(...)  # reutiliza la base existente
    # si no existe, recién ahí carga, fragmenta e indexa
```

### 3. `schemas.py` — El contrato de salida

```python
class RAGResponse(BaseModel):
    respuesta: str
    referencias: List[str]  # vacío si la respuesta es "no lo sé"
```

### 4. `rag_chain.py` — La cadena RAG completa

El prompt de sistema actúa como **filtro de veracidad**:

> *"Respondé ÚNICAMENTE con información que esté explícitamente en el
> CONTEXTO. [...] Si la respuesta NO se encuentra en el CONTEXTO,
> respondé exactamente: 'No tengo acceso a esa información...' y dejá el
> campo 'referencias' vacío."*

La cadena LCEL:

```python
chain = PROMPT | llm | parser
```

donde `parser` es un `PydanticOutputParser(pydantic_object=RAGResponse)` —
la respuesta cruda del LLM se parsea directamente a un objeto
`RAGResponse` ya validado, nunca queda como texto libre sin estructura.

### 5. `get_rag_response()` — La función asíncrona principal

```python
async def get_rag_response(query: str, provider="openai", top_k=4) -> RAGResponse:
    docs = await retriever.ainvoke(query)        # búsqueda de similitud
    contexto = _formatear_contexto(docs)           # arma el contexto citado
    resultado = await chain.ainvoke({...})          # LLM async + parser
    return resultado
```

**top_k = 4** por defecto (configurable, entre 3 y 5 como recomienda la
consigna) — pasar demasiados fragmentos al LLM degrada la atención del
modelo ("lost in the middle") y puede generar errores de límite de
tokens, por eso no se recuperan decenas de chunks.

---

## 🚀 Instalación y uso

### 1. Instalar dependencias

```bash
cd entregable_3_rag
python3.12 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Instalar y configurar Ollama (generación 100% gratis y local)

Por defecto, este proyecto genera las respuestas con **Ollama**, que corre
un modelo de lenguaje directamente en tu compu — sin ninguna API key ni
costo.

1. Descargá Ollama desde [ollama.com/download](https://ollama.com/download)
   e instalalo (Windows, Mac o Linux).
2. Una vez instalado, Ollama queda corriendo en segundo plano
   automáticamente (escuchando en `localhost:11434`).
3. Descargá el modelo que usa este proyecto por defecto:
   ```bash
   ollama pull llama3.2
   ```
   Esto baja unos ~2 GB la primera vez.
4. Listo — mientras Ollama esté corriendo (lo hace solo, no hace falta
   iniciarlo a mano cada vez), el proyecto ya puede generar respuestas
   sin ninguna configuración adicional.

Si más adelante preferís usar Claude o GPT-4o en vez de un modelo local
(por ejemplo, si cargás crédito en esas cuentas), simplemente cambiá
`GENERATION_PROVIDER` en tu `.env` a `anthropic` u `openai` y completá la
API key correspondiente — no hace falta tocar código.

### 3. Configurar API keys (opcional)

```bash
cp .env.example .env
```

Si querés usar Anthropic u OpenAI en vez de Ollama:

```bash
cp .env.example .env
```

Editá `.env`, cambiá `GENERATION_PROVIDER` a `anthropic` u `openai`, y
completá la API key correspondiente. Los **embeddings nunca necesitan
ninguna API key** — siempre corren localmente.

### 3. Indexar los documentos (primera vez)

```bash
python ingest.py
```

Esto lee `/data`, fragmenta los documentos y crea la carpeta
`vectorstore/` con la base vectorial persistida. Si volvés a correr este
comando, va a detectar que la base ya existe y no va a reindexar (a menos
que borres la carpeta `vectorstore/` o llames a `ingestar(forzar_reindexado=True)`).

### 4. Correr el script de prueba

```bash
python main.py
```

Esto corre dos casos:
1. **Pregunta respondible**: *"¿Qué framework se usa para los endpoints
   REST y por qué se eligió?"* → el sistema recupera el fragmento de
   `01_arquitectura.md` y responde citando esa fuente.
2. **Pregunta trampa**: *"¿Cuál es el presupuesto anual asignado al
   equipo de infraestructura?"* → ningún documento tiene esa información,
   así que el sistema responde que no tiene acceso a esos datos, con
   `referencias` vacío.

### 5. Usarlo en tu propio código

```python
import asyncio
from rag_chain import get_rag_response

async def main():
    respuesta = await get_rag_response("¿Qué base de datos se usa para persistencia?")
    print(respuesta.respuesta)
    print(respuesta.referencias)

asyncio.run(main())
```

---

## 🧪 Ejemplo de salida esperada

Para la pregunta *"¿Qué framework se usa para los endpoints REST?"*:

```json
{
  "respuesta": "El framework utilizado para los endpoints REST es FastAPI, elegido por su soporte nativo de async/await y su integración con Pydantic para validación de datos.",
  "referencias": ["01_arquitectura.md"]
}
```

Para la pregunta trampa *"¿Cuál es el presupuesto anual del equipo de infraestructura?"*:

```json
{
  "respuesta": "No tengo acceso a esa información en la documentación disponible.",
  "referencias": []
}
```

---

## ⚠️ Errores comunes que este proyecto evita explícitamente

| Error común | Cómo se evita acá |
|---|---|
| Embeddings no coincidentes (indexar y consultar con modelos distintos) | `embeddings.py` centraliza el modelo en una sola función usada por ambos módulos |
| "Contexto infinito" (pasar demasiados fragmentos al LLM) | `top_k` limitado a 4 por defecto (entre 3 y 5) |
| Reindexar todo en cada corrida (desperdicio de tiempo/costo) | `ingest.py` verifica si la base ya existe antes de reindexar |
| Alucinación cuando la info no está en el contexto | El prompt de sistema instruye explícitamente a decir "no lo sé" y dejar `referencias` vacío |

---

## 📊 Configuración

| Variable de entorno | Default | Descripción |
|---|---|---|
| `EMBEDDINGS_PROVIDER` | `local` | `local` (fastembed, gratis) u `openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Modelo de embeddings local (fastembed) |
| `GENERATION_PROVIDER` | `ollama` | `ollama` (gratis, local), `anthropic` u `openai` |
| `OLLAMA_MODEL` | `llama3.2` | Modelo de Ollama a usar para generar respuestas |
| `OPENAI_API_KEY` | - | Requerida solo si `GENERATION_PROVIDER=openai` o `EMBEDDINGS_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | - | Requerida solo si `GENERATION_PROVIDER=anthropic` |
| `RAG_TOP_K` | `4` | Cantidad de fragmentos a recuperar por consulta |

---

## 🧪 Tests automatizados (sin necesitar ninguna API key)

Además de las pruebas manuales de `main.py`, el proyecto tiene una suite
de tests con `pytest` que **mockea el LLM y el vectorstore**, así que
corre en cualquier entorno sin necesitar Ollama, ni ninguna API key
configurada:

```bash
pytest -v
```

Qué cubre cada archivo:

- **`test_schemas.py`**: valida el modelo `RAGResponse` — acepta datos
  correctos, rechaza respuestas vacías o demasiado cortas, y confirma
  que `referencias` queda vacío por default.
- **`test_ingest.py`**: valida que la carga y el chunking de los
  documentos de `/data` funcionen como se espera.
- **`test_rag_chain.py`**: el más importante — **simula el LLM con un
  mock** (`RunnableLambda`) que devuelve una respuesta fija, en vez de
  llamar a una API real. Esto permite probar el comportamiento
  anti-alucinación de forma determinística:
  - Con contexto relevante → el sistema devuelve la respuesta y cita la
    fuente correcta.
  - Con contexto vacío (simulando la "pregunta trampa") → el sistema
    devuelve `referencias: []` y una respuesta que admite no tener la
    información, nunca una alucinación.

```python
# Extracto de test_rag_chain.py
def test_get_rag_response_sin_contexto_no_alucina(monkeypatch):
    monkeypatch.setattr(rag_chain, "_get_vectorstore", lambda: _VectorstoreFalso([]))
    monkeypatch.setattr(rag_chain, "get_llm", lambda **kw: _llm_falso_que_devuelve({
        "respuesta": "No tengo acceso a esa información en la documentación disponible.",
        "referencias": [],
    }))
    resultado = asyncio.run(rag_chain.get_rag_response("¿pregunta trampa?"))
    assert resultado.referencias == []
```

---

## 📝 Evidencia de ejecución real

Cada vez que corrés `python main.py`, el script guarda automáticamente
los resultados reales de ambas preguntas (incluida la respuesta real del
LLM a la pregunta trampa) en `resultados/resultados_pruebas.md`. Ese
archivo queda en el repo como evidencia de que el sistema fue probado de
punta a punta con un proveedor real, no solo con los tests mockeados.

---

## ✅ Checklist de la consigna

| Requisito | Dónde está |
|---|---|
| Módulo de Ingesta: chunking + persistencia en ChromaDB | `ingest.py` |
| Capa de Recuperación (embedding de la consulta + búsqueda de similitud) | `rag_chain.py` → `get_rag_response()` |
| Generación Grounded con cadena LCEL, prompt con "filtro de veracidad" | `rag_chain.py` → `PROMPT` + `chain` |
| Chunking mínimo 500 tokens, overlap 50 | `ingest.py` → `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` |
| Mismo modelo de embeddings para indexar y consultar | `embeddings.py` (fuente única) |
| top_k entre 3 y 5 (evita "contexto infinito") | `rag_chain.py` → `TOP_K = 4` |
| Verificación de persistencia antes de reindexar | `ingest.py` → `_base_vectorial_existe()` |
| Función asíncrona `get_rag_response()` con `.ainvoke()` | `rag_chain.py` |
| Salida parseada con `PydanticOutputParser` (texto + referencias) | `schemas.py` + `rag_chain.py` |
| Prueba con pregunta respondible y pregunta trampa | `main.py` |
| Dataset de ejemplo (.txt/.md) | `data/` (4 archivos) |
| Sin API keys en el código (usa `.env`) | Todos los módulos leen de `os.getenv()` |
| Tests automatizados con LLM mockeado (sin API key) | `tests/test_rag_chain.py` |
| Evidencia real de ejecución (incluida la pregunta trampa) | `resultados/resultados_pruebas.md` (se genera al correr `main.py`) |

---

**Listo para conectar con documentación real de cualquier dominio técnico. 🚀**
