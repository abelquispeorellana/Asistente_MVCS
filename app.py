import sys, os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from mvcs_assistant.config.settings import settings
from mvcs_assistant.ingestion.loaders import load_documents
from mvcs_assistant.ingestion.preprocess import enrich_metadata, deduplicate_docs, chunk_documents
from mvcs_assistant.rag.vectorstore import get_vectorstore
from mvcs_assistant.rag.pipeline import ask_rag
from mvcs_assistant.utils.logger import setup_logger

logger = setup_logger("streamlit")

EJEMPLOS = [
    "¿Qué requisitos necesito para obtener una constancia de registro MVCS?",
    "¿Cuál es el plazo de atención para los trámites del MVCS?",
    "¿Cuánto cuesta solicitar una constancia de registro?",
    "¿Por qué canal puedo presentar mi solicitud al MVCS?",
    "¿Qué documentos debo adjuntar a mi solicitud?",
]

st.set_page_config(page_title="Asistente MVCS", page_icon="🏛️", layout="wide")
st.title("🏛️ Asistente de Trámites MVCS")
st.caption("Consulta requisitos, plazos, costos y canales de atención usando fuentes oficiales.")


@st.cache_resource(show_spinner="Cargando base de conocimiento MVCS...")
def init_vectorstore():
    vs = get_vectorstore()
    if vs._collection.count() == 0:
        with st.spinner("Indexando documentos por primera vez..."):
            docs = load_documents(settings.raw_data_dir)
            docs = [enrich_metadata(d) for d in docs]
            docs = deduplicate_docs(docs)
            chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
            vs.add_documents(chunks)
            logger.info("Auto-ingesta completada: %s chunks", len(chunks))
    return vs


init_vectorstore()

# --- Sidebar ---
with st.sidebar:
    st.header("Ejemplos de consultas")
    st.caption("Haz clic para usar como pregunta:")
    for ej in EJEMPLOS:
        if st.button(ej, use_container_width=True):
            st.session_state["_ejemplo"] = ej
    st.divider()
    st.info("Fuentes cargadas:\n- TUPA MVCS (105 págs.)\n- Ejemplo de trámite")

# --- Main ---
if "history" not in st.session_state:
    st.session_state.history = []

pregunta_default = st.session_state.pop("_ejemplo", "") if "_ejemplo" in st.session_state else ""

question = st.text_input(
    "Escribe tu pregunta",
    value=pregunta_default,
    placeholder="Ejemplo: ¿Qué requisitos necesito para obtener una licencia de construcción?",
)

if st.button("Consultar", type="primary") and question:
    with st.spinner("Buscando en fuentes oficiales..."):
        try:
            result = ask_rag(question)
            st.session_state.history.insert(0, {
                "q": question,
                "a": result["answer"],
                "sources": result["sources"],
            })
        except Exception as e:
            logger.exception("Error en consulta")
            st.error(f"Ocurrió un error: {e}")

# --- Historial ---
for item in st.session_state.history:
    with st.container(border=True):
        st.markdown(f"**Pregunta:** {item['q']}")
        st.markdown(f"**Respuesta:** {item['a']}")
        if item["sources"]:
            with st.expander("Ver fuentes"):
                for s in item["sources"]:
                    st.write(f"- `{os.path.basename(s['source'])}` | pág: {s['page']} | score: {s['score']:.4f}")
