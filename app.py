import streamlit as st
from mvcs_assistant.rag.pipeline import ask_rag
from mvcs_assistant.utils.logger import setup_logger

logger = setup_logger("streamlit")

st.set_page_config(page_title="Asistente MVCS", page_icon="🏛️", layout="wide")
st.title("🏛️ Asistente de Trámites MVCS")
st.caption("Consulta requisitos, plazos, costos y canales usando fuentes oficiales cargadas.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Escribe tu pregunta", placeholder="Ejemplo: ¿Qué requisitos necesito para ...?")

if st.button("Consultar") and question:
    try:
        result = ask_rag(question)
        st.session_state.history.append({"q": question, "a": result["answer"], "sources": result["sources"]})
    except Exception as e:
        logger.exception("Error en consulta")
        st.error(f"Ocurrió un error: {e}")

for i, item in enumerate(reversed(st.session_state.history), start=1):
    st.markdown(f"### Consulta {i}")
    st.write(f"**Pregunta:** {item['q']}")
    st.write(f"**Respuesta:** {item['a']}")
    if item["sources"]:
        st.write("**Fuentes**")
        for s in item["sources"]:
            st.write(f"- {s['source']} | pág: {s['page']} | score: {s['score']:.4f}")
