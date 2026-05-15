from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from mvcs_assistant.config.settings import settings
from mvcs_assistant.prompts.templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from mvcs_assistant.rag.vectorstore import get_vectorstore


def build_context_with_scores(results: list[tuple]) -> str:
    lines = []
    for doc, score in results:
        source = doc.metadata.get("source", "desconocido")
        page = doc.metadata.get("page", "N/A")
        lines.append(f"{doc.page_content}\n[fuente={source} | página={page} | score={score:.4f}]")
    return "\n\n".join(lines)


def _raw_answer(results: list[tuple]) -> str:
    lines = ["**Información encontrada en fuentes oficiales** *(síntesis de IA no disponible por límite de cuota — reintenta en 1 minuto)*\n"]
    for doc, score in results:
        page = doc.metadata.get("page", "N/A")
        lines.append(f"— {doc.page_content.strip()} *(pág. {page})*")
    return "\n\n".join(lines)


def ask_rag(question: str) -> dict:
    vs = get_vectorstore()
    retriever_results = vs.similarity_search_with_relevance_scores(question, k=settings.k_retrieval)

    if not retriever_results or max(score for _, score in retriever_results) < settings.score_threshold:
        return {
            "answer": "No tengo evidencia suficiente en las fuentes cargadas para responder esta consulta.",
            "sources": [],
        }

    context = build_context_with_scores(retriever_results)
    sources = [
        {
            "source": d.metadata.get("source", "desconocido"),
            "page": d.metadata.get("page", "N/A"),
            "score": float(score),
        }
        for d, score in retriever_results
    ]

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT_TEMPLATE),
        ])
        response = (prompt | llm).invoke({"question": question, "context": context})
        return {"answer": response.content, "sources": sources}

    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return {"answer": _raw_answer(retriever_results), "sources": sources}
        raise
