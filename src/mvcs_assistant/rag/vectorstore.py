from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from mvcs_assistant.config.settings import settings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=settings.google_api_key)


def get_vectorstore(collection_name: str = "mvcs_docs") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(),
    )
