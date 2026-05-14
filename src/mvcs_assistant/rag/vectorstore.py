from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from mvcs_assistant.config.settings import settings

_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=_EMBED_MODEL)


def get_vectorstore(collection_name: str = "mvcs_docs") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(),
    )
