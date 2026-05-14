from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from mvcs_assistant.config.settings import settings


class GeminiV1Embeddings(Embeddings):
    """Embeddings usando google-genai SDK con api_version=v1 para text-embedding-004."""

    def __init__(self, model: str, api_key: str):
        from google import genai
        self._model = model
        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1"},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = []
        for text in texts:
            resp = self._client.models.embed_content(model=self._model, contents=[text])
            result.append(list(resp.embeddings[0].values))
        return result

    def embed_query(self, text: str) -> List[float]:
        resp = self._client.models.embed_content(model=self._model, contents=[text])
        return list(resp.embeddings[0].values)


def get_embeddings() -> GeminiV1Embeddings:
    return GeminiV1Embeddings(model=settings.embedding_model, api_key=settings.google_api_key)


def get_vectorstore(collection_name: str = "mvcs_docs") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(),
    )
