from typing import List

import requests
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from mvcs_assistant.config.settings import settings


class GeminiV1Embeddings(Embeddings):
    """Llama directamente al endpoint v1 (no v1beta) de la Generative Language API."""

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        self.api_key = api_key
        self.model = model
        self._url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
        )

    def _embed(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        resp = requests.post(
            self._url,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t, "retrieval_document") for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, "retrieval_query")


def get_embeddings() -> GeminiV1Embeddings:
    model = settings.embedding_model.removeprefix("models/")
    return GeminiV1Embeddings(api_key=settings.google_api_key, model=model)


def get_vectorstore(collection_name: str = "mvcs_docs") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(),
    )
