from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredHTMLLoader
from langchain_core.documents import Document


def load_documents(input_dir: Path) -> list[Document]:
    docs = []
    for path in input_dir.rglob("*"):
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
        elif path.suffix.lower() == ".csv":
            docs.extend(CSVLoader(str(path), encoding="utf-8").load())
        elif path.suffix.lower() in {".html", ".htm"}:
            docs.extend(UnstructuredHTMLLoader(str(path)).load())
    return docs
