import hashlib
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def enrich_metadata(doc: Document) -> Document:
    src = doc.metadata.get("source", "unknown")
    raw = doc.page_content.strip()
    doc.metadata["doc_hash"] = hashlib.md5(raw.encode("utf-8")).hexdigest()
    doc.metadata["entity"] = "MVCS"
    doc.metadata["source"] = src
    doc.page_content = clean_text(raw)
    return doc


def deduplicate_docs(docs: list[Document]) -> list[Document]:
    seen = set()
    out = []
    for d in docs:
        key = d.metadata.get("doc_hash")
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
