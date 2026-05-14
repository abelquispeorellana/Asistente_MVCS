from mvcs_assistant.config.settings import settings
from mvcs_assistant.ingestion.loaders import load_documents
from mvcs_assistant.ingestion.preprocess import enrich_metadata, deduplicate_docs, chunk_documents
from mvcs_assistant.rag.vectorstore import get_vectorstore
from mvcs_assistant.utils.logger import setup_logger


def run_indexing() -> None:
    logger = setup_logger("indexer", level=settings.log_level)
    docs = load_documents(settings.raw_data_dir)
    logger.info("Documentos cargados: %s", len(docs))

    docs = [enrich_metadata(d) for d in docs]
    docs = deduplicate_docs(docs)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)

    vs = get_vectorstore()
    vs.add_documents(chunks)
    logger.info("Chunks indexados: %s", len(chunks))


if __name__ == "__main__":
    run_indexing()
