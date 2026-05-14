from langchain.schema import Document
from mvcs_assistant.ingestion.preprocess import clean_text, enrich_metadata, deduplicate_docs


def test_clean_text():
    assert clean_text("hola   mundo\n") == "hola mundo"


def test_deduplicate_docs():
    d1 = enrich_metadata(Document(page_content="abc", metadata={"source": "a"}))
    d2 = enrich_metadata(Document(page_content="abc", metadata={"source": "b"}))
    out = deduplicate_docs([d1, d2])
    assert len(out) == 1
