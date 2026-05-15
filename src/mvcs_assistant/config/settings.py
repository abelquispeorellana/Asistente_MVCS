from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "./data/chroma"))
    raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", "./data"))
    processed_data_dir: Path = Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    chunk_size: int = 1200
    chunk_overlap: int = 180
    k_retrieval: int = 4
    score_threshold: float = 0.35


settings = Settings()
