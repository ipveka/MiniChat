"""
Configuration settings for MiniChat application.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class Settings:
    """Application configuration settings."""
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2"
    
    # Database settings
    sqlite_db_path: str = "data/minichat.db"
    chroma_persist_dir: str = "data/chroma"
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Logging settings
    log_dir: str = "logs"
    log_level: str = "INFO"
    
    def get_absolute_sqlite_path(self, base_dir: Optional[Path] = None) -> Path:
        """Get absolute path for SQLite database."""
        base = base_dir or Path(__file__).parent.parent
        return base / self.sqlite_db_path
    
    def get_absolute_chroma_path(self, base_dir: Optional[Path] = None) -> Path:
        """Get absolute path for ChromaDB directory."""
        base = base_dir or Path(__file__).parent.parent
        return base / self.chroma_persist_dir
    
    def get_absolute_log_dir(self, base_dir: Optional[Path] = None) -> Path:
        """Get absolute path for logs directory."""
        base = base_dir or Path(__file__).parent.parent
        return base / self.log_dir


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def load_settings() -> Settings:
    """
    Load settings from environment variables or use defaults.
    
    Environment variables:
        MINICHAT_OLLAMA_URL: Ollama base URL
        MINICHAT_MODEL: Default LLM model
        MINICHAT_DB_PATH: SQLite database path
        MINICHAT_CHROMA_DIR: ChromaDB persist directory
        MINICHAT_EMBEDDING_MODEL: Sentence transformer model
        MINICHAT_CHUNK_SIZE: Document chunk size
        MINICHAT_CHUNK_OVERLAP: Chunk overlap size
        MINICHAT_LOG_DIR: Log directory
        MINICHAT_LOG_LEVEL: Logging level
    """
    return Settings(
        ollama_base_url=os.environ.get("MINICHAT_OLLAMA_URL", "http://localhost:11434"),
        default_model=os.environ.get("MINICHAT_MODEL", "llama3.2"),
        sqlite_db_path=os.environ.get("MINICHAT_DB_PATH", "data/minichat.db"),
        chroma_persist_dir=os.environ.get("MINICHAT_CHROMA_DIR", "data/chroma"),
        embedding_model=os.environ.get("MINICHAT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chunk_size=int(os.environ.get("MINICHAT_CHUNK_SIZE", "500")),
        chunk_overlap=int(os.environ.get("MINICHAT_CHUNK_OVERLAP", "50")),
        log_dir=os.environ.get("MINICHAT_LOG_DIR", "logs"),
        log_level=os.environ.get("MINICHAT_LOG_LEVEL", "INFO"),
    )
