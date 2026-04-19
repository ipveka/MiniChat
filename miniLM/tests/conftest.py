"""
Pytest configuration and fixtures for MiniChat tests.
"""
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from miniLM.src.database.init_db import init_database
from miniLM.src.database.chat_db import ChatDatabase


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path


@pytest.fixture
def chat_db(temp_db_path: Path) -> Generator[ChatDatabase, None, None]:
    """Create a ChatDatabase instance with a temporary database."""
    db = ChatDatabase(db_path=temp_db_path)
    yield db
