"""
Database initialization module for MiniChat application.
Creates SQLite database with required tables for conversations, messages, and agents.
"""
import sqlite3
from pathlib import Path
from typing import Optional

from miniLM.config.settings import get_settings
from miniLM.src.utils.logger import get_logger

logger = get_logger(__name__)

# SQL statements for table creation
CREATE_AGENTS_TABLE = """
CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    system_prompt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_id INTEGER REFERENCES agents(id)
);
"""

CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_id INTEGER REFERENCES agents(id)
);
"""

# Index creation for performance
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);",
    "CREATE INDEX IF NOT EXISTS idx_conversations_agent ON conversations(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);",
]


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get a database connection.
    
    Args:
        db_path: Optional path to the database file. If not provided,
                 uses the path from settings.
    
    Returns:
        sqlite3.Connection: Database connection with row factory set.
    
    Raises:
        sqlite3.Error: If connection fails.
    """
    if db_path is None:
        settings = get_settings()
        db_path = settings.get_absolute_sqlite_path()
    
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Enable foreign key support
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_database(db_path: Optional[Path] = None) -> None:
    """
    Initialize the SQLite database with required tables.
    
    Creates the conversations, messages, and agents tables if they don't exist,
    along with necessary indexes for performance.
    
    Args:
        db_path: Optional path to the database file. If not provided,
                 uses the path from settings.
    
    Raises:
        sqlite3.Error: If database initialization fails.
    """
    conn = None
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        
        # Create tables in order (agents first due to foreign key references)
        logger.info("Creating agents table...")
        cursor.execute(CREATE_AGENTS_TABLE)
        
        logger.info("Creating conversations table...")
        cursor.execute(CREATE_CONVERSATIONS_TABLE)
        
        logger.info("Creating messages table...")
        cursor.execute(CREATE_MESSAGES_TABLE)
        
        # Create indexes
        logger.info("Creating indexes...")
        for index_sql in CREATE_INDEXES:
            cursor.execute(index_sql)
        
        conn.commit()
        logger.info("Database initialization completed successfully.")
        
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        if conn:
            conn.close()


def verify_database(db_path: Optional[Path] = None) -> bool:
    """
    Verify that the database has all required tables.
    
    Args:
        db_path: Optional path to the database file.
    
    Returns:
        bool: True if all required tables exist, False otherwise.
    """
    required_tables = {'agents', 'conversations', 'messages'}
    
    conn = None
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
        existing_tables = {row['name'] for row in cursor.fetchall()}
        
        return required_tables.issubset(existing_tables)
        
    except sqlite3.Error as e:
        logger.error(f"Database verification failed: {e}")
        return False
    finally:
        if conn:
            conn.close()
