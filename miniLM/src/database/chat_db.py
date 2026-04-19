"""
Chat database operations for MiniChat application.
Provides CRUD operations for messages, conversations, and agents.
"""
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict

from miniLM.src.database.init_db import get_connection, init_database
from miniLM.src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """Represents a chat message."""
    id: Optional[int]
    conversation_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime
    agent_id: Optional[int] = None


@dataclass
class Conversation:
    """Represents a chat conversation."""
    id: str
    title: str
    created_at: datetime
    agent_id: Optional[int] = None


@dataclass
class Agent:
    """Represents a custom agent with system prompt."""
    id: Optional[int]
    name: str
    description: str
    system_prompt: str
    created_at: datetime
    updated_at: datetime


# Prebuilt Equity Research Analyzer agent
EQUITY_RESEARCH_AGENT = Agent(
    id=None,
    name="Equity Research Analyzer",
    description="Analyzes investment documents such as quarterly fund letters and summarizes what each document says about referenced companies.",
    system_prompt="""You are an equity research assistant.
When given one or more documents, your task is to:
1. Identify all companies discussed in the documents.
2. Normalize company names and avoid duplicates.
3. For each document, summarize what is said about each company, focusing on:
   - Investment thesis or reasoning
   - Positive and negative views
   - Changes in position or conviction
   - Forward-looking commentary
4. Keep all summaries grounded strictly in the document content.
5. Clearly indicate which document each summary comes from.

If a company is only mentioned briefly, note it as a passing reference.""",
    created_at=datetime.now(),
    updated_at=datetime.now()
)


class ChatDatabase:
    """Database operations for chat functionality."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize ChatDatabase.
        
        Args:
            db_path: Optional path to the database file.
        """
        self.db_path = db_path
        # Ensure database is initialized
        init_database(db_path)
        # Add prebuilt agent if not exists
        self._ensure_prebuilt_agents()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        return get_connection(self.db_path)
    
    def _ensure_prebuilt_agents(self) -> None:
        """Ensure prebuilt agents exist in the database."""
        existing = self.get_agent_by_name(EQUITY_RESEARCH_AGENT.name)
        if existing is None:
            logger.info("Adding prebuilt Equity Research Analyzer agent...")
            self.create_agent(EQUITY_RESEARCH_AGENT)

    
    # ==================== Message Operations ====================
    
    def save_message(self, message: Message) -> int:
        """
        Save a message to the database.
        
        Args:
            message: Message object to save.
        
        Returns:
            int: The ID of the saved message.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (conversation_id, role, content, timestamp, agent_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.timestamp.isoformat(),
                    message.agent_id
                )
            )
            conn.commit()
            message_id = cursor.lastrowid
            logger.debug(f"Saved message {message_id} to conversation {message.conversation_id}")
            return message_id
        finally:
            conn.close()
    
    def get_messages(self, conversation_id: str) -> List[Message]:
        """
        Get all messages for a conversation.
        
        Args:
            conversation_id: The conversation ID.
        
        Returns:
            List[Message]: List of messages ordered by timestamp.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, conversation_id, role, content, timestamp, agent_id
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                """,
                (conversation_id,)
            )
            rows = cursor.fetchall()
            return [
                Message(
                    id=row['id'],
                    conversation_id=row['conversation_id'],
                    role=row['role'],
                    content=row['content'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    agent_id=row['agent_id']
                )
                for row in rows
            ]
        finally:
            conn.close()
    
    def delete_last_message(self, conversation_id: str) -> bool:
        """
        Delete the last message in a conversation.
        
        Args:
            conversation_id: The conversation ID.
        
        Returns:
            bool: True if a message was deleted, False otherwise.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            # Find the last message
            cursor.execute(
                """
                SELECT id FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (conversation_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return False
            
            # Delete it
            cursor.execute("DELETE FROM messages WHERE id = ?", (row['id'],))
            conn.commit()
            logger.debug(f"Deleted last message {row['id']} from conversation {conversation_id}")
            return True
        finally:
            conn.close()

    
    # ==================== Conversation Operations ====================
    
    def create_conversation(self, conversation: Conversation) -> str:
        """
        Create a new conversation.
        
        Args:
            conversation: Conversation object to create.
        
        Returns:
            str: The conversation ID.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversations (id, title, created_at, agent_id)
                VALUES (?, ?, ?, ?)
                """,
                (
                    conversation.id,
                    conversation.title,
                    conversation.created_at.isoformat(),
                    conversation.agent_id
                )
            )
            conn.commit()
            logger.debug(f"Created conversation {conversation.id}")
            return conversation.id
        finally:
            conn.close()
    
    def get_conversations(self) -> List[Conversation]:
        """
        Get all conversations.
        
        Returns:
            List[Conversation]: List of conversations ordered by creation date (newest first).
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, created_at, agent_id
                FROM conversations
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()
            return [
                Conversation(
                    id=row['id'],
                    title=row['title'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    agent_id=row['agent_id']
                )
                for row in rows
            ]
        finally:
            conn.close()
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get a specific conversation by ID.
        
        Args:
            conversation_id: The conversation ID.
        
        Returns:
            Optional[Conversation]: The conversation if found, None otherwise.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, created_at, agent_id
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return Conversation(
                id=row['id'],
                title=row['title'],
                created_at=datetime.fromisoformat(row['created_at']),
                agent_id=row['agent_id']
            )
        finally:
            conn.close()

    
    # ==================== Agent Operations ====================
    
    def create_agent(self, agent: Agent) -> int:
        """
        Create a new agent.
        
        Args:
            agent: Agent object to create.
        
        Returns:
            int: The ID of the created agent.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            now = datetime.now()
            cursor.execute(
                """
                INSERT INTO agents (name, description, system_prompt, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    agent.name,
                    agent.description,
                    agent.system_prompt,
                    now.isoformat(),
                    now.isoformat()
                )
            )
            conn.commit()
            agent_id = cursor.lastrowid
            logger.debug(f"Created agent {agent_id}: {agent.name}")
            return agent_id
        finally:
            conn.close()
    
    def get_agents(self) -> List[Agent]:
        """
        Get all agents.
        
        Returns:
            List[Agent]: List of all agents.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, description, system_prompt, created_at, updated_at
                FROM agents
                ORDER BY name ASC
                """
            )
            rows = cursor.fetchall()
            return [
                Agent(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    system_prompt=row['system_prompt'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                for row in rows
            ]
        finally:
            conn.close()
    
    def get_agent(self, agent_id: int) -> Optional[Agent]:
        """
        Get a specific agent by ID.
        
        Args:
            agent_id: The agent ID.
        
        Returns:
            Optional[Agent]: The agent if found, None otherwise.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, description, system_prompt, created_at, updated_at
                FROM agents
                WHERE id = ?
                """,
                (agent_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return Agent(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                system_prompt=row['system_prompt'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            )
        finally:
            conn.close()
    
    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """
        Get a specific agent by name.
        
        Args:
            name: The agent name.
        
        Returns:
            Optional[Agent]: The agent if found, None otherwise.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, description, system_prompt, created_at, updated_at
                FROM agents
                WHERE name = ?
                """,
                (name,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return Agent(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                system_prompt=row['system_prompt'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            )
        finally:
            conn.close()

    
    def update_agent(self, agent: Agent) -> bool:
        """
        Update an existing agent.
        
        Args:
            agent: Agent object with updated values. Must have a valid ID.
        
        Returns:
            bool: True if the agent was updated, False if not found.
        """
        if agent.id is None:
            raise ValueError("Agent ID is required for update")
        
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            now = datetime.now()
            cursor.execute(
                """
                UPDATE agents
                SET name = ?, description = ?, system_prompt = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    agent.name,
                    agent.description,
                    agent.system_prompt,
                    now.isoformat(),
                    agent.id
                )
            )
            conn.commit()
            updated = cursor.rowcount > 0
            if updated:
                logger.debug(f"Updated agent {agent.id}: {agent.name}")
            return updated
        finally:
            conn.close()
    
    def delete_agent(self, agent_id: int) -> bool:
        """
        Delete an agent.
        
        Args:
            agent_id: The ID of the agent to delete.
        
        Returns:
            bool: True if the agent was deleted, False if not found.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug(f"Deleted agent {agent_id}")
            return deleted
        finally:
            conn.close()
