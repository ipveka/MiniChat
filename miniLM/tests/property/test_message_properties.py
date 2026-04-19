"""
Property-based tests for message persistence.
"""
import tempfile
from datetime import datetime
from pathlib import Path

from hypothesis import given, strategies as st, settings

from miniLM.src.database.chat_db import ChatDatabase, Message, Conversation
from miniLM.src.utils.helpers import generate_uuid


# Strategy for valid message roles
role_strategy = st.sampled_from(['user', 'assistant', 'system'])

# Strategy for non-empty text content
content_strategy = st.text(min_size=1, max_size=1000).filter(lambda x: x.strip())


@settings(max_examples=100, deadline=None)
@given(
    role=role_strategy,
    content=content_strategy
)
def test_message_persistence_round_trip(role: str, content: str):
    """
    **Feature: minichat-app, Property 1: Message Persistence Round-Trip**
    
    For any valid Message object with conversation_id, role, content, and timestamp,
    saving it to SQLite and then retrieving messages for that conversation_id
    should return a message with equivalent properties.
    
    **Validates: Requirements 1.2, 1.5, 1.6**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = ChatDatabase(db_path=db_path)
        
        # Create a conversation first (required for foreign key)
        conversation_id = generate_uuid()
        conversation = Conversation(
            id=conversation_id,
            title="Test Conversation",
            created_at=datetime.now(),
            agent_id=None
        )
        db.create_conversation(conversation)
        
        # Create and save a message
        timestamp = datetime.now()
        message = Message(
            id=None,
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=timestamp,
            agent_id=None
        )
        
        saved_id = db.save_message(message)
        
        # Retrieve messages for the conversation
        retrieved_messages = db.get_messages(conversation_id)
        
        # Verify round-trip
        assert len(retrieved_messages) >= 1, "Should have at least one message"
        
        # Find our message (there might be others from setup)
        matching = [m for m in retrieved_messages if m.id == saved_id]
        assert len(matching) == 1, "Should find exactly one matching message"
        
        retrieved = matching[0]
        assert retrieved.conversation_id == conversation_id
        assert retrieved.role == role
        assert retrieved.content == content
        # Timestamp comparison with tolerance for serialization
        assert abs((retrieved.timestamp - timestamp).total_seconds()) < 1
