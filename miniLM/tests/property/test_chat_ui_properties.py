"""
Property-based tests for chat UI functionality.
"""
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

from hypothesis import given, strategies as st, settings, assume

from miniLM.src.database.chat_db import ChatDatabase, Message, Conversation
from miniLM.src.utils.helpers import generate_uuid


# Strategy for valid message roles
role_strategy = st.sampled_from(['user', 'assistant'])

# Strategy for non-empty text content
content_strategy = st.text(min_size=1, max_size=500).filter(lambda x: x.strip())


def create_test_messages(
    db: ChatDatabase,
    conversation_id: str,
    message_specs: List[tuple]
) -> List[Message]:
    """
    Helper to create test messages in the database.
    
    Args:
        db: ChatDatabase instance.
        conversation_id: Conversation ID.
        message_specs: List of (role, content) tuples.
    
    Returns:
        List of created Message objects with IDs.
    """
    messages = []
    for role, content in message_specs:
        msg = Message(
            id=None,
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            agent_id=None
        )
        msg_id = db.save_message(msg)
        msg.id = msg_id
        messages.append(msg)
    return messages


@settings(max_examples=100, deadline=None)
@given(
    user_contents=st.lists(content_strategy, min_size=1, max_size=5),
    assistant_contents=st.lists(content_strategy, min_size=1, max_size=5)
)
def test_regenerate_replaces_last_response(
    user_contents: List[str],
    assistant_contents: List[str]
):
    """
    **Feature: minichat-app, Property 12: Regenerate Replaces Last Response**
    
    For any conversation with at least one assistant message, regenerating
    should result in the same number of messages but with a potentially
    different last assistant message content.
    
    This test validates the database-level behavior of delete_last_message,
    which is the core operation used by handle_regenerate.
    
    **Validates: Requirements 1.3**
    """
    # Ensure we have at least one of each
    assume(len(user_contents) >= 1)
    assume(len(assistant_contents) >= 1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = ChatDatabase(db_path=db_path)
        
        # Create a conversation
        conversation_id = generate_uuid()
        conversation = Conversation(
            id=conversation_id,
            title="Test Conversation",
            created_at=datetime.now(),
            agent_id=None
        )
        db.create_conversation(conversation)
        
        # Create alternating user/assistant messages
        message_specs = []
        for i in range(min(len(user_contents), len(assistant_contents))):
            message_specs.append(("user", user_contents[i]))
            message_specs.append(("assistant", assistant_contents[i]))
        
        messages = create_test_messages(db, conversation_id, message_specs)
        initial_count = len(messages)
        
        # Get messages before regenerate
        messages_before = db.get_messages(conversation_id)
        assert len(messages_before) == initial_count
        
        # Find the last assistant message
        last_assistant = None
        for msg in reversed(messages_before):
            if msg.role == "assistant":
                last_assistant = msg
                break
        
        assert last_assistant is not None, "Should have at least one assistant message"
        
        # Simulate regenerate: delete last message
        deleted = db.delete_last_message(conversation_id)
        assert deleted, "Should successfully delete last message"
        
        # Get messages after deletion
        messages_after = db.get_messages(conversation_id)
        
        # Should have one fewer message
        assert len(messages_after) == initial_count - 1
        
        # The deleted message should not be in the list
        remaining_ids = [m.id for m in messages_after]
        assert last_assistant.id not in remaining_ids
        
        # All other messages should still be present
        for msg in messages_before:
            if msg.id != last_assistant.id:
                assert msg.id in remaining_ids


@settings(max_examples=100, deadline=None)
@given(
    user_content=content_strategy,
    assistant_content=content_strategy,
    new_assistant_content=content_strategy
)
def test_regenerate_allows_new_response(
    user_content: str,
    assistant_content: str,
    new_assistant_content: str
):
    """
    **Feature: minichat-app, Property 12: Regenerate Replaces Last Response (Part 2)**
    
    After deleting the last assistant message, a new assistant message
    can be added, resulting in the same total message count but with
    different content for the last assistant message.
    
    **Validates: Requirements 1.3**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = ChatDatabase(db_path=db_path)
        
        # Create a conversation
        conversation_id = generate_uuid()
        conversation = Conversation(
            id=conversation_id,
            title="Test Conversation",
            created_at=datetime.now(),
            agent_id=None
        )
        db.create_conversation(conversation)
        
        # Create user message and assistant response
        message_specs = [
            ("user", user_content),
            ("assistant", assistant_content)
        ]
        create_test_messages(db, conversation_id, message_specs)
        
        # Get initial state
        messages_before = db.get_messages(conversation_id)
        assert len(messages_before) == 2
        
        original_assistant_content = messages_before[1].content
        assert original_assistant_content == assistant_content
        
        # Delete last message (simulating regenerate)
        db.delete_last_message(conversation_id)
        
        # Add new assistant message (simulating new generation)
        new_msg = Message(
            id=None,
            conversation_id=conversation_id,
            role="assistant",
            content=new_assistant_content,
            timestamp=datetime.now(),
            agent_id=None
        )
        db.save_message(new_msg)
        
        # Get final state
        messages_after = db.get_messages(conversation_id)
        
        # Should have same count
        assert len(messages_after) == 2
        
        # User message should be unchanged
        assert messages_after[0].role == "user"
        assert messages_after[0].content == user_content
        
        # Assistant message should have new content
        assert messages_after[1].role == "assistant"
        assert messages_after[1].content == new_assistant_content



from miniLM.src.ui.chat import build_chat_context
from miniLM.src.llm.ollama_client import ChatMessage


# Strategy for system prompts
system_prompt_strategy = st.text(min_size=1, max_size=500).filter(lambda x: x.strip())


@settings(max_examples=100, deadline=None)
@given(
    system_prompt=system_prompt_strategy,
    user_contents=st.lists(content_strategy, min_size=1, max_size=5),
    assistant_contents=st.lists(content_strategy, min_size=0, max_size=5)
)
def test_agent_system_prompt_applied_to_context(
    system_prompt: str,
    user_contents: List[str],
    assistant_contents: List[str]
):
    """
    **Feature: minichat-app, Property 11: Agent System Prompt Applied to Context**
    
    For any selected Agent with a system_prompt, when generating a chat or RAG
    response, the system_prompt should be included in the messages sent to Ollama.
    
    **Validates: Requirements 1.4, 2.4**
    """
    # Build messages list
    messages = []
    for i, content in enumerate(user_contents):
        messages.append(Message(
            id=i * 2,
            conversation_id="test",
            role="user",
            content=content,
            timestamp=datetime.now(),
            agent_id=None
        ))
        if i < len(assistant_contents):
            messages.append(Message(
                id=i * 2 + 1,
                conversation_id="test",
                role="assistant",
                content=assistant_contents[i],
                timestamp=datetime.now(),
                agent_id=None
            ))
    
    # Build context with system prompt
    chat_context = build_chat_context(messages, system_prompt=system_prompt)
    
    # Verify system prompt is included as first message
    assert len(chat_context) >= 1, "Should have at least the system message"
    assert chat_context[0].role == "system"
    assert chat_context[0].content == system_prompt
    
    # Verify all user/assistant messages are included after system prompt
    non_system_messages = chat_context[1:]
    expected_count = len(user_contents) + min(len(assistant_contents), len(user_contents))
    assert len(non_system_messages) == expected_count


@settings(max_examples=100, deadline=None)
@given(
    user_contents=st.lists(content_strategy, min_size=1, max_size=5),
    assistant_contents=st.lists(content_strategy, min_size=0, max_size=5)
)
def test_no_system_prompt_when_none_selected(
    user_contents: List[str],
    assistant_contents: List[str]
):
    """
    **Feature: minichat-app, Property 11: Agent System Prompt Applied to Context (Part 2)**
    
    When no agent is selected (system_prompt is None), the context should not
    include a system message.
    
    **Validates: Requirements 1.4, 2.4**
    """
    # Build messages list
    messages = []
    for i, content in enumerate(user_contents):
        messages.append(Message(
            id=i * 2,
            conversation_id="test",
            role="user",
            content=content,
            timestamp=datetime.now(),
            agent_id=None
        ))
        if i < len(assistant_contents):
            messages.append(Message(
                id=i * 2 + 1,
                conversation_id="test",
                role="assistant",
                content=assistant_contents[i],
                timestamp=datetime.now(),
                agent_id=None
            ))
    
    # Build context without system prompt
    chat_context = build_chat_context(messages, system_prompt=None)
    
    # Verify no system message is included
    for msg in chat_context:
        assert msg.role != "system", "Should not have system message when no agent selected"
    
    # Verify all user/assistant messages are included
    expected_count = len(user_contents) + min(len(assistant_contents), len(user_contents))
    assert len(chat_context) == expected_count



# Strategy for conversation titles
title_strategy = st.text(min_size=1, max_size=100).filter(lambda x: x.strip())


@settings(max_examples=100, deadline=None)
@given(
    titles=st.lists(title_strategy, min_size=1, max_size=10, unique=True)
)
def test_conversation_list_contains_all_conversations(titles: List[str]):
    """
    **Feature: minichat-app, Property 13: Conversation List Contains All Conversations**
    
    For any set of created conversations, retrieving the conversation list
    should return all created conversations.
    
    **Validates: Requirements 3.2**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = ChatDatabase(db_path=db_path)
        
        # Create conversations
        created_ids = []
        for title in titles:
            conversation_id = generate_uuid()
            conversation = Conversation(
                id=conversation_id,
                title=title,
                created_at=datetime.now(),
                agent_id=None
            )
            db.create_conversation(conversation)
            created_ids.append(conversation_id)
        
        # Retrieve all conversations
        retrieved = db.get_conversations()
        retrieved_ids = [c.id for c in retrieved]
        
        # Verify all created conversations are in the list
        for conv_id in created_ids:
            assert conv_id in retrieved_ids, f"Conversation {conv_id} should be in the list"
        
        # Verify count matches
        assert len(retrieved) >= len(created_ids), "Should have at least as many conversations as created"


@settings(max_examples=100, deadline=None)
@given(
    title=title_strategy
)
def test_conversation_retrievable_by_id(title: str):
    """
    **Feature: minichat-app, Property 13: Conversation List Contains All Conversations (Part 2)**
    
    For any created conversation, it should be retrievable by its ID.
    
    **Validates: Requirements 3.2**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = ChatDatabase(db_path=db_path)
        
        # Create a conversation
        conversation_id = generate_uuid()
        conversation = Conversation(
            id=conversation_id,
            title=title,
            created_at=datetime.now(),
            agent_id=None
        )
        db.create_conversation(conversation)
        
        # Retrieve by ID
        retrieved = db.get_conversation(conversation_id)
        
        # Verify retrieval
        assert retrieved is not None, "Should retrieve the conversation"
        assert retrieved.id == conversation_id
        assert retrieved.title == title
