"""
Chat UI components for MiniChat application.
Provides the chat interface with message display, input, and agent selection.
"""
from datetime import datetime
from typing import List, Optional

import streamlit as st

from miniLM.src.database.chat_db import ChatDatabase, Message, Conversation, Agent
from miniLM.src.llm.ollama_client import OllamaClient, ChatMessage, OllamaConnectionError, ModelNotFoundError
from miniLM.src.utils.helpers import generate_uuid, format_timestamp


def render_chat_page(
    chat_db: ChatDatabase,
    ollama_client: OllamaClient,
    agents: List[Agent]
) -> None:
    """
    Render the main chat page.
    
    Args:
        chat_db: ChatDatabase instance for message persistence.
        ollama_client: OllamaClient for LLM inference.
        agents: List of available agents.
    """
    st.header("💬 Chat")
    
    # Initialize session state
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_agent_id" not in st.session_state:
        st.session_state.selected_agent_id = None
    
    # Sidebar for conversation management
    with st.sidebar:
        st.subheader("Conversations")
        
        # New conversation button
        if st.button("➕ New Conversation", use_container_width=True):
            _create_new_conversation(chat_db)
        
        st.divider()
        
        # Agent selection dropdown
        agent_options = {"None": None}
        agent_options.update({agent.name: agent.id for agent in agents})
        
        selected_agent_name = st.selectbox(
            "Select Agent",
            options=list(agent_options.keys()),
            index=0,
            help="Select an agent to apply its system prompt to the conversation"
        )
        st.session_state.selected_agent_id = agent_options[selected_agent_name]
        
        st.divider()
        
        # List existing conversations
        conversations = chat_db.get_conversations()
        for conv in conversations:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    f"📝 {conv.title[:30]}...",
                    key=f"conv_{conv.id}",
                    use_container_width=True
                ):
                    _load_conversation(chat_db, conv.id)
            with col2:
                pass  # Could add delete button here
    
    # Main chat area
    if st.session_state.current_conversation_id is None:
        st.info("Start a new conversation or select an existing one from the sidebar.")
        return
    
    # Display messages
    for message in st.session_state.messages:
        render_message(message)
    
    # Regenerate button (only if there's at least one assistant message)
    assistant_messages = [m for m in st.session_state.messages if m.role == "assistant"]
    if assistant_messages:
        if st.button("🔄 Regenerate Last Response"):
            handle_regenerate(
                st.session_state.current_conversation_id,
                chat_db,
                ollama_client,
                agents
            )
            st.rerun()
    
    # Chat input
    user_input = render_chat_input()
    if user_input:
        _handle_user_message(
            user_input,
            chat_db,
            ollama_client,
            agents
        )


def render_message(message: Message) -> None:
    """
    Render a single chat message.
    
    Args:
        message: Message object to display.
    """
    if message.role == "user":
        with st.chat_message("user"):
            st.markdown(message.content)
            st.caption(format_timestamp(message.timestamp))
    elif message.role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message.content)
            st.caption(format_timestamp(message.timestamp))
    elif message.role == "system":
        with st.chat_message("assistant", avatar="⚙️"):
            st.markdown(f"*System: {message.content}*")


def render_chat_input() -> Optional[str]:
    """
    Render the chat input field.
    
    Returns:
        User input string if provided, None otherwise.
    """
    return st.chat_input("Type your message...")



def handle_regenerate(
    conversation_id: str,
    chat_db: ChatDatabase,
    ollama_client: OllamaClient,
    agents: List[Agent]
) -> None:
    """
    Handle regeneration of the last assistant response.
    
    Removes the last assistant message and generates a new response
    using the same conversation context.
    
    Args:
        conversation_id: The conversation ID.
        chat_db: ChatDatabase instance.
        ollama_client: OllamaClient for LLM inference.
        agents: List of available agents.
    """
    # Delete the last message (should be assistant)
    messages = st.session_state.messages
    if not messages:
        return
    
    # Find and remove the last assistant message
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            last_assistant_idx = i
            break
    
    if last_assistant_idx is None:
        return
    
    # Delete from database
    chat_db.delete_last_message(conversation_id)
    
    # Remove from session state
    st.session_state.messages = messages[:last_assistant_idx]
    
    # Get the last user message to regenerate response
    user_messages = [m for m in st.session_state.messages if m.role == "user"]
    if not user_messages:
        return
    
    # Generate new response
    _generate_response(
        chat_db,
        ollama_client,
        agents,
        conversation_id
    )


def _create_new_conversation(chat_db: ChatDatabase) -> None:
    """Create a new conversation and set it as current."""
    conversation_id = generate_uuid()
    conversation = Conversation(
        id=conversation_id,
        title=f"New Chat {datetime.now().strftime('%H:%M')}",
        created_at=datetime.now(),
        agent_id=st.session_state.get("selected_agent_id")
    )
    chat_db.create_conversation(conversation)
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages = []


def _load_conversation(chat_db: ChatDatabase, conversation_id: str) -> None:
    """Load an existing conversation."""
    st.session_state.current_conversation_id = conversation_id
    messages = chat_db.get_messages(conversation_id)
    st.session_state.messages = messages
    
    # Load agent if conversation has one
    conversation = chat_db.get_conversation(conversation_id)
    if conversation and conversation.agent_id:
        st.session_state.selected_agent_id = conversation.agent_id


def _handle_user_message(
    user_input: str,
    chat_db: ChatDatabase,
    ollama_client: OllamaClient,
    agents: List[Agent]
) -> None:
    """Handle a new user message."""
    conversation_id = st.session_state.current_conversation_id
    
    # Create and save user message
    user_message = Message(
        id=None,
        conversation_id=conversation_id,
        role="user",
        content=user_input,
        timestamp=datetime.now(),
        agent_id=st.session_state.selected_agent_id
    )
    chat_db.save_message(user_message)
    st.session_state.messages.append(user_message)
    
    # Display user message
    render_message(user_message)
    
    # Generate and display assistant response
    _generate_response(chat_db, ollama_client, agents, conversation_id)


def _generate_response(
    chat_db: ChatDatabase,
    ollama_client: OllamaClient,
    agents: List[Agent],
    conversation_id: str
) -> None:
    """Generate and display an assistant response with streaming."""
    # Get system prompt from selected agent
    system_prompt = None
    agent_id = st.session_state.selected_agent_id
    if agent_id:
        agent = next((a for a in agents if a.id == agent_id), None)
        if agent:
            system_prompt = agent.system_prompt
    
    # Build chat messages for Ollama
    chat_messages = [
        ChatMessage(role=m.role, content=m.content)
        for m in st.session_state.messages
        if m.role in ("user", "assistant")
    ]
    
    # Stream the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk in ollama_client.chat(chat_messages, system_prompt=system_prompt):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
                if chunk.done:
                    break
            
            message_placeholder.markdown(full_response)
            
        except (OllamaConnectionError, ModelNotFoundError) as e:
            st.error(e.user_message)
            return
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return
    
    # Save assistant message
    assistant_message = Message(
        id=None,
        conversation_id=conversation_id,
        role="assistant",
        content=full_response,
        timestamp=datetime.now(),
        agent_id=agent_id
    )
    chat_db.save_message(assistant_message)
    st.session_state.messages.append(assistant_message)


def build_chat_context(
    messages: List[Message],
    system_prompt: Optional[str] = None
) -> List[ChatMessage]:
    """
    Build chat context for Ollama from messages and optional system prompt.
    
    This function is exposed for testing Property 11: Agent System Prompt Applied to Context.
    
    Args:
        messages: List of Message objects from the conversation.
        system_prompt: Optional system prompt from selected agent.
    
    Returns:
        List of ChatMessage objects ready for Ollama.
    """
    chat_messages = []
    
    # Add system prompt if provided
    if system_prompt:
        chat_messages.append(ChatMessage(role="system", content=system_prompt))
    
    # Add conversation messages
    for m in messages:
        if m.role in ("user", "assistant"):
            chat_messages.append(ChatMessage(role=m.role, content=m.content))
    
    return chat_messages
