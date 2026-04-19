"""
Agents UI components for MiniChat application.
Provides the agent management interface for creating, editing, and deleting agents.
"""
from datetime import datetime
from typing import Optional

import streamlit as st

from miniLM.src.database.chat_db import ChatDatabase, Agent


def render_agents_page(chat_db: ChatDatabase) -> None:
    """
    Render the Agents management page.
    
    Args:
        chat_db: ChatDatabase instance for agent operations.
    """
    st.header("🤖 Agents")
    st.caption("Create and manage custom agents with system prompts")
    
    # Initialize session state
    if "editing_agent_id" not in st.session_state:
        st.session_state.editing_agent_id = None
    if "show_create_form" not in st.session_state:
        st.session_state.show_create_form = False
    if "delete_confirm_id" not in st.session_state:
        st.session_state.delete_confirm_id = None
    
    # Create new agent button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("➕ New Agent", type="primary", use_container_width=True):
            st.session_state.show_create_form = True
            st.session_state.editing_agent_id = None
    
    # Show create/edit form
    if st.session_state.show_create_form:
        st.divider()
        st.subheader("Create New Agent")
        new_agent = render_agent_form()
        if new_agent:
            try:
                chat_db.create_agent(new_agent)
                st.success(f"✅ Agent '{new_agent.name}' created successfully!")
                st.session_state.show_create_form = False
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error creating agent: {str(e)}")
        
        if st.button("Cancel", key="cancel_create"):
            st.session_state.show_create_form = False
            st.rerun()
    
    elif st.session_state.editing_agent_id is not None:
        st.divider()
        agent = chat_db.get_agent(st.session_state.editing_agent_id)
        if agent:
            st.subheader(f"Edit Agent: {agent.name}")
            updated_agent = render_agent_form(agent)
            if updated_agent:
                try:
                    chat_db.update_agent(updated_agent)
                    st.success(f"✅ Agent '{updated_agent.name}' updated successfully!")
                    st.session_state.editing_agent_id = None
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error updating agent: {str(e)}")
            
            if st.button("Cancel", key="cancel_edit"):
                st.session_state.editing_agent_id = None
                st.rerun()
    
    # List existing agents
    st.divider()
    st.subheader("Your Agents")
    
    agents = chat_db.get_agents()
    render_agent_list(agents, chat_db)


def render_agent_form(agent: Optional[Agent] = None) -> Optional[Agent]:
    """
    Render the agent create/edit form.
    
    Args:
        agent: Optional existing agent for editing. None for creating new.
    
    Returns:
        Agent object if form is submitted, None otherwise.
    """
    with st.form(key="agent_form"):
        name = st.text_input(
            "Agent Name",
            value=agent.name if agent else "",
            placeholder="e.g., Code Reviewer",
            help="A unique name for this agent"
        )
        
        description = st.text_area(
            "Description",
            value=agent.description if agent else "",
            placeholder="Describe what this agent does...",
            height=100,
            help="A brief description of the agent's purpose"
        )
        
        system_prompt = st.text_area(
            "System Prompt",
            value=agent.system_prompt if agent else "",
            placeholder="You are a helpful assistant that...",
            height=200,
            help="The system prompt that defines the agent's behavior"
        )
        
        submitted = st.form_submit_button(
            "Save Agent" if agent else "Create Agent",
            type="primary"
        )
        
        if submitted:
            # Validate inputs
            if not name or not name.strip():
                st.error("Agent name is required")
                return None
            if not system_prompt or not system_prompt.strip():
                st.error("System prompt is required")
                return None
            
            now = datetime.now()
            return Agent(
                id=agent.id if agent else None,
                name=name.strip(),
                description=description.strip() if description else "",
                system_prompt=system_prompt.strip(),
                created_at=agent.created_at if agent else now,
                updated_at=now
            )
    
    return None



def render_agent_list(agents: list, chat_db: ChatDatabase) -> None:
    """
    Render the list of agents with edit/delete actions.
    
    Args:
        agents: List of Agent objects to display.
        chat_db: ChatDatabase instance for agent operations.
    """
    if not agents:
        st.info("No agents created yet. Click 'New Agent' to create one.")
        return
    
    for agent in agents:
        with st.container():
            col1, col2, col3 = st.columns([4, 1, 1])
            
            with col1:
                st.markdown(f"### {agent.name}")
                st.write(agent.description if agent.description else "*No description*")
                
                with st.expander("View System Prompt"):
                    st.code(agent.system_prompt, language=None)
            
            with col2:
                if st.button("✏️ Edit", key=f"edit_{agent.id}", use_container_width=True):
                    st.session_state.editing_agent_id = agent.id
                    st.session_state.show_create_form = False
                    st.rerun()
            
            with col3:
                # Delete with confirmation
                if st.session_state.delete_confirm_id == agent.id:
                    if st.button("⚠️ Confirm", key=f"confirm_delete_{agent.id}", use_container_width=True):
                        try:
                            chat_db.delete_agent(agent.id)
                            st.success(f"Deleted agent '{agent.name}'")
                            st.session_state.delete_confirm_id = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting agent: {str(e)}")
                else:
                    if st.button("🗑️ Delete", key=f"delete_{agent.id}", use_container_width=True):
                        st.session_state.delete_confirm_id = agent.id
                        st.rerun()
            
            st.divider()
