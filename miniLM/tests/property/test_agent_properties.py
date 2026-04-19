"""
Property-based tests for agent persistence.
"""
import tempfile
from datetime import datetime
from pathlib import Path

from hypothesis import given, strategies as st, settings, assume

from miniLM.src.database.chat_db import ChatDatabase, Agent


# Strategy for valid agent names (non-empty, unique-friendly)
name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S'), whitelist_characters=' -_'),
    min_size=1,
    max_size=100
).filter(lambda x: x.strip())

# Strategy for description (can be empty)
description_strategy = st.text(min_size=0, max_size=500)

# Strategy for system prompt (non-empty)
system_prompt_strategy = st.text(min_size=1, max_size=2000).filter(lambda x: x.strip())


@settings(max_examples=100, deadline=None)
@given(
    name=name_strategy,
    description=description_strategy,
    system_prompt=system_prompt_strategy
)
def test_agent_crud_round_trip(name: str, description: str, system_prompt: str):
    """
    **Feature: minichat-app, Property 2: Agent CRUD Round-Trip**
    
    For any valid Agent object with name, description, and system_prompt,
    creating it in SQLite and then retrieving it by ID should return
    an agent with equivalent properties.
    
    **Validates: Requirements 3.1, 3.6**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = ChatDatabase(db_path=db_path)
        
        # Create an agent
        agent = Agent(
            id=None,
            name=name,
            description=description,
            system_prompt=system_prompt,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        agent_id = db.create_agent(agent)
        
        # Retrieve the agent by ID
        retrieved = db.get_agent(agent_id)
        
        # Verify round-trip
        assert retrieved is not None, "Agent should be retrievable"
        assert retrieved.id == agent_id
        assert retrieved.name == name
        assert retrieved.description == description
        assert retrieved.system_prompt == system_prompt



@settings(max_examples=100, deadline=None)
@given(
    original_name=name_strategy,
    original_description=description_strategy,
    original_prompt=system_prompt_strategy,
    updated_name=name_strategy,
    updated_description=description_strategy,
    updated_prompt=system_prompt_strategy
)
def test_agent_update_persistence(
    original_name: str,
    original_description: str,
    original_prompt: str,
    updated_name: str,
    updated_description: str,
    updated_prompt: str
):
    """
    **Feature: minichat-app, Property 3: Agent Update Persistence**
    
    For any existing Agent and valid update to its properties,
    updating the agent and then retrieving it should reflect the updated values.
    
    **Validates: Requirements 3.3**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = ChatDatabase(db_path=db_path)
        
        # Create an agent
        agent = Agent(
            id=None,
            name=original_name,
            description=original_description,
            system_prompt=original_prompt,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        agent_id = db.create_agent(agent)
        
        # Update the agent with new values
        updated_agent = Agent(
            id=agent_id,
            name=updated_name,
            description=updated_description,
            system_prompt=updated_prompt,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        success = db.update_agent(updated_agent)
        assert success, "Update should succeed"
        
        # Retrieve and verify
        retrieved = db.get_agent(agent_id)
        
        assert retrieved is not None, "Agent should still exist"
        assert retrieved.id == agent_id
        assert retrieved.name == updated_name
        assert retrieved.description == updated_description
        assert retrieved.system_prompt == updated_prompt



@settings(max_examples=100, deadline=None)
@given(
    name=name_strategy,
    description=description_strategy,
    system_prompt=system_prompt_strategy
)
def test_agent_deletion_removes_agent(name: str, description: str, system_prompt: str):
    """
    **Feature: minichat-app, Property 4: Agent Deletion Removes Agent**
    
    For any existing Agent, deleting it should result in the agent
    no longer being retrievable from the database.
    
    **Validates: Requirements 3.4**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = ChatDatabase(db_path=db_path)
        
        # Create an agent
        agent = Agent(
            id=None,
            name=name,
            description=description,
            system_prompt=system_prompt,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        agent_id = db.create_agent(agent)
        
        # Verify it exists
        retrieved = db.get_agent(agent_id)
        assert retrieved is not None, "Agent should exist before deletion"
        
        # Delete the agent
        success = db.delete_agent(agent_id)
        assert success, "Deletion should succeed"
        
        # Verify it no longer exists
        retrieved_after = db.get_agent(agent_id)
        assert retrieved_after is None, "Agent should not exist after deletion"
