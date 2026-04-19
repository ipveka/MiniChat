"""
Property-based tests for vector database operations.
"""
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from hypothesis import given, strategies as st, settings

from miniLM.src.database.vector_db import VectorDatabase, DocumentChunk
from miniLM.src.utils.helpers import generate_uuid


# Strategy for non-empty content
content_strategy = st.text(min_size=1, max_size=500).filter(lambda x: x.strip())

# Strategy for source filename
source_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='-_.'),
    min_size=1,
    max_size=50
).filter(lambda x: x.strip())

# Strategy for embedding vectors (384 dimensions for all-MiniLM-L6-v2)
embedding_strategy = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=384,
    max_size=384
)


@settings(max_examples=100, deadline=None)
@given(
    content=content_strategy,
    source=source_strategy,
    chunk_index=st.integers(min_value=0, max_value=1000),
    embedding=embedding_strategy
)
def test_document_chunk_round_trip(
    content: str,
    source: str,
    chunk_index: int,
    embedding: list
):
    """
    **Feature: minichat-app, Property 5: Document Chunk Round-Trip**
    
    For any valid DocumentChunk with content, metadata, and embedding,
    storing it in ChromaDB and querying with the same embedding should
    return the chunk with equivalent content and metadata.
    
    **Validates: Requirements 2.5, 2.7, 6.2**
    """
    # Use a unique directory for each test run to avoid conflicts
    test_id = str(uuid.uuid4())[:8]
    persist_dir = os.path.join(tempfile.gettempdir(), f"chroma_test_{test_id}")
    
    try:
        db = VectorDatabase(persist_dir=persist_dir)
        
        # Create a document chunk
        chunk_id = generate_uuid()
        metadata = {
            "source": source,
            "chunk_index": chunk_index,
            "created_at": datetime.now().isoformat()
        }
        
        chunk = DocumentChunk(
            id=chunk_id,
            content=content,
            metadata=metadata,
            embedding=embedding
        )
        
        # Add the chunk
        db.add_chunks([chunk])
        
        # Query with the same embedding
        results = db.query(query_embedding=embedding, n_results=1)
        
        # Verify round-trip
        assert len(results) >= 1, "Should return at least one result"
        
        # The first result should be our chunk (exact match)
        retrieved = results[0]
        assert retrieved.id == chunk_id
        assert retrieved.content == content
        assert retrieved.metadata["source"] == source
        assert retrieved.metadata["chunk_index"] == chunk_index
    finally:
        # Clean up
        try:
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir, ignore_errors=True)
        except Exception:
            pass
