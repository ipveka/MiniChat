"""
Property-based tests for retriever operations.
"""
import os
import shutil
import tempfile
import uuid
from datetime import datetime

from hypothesis import given, strategies as st, settings, assume

from miniLM.src.database.vector_db import VectorDatabase, DocumentChunk
from miniLM.src.llm.embeddings import EmbeddingService
from miniLM.src.rag.retriever import Retriever, RetrievalResult
from miniLM.src.utils.helpers import generate_uuid


# Strategy for non-empty content that's meaningful for embedding
content_strategy = st.text(
    min_size=10,
    max_size=200,
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'))
).filter(lambda x: len(x.strip()) >= 10)

# Strategy for source filename
source_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='-_.'),
    min_size=1,
    max_size=30
).filter(lambda x: x.strip())


@settings(max_examples=20, deadline=None)
@given(
    target_content=content_strategy,
    distractor_content=content_strategy,
    source=source_strategy
)
def test_semantic_similarity_search_returns_relevant_results(
    target_content: str,
    distractor_content: str,
    source: str
):
    """
    **Feature: minichat-app, Property 9: Semantic Similarity Search Returns Relevant Results**
    
    For any set of stored document chunks and a query semantically similar to
    one chunk's content, querying should return that chunk with a higher rank
    than dissimilar chunks.
    
    **Validates: Requirements 2.2, 6.3**
    """
    # Ensure contents are different enough
    assume(target_content.strip() != distractor_content.strip())
    assume(len(target_content.strip()) >= 10)
    assume(len(distractor_content.strip()) >= 10)
    
    test_id = str(uuid.uuid4())[:8]
    persist_dir = os.path.join(tempfile.gettempdir(), f"chroma_retriever_test_{test_id}")
    
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        vector_db = VectorDatabase(persist_dir=persist_dir)
        retriever = Retriever(vector_db=vector_db, embedding_service=embedding_service)
        
        # Create target chunk (the one we want to find)
        target_embedding = embedding_service.embed_text(target_content)
        target_chunk = DocumentChunk(
            id=generate_uuid(),
            content=target_content,
            metadata={
                "source": f"target_{source}",
                "chunk_index": 0,
                "created_at": datetime.now().isoformat()
            },
            embedding=target_embedding
        )
        
        # Create distractor chunk
        distractor_embedding = embedding_service.embed_text(distractor_content)
        distractor_chunk = DocumentChunk(
            id=generate_uuid(),
            content=distractor_content,
            metadata={
                "source": f"distractor_{source}",
                "chunk_index": 0,
                "created_at": datetime.now().isoformat()
            },
            embedding=distractor_embedding
        )
        
        # Add both chunks
        vector_db.add_chunks([target_chunk, distractor_chunk])
        
        # Query with the target content (should find target chunk first)
        result = retriever.retrieve(target_content, n_results=2)
        
        # Verify we got results
        assert len(result.chunks) >= 1, "Should return at least one result"
        
        # The first result should be the target chunk (exact semantic match)
        first_result = result.chunks[0]
        assert first_result.content == target_content, \
            "First result should be the semantically matching chunk"
        
    finally:
        try:
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir, ignore_errors=True)
        except Exception:
            pass



@settings(max_examples=20, deadline=None)
@given(
    content=content_strategy,
    source=source_strategy,
    chunk_index=st.integers(min_value=0, max_value=100)
)
def test_retrieval_results_include_source_metadata(
    content: str,
    source: str,
    chunk_index: int
):
    """
    **Feature: minichat-app, Property 10: Retrieval Results Include Source Metadata**
    
    For any retrieval query that returns results, each returned chunk should
    include source metadata identifying the original document.
    
    **Validates: Requirements 2.3**
    """
    assume(len(content.strip()) >= 10)
    assume(source.strip())
    
    test_id = str(uuid.uuid4())[:8]
    persist_dir = os.path.join(tempfile.gettempdir(), f"chroma_metadata_test_{test_id}")
    
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        vector_db = VectorDatabase(persist_dir=persist_dir)
        retriever = Retriever(vector_db=vector_db, embedding_service=embedding_service)
        
        # Create chunk with specific source metadata
        embedding = embedding_service.embed_text(content)
        chunk = DocumentChunk(
            id=generate_uuid(),
            content=content,
            metadata={
                "source": source,
                "chunk_index": chunk_index,
                "created_at": datetime.now().isoformat()
            },
            embedding=embedding
        )
        
        # Add the chunk
        vector_db.add_chunks([chunk])
        
        # Query for the chunk
        result = retriever.retrieve(content, n_results=1)
        
        # Verify we got results
        assert len(result.chunks) >= 1, "Should return at least one result"
        
        # Verify each result has source metadata
        for retrieved_chunk in result.chunks:
            assert retrieved_chunk.metadata is not None, \
                "Retrieved chunk should have metadata"
            assert "source" in retrieved_chunk.metadata, \
                "Retrieved chunk metadata should include 'source'"
            assert retrieved_chunk.metadata["source"], \
                "Source metadata should not be empty"
            
            # Verify the source matches what we stored
            assert retrieved_chunk.metadata["source"] == source, \
                f"Source should be '{source}', got '{retrieved_chunk.metadata['source']}'"
            
            # Verify chunk_index is present
            assert "chunk_index" in retrieved_chunk.metadata, \
                "Retrieved chunk metadata should include 'chunk_index'"
        
    finally:
        try:
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir, ignore_errors=True)
        except Exception:
            pass
