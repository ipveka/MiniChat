"""
Property-based tests for embedding service.
"""
from hypothesis import given, strategies as st, settings

from miniLM.src.llm.embeddings import EmbeddingService


# Strategy for non-empty text content (avoiding empty/whitespace-only strings)
text_strategy = st.text(min_size=1, max_size=500).filter(lambda x: x.strip())


@settings(max_examples=100, deadline=None)
@given(text=text_strategy)
def test_embedding_dimension_consistency(text: str):
    """
    **Feature: minichat-app, Property 8: Embedding Dimension Consistency**
    
    For any non-empty text string, generating an embedding should produce
    a vector of consistent dimension (384 for all-MiniLM-L6-v2), and
    embedding the same text twice should produce identical vectors.
    
    **Validates: Requirements 6.1**
    """
    service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    
    # Generate embedding
    embedding = service.embed_text(text)
    
    # Verify dimension is 384 (all-MiniLM-L6-v2 produces 384-dimensional vectors)
    assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"
    
    # Verify all values are floats
    assert all(isinstance(v, float) for v in embedding), "All embedding values should be floats"
    
    # Generate embedding again for the same text
    embedding_again = service.embed_text(text)
    
    # Verify determinism - same text should produce identical embeddings
    assert len(embedding_again) == 384, "Second embedding should also have 384 dimensions"
    
    # Compare embeddings (should be identical for same input)
    for i, (v1, v2) in enumerate(zip(embedding, embedding_again)):
        assert v1 == v2, f"Embedding values differ at index {i}: {v1} != {v2}"
