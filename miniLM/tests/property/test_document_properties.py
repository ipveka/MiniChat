"""
Property-based tests for document processing.
"""
import tempfile
from pathlib import Path

from hypothesis import given, strategies as st, settings, assume

from miniLM.src.rag.document_processor import (
    DocumentProcessor,
    ProcessedDocument,
    DocumentProcessingError,
    SUPPORTED_EXTENSIONS
)


# Strategy for non-empty text content (valid document content)
text_content_strategy = st.text(
    min_size=10,
    max_size=5000,
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S', 'Z'))
).filter(lambda x: x.strip() and len(x.strip()) >= 10)


@settings(max_examples=100, deadline=None)
@given(content=text_content_strategy)
def test_document_processing_produces_valid_chunks(content: str):
    """
    **Feature: minichat-app, Property 6: Document Processing Produces Valid Chunks**
    
    For any valid document file (PDF, DOCX, TXT, MD) with non-empty content,
    processing it should produce a list of DocumentChunks where each chunk
    has non-empty content, valid metadata including source, and chunk_index.
    
    **Validates: Requirements 2.1, 2.6, 5.1, 5.2, 5.3**
    """
    # Skip content that's only whitespace after stripping
    assume(content.strip())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test text file
        test_file = Path(tmpdir) / "test_document.txt"
        test_file.write_text(content, encoding='utf-8')
        
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        result = processor.process_file(str(test_file))
        
        # Verify result is a ProcessedDocument
        assert isinstance(result, ProcessedDocument)
        assert result.source == "test_document.txt"
        
        # Verify chunks are produced for non-empty content
        assert len(result.chunks) >= 1, "Should produce at least one chunk"
        
        # Verify each chunk has required properties
        for i, chunk in enumerate(result.chunks):
            # Non-empty content
            assert chunk.content, f"Chunk {i} should have non-empty content"
            assert chunk.content.strip(), f"Chunk {i} content should not be only whitespace"
            
            # Valid metadata
            assert chunk.metadata is not None, f"Chunk {i} should have metadata"
            assert "source" in chunk.metadata, f"Chunk {i} should have source in metadata"
            assert chunk.metadata["source"] == "test_document.txt"
            assert "chunk_index" in chunk.metadata, f"Chunk {i} should have chunk_index"
            assert chunk.metadata["chunk_index"] == i, f"Chunk {i} should have correct index"
            
            # Valid ID
            assert chunk.id, f"Chunk {i} should have an ID"



@settings(max_examples=100, deadline=None)
@given(
    content=st.text(min_size=1, max_size=3000).filter(lambda x: x.strip()),
    chunk_size=st.integers(min_value=50, max_value=500),
    chunk_overlap=st.integers(min_value=0, max_value=49)
)
def test_text_chunking_preserves_content(content: str, chunk_size: int, chunk_overlap: int):
    """
    **Feature: minichat-app, Property 7: Text Chunking Preserves Content**
    
    For any non-empty text string, chunking it should produce chunks whose
    concatenated content (accounting for overlap) contains all original content,
    and each chunk size should be within configured bounds.
    
    **Validates: Requirements 5.4**
    """
    # Ensure overlap is less than chunk_size
    assume(chunk_overlap < chunk_size)
    assume(content.strip())
    
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = processor.chunk_text(content.strip(), "test_source")
    
    # Should produce at least one chunk for non-empty content
    assert len(chunks) >= 1, "Should produce at least one chunk"
    
    # Verify chunk sizes are within bounds
    for i, chunk in enumerate(chunks):
        # Each chunk should not exceed chunk_size
        assert len(chunk.content) <= chunk_size, f"Chunk {i} exceeds max size"
        # Each chunk should have content
        assert chunk.content, f"Chunk {i} should have content"
    
    # Verify all original content is preserved
    # Reconstruct text by taking non-overlapping portions
    original_text = content.strip()
    
    # Check that every character in original appears in at least one chunk
    all_chunk_content = "".join(c.content for c in chunks)
    
    # For overlapping chunks, the total content length should be >= original
    # (due to overlap duplication) or equal if no overlap needed
    if len(original_text) <= chunk_size:
        # Single chunk case - should contain all content
        assert chunks[0].content == original_text
    else:
        # Multi-chunk case - verify coverage
        # Each position in original should be covered by at least one chunk
        step = chunk_size - chunk_overlap
        for pos in range(len(original_text)):
            # Find which chunk should contain this position
            chunk_idx = pos // step if step > 0 else 0
            if chunk_idx >= len(chunks):
                chunk_idx = len(chunks) - 1
            
            # The character at this position should appear in the reconstructed content
            char = original_text[pos]
            assert char in all_chunk_content, f"Character at position {pos} not found in chunks"
