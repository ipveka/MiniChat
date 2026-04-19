"""
Retriever for MiniChat RAG pipeline.
Handles semantic search and context formatting for LLM prompts.
"""
from dataclasses import dataclass
from typing import List, Optional

from miniLM.src.database.vector_db import VectorDatabase, DocumentChunk
from miniLM.src.llm.embeddings import EmbeddingService
from miniLM.src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""
    chunks: List[DocumentChunk]
    query: str


class Retriever:
    """
    Retriever for semantic search over document chunks.
    
    Combines embedding service and vector database to find
    relevant document chunks for a given query.
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_service: EmbeddingService
    ):
        """
        Initialize the Retriever.
        
        Args:
            vector_db: VectorDatabase instance for chunk storage/retrieval.
            embedding_service: EmbeddingService for generating query embeddings.
        """
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        logger.info("Retriever initialized")
    
    def retrieve(self, query: str, n_results: int = 5) -> RetrievalResult:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: The search query text.
            n_results: Maximum number of results to return.
        
        Returns:
            RetrievalResult: Contains the query and matching chunks.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return RetrievalResult(chunks=[], query=query)
        
        # Check if database has documents
        if self.vector_db.get_document_count() == 0:
            logger.info("Vector database is empty, no documents to search")
            return RetrievalResult(chunks=[], query=query)
        
        # Generate embedding for the query
        query_embedding = self.embedding_service.embed_text(query)
        
        # Search for similar chunks
        chunks = self.vector_db.query(query_embedding, n_results=n_results)
        
        logger.debug(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
        return RetrievalResult(chunks=chunks, query=query)
    
    def format_context(self, result: RetrievalResult) -> str:
        """
        Format retrieval results as context for LLM prompt.
        
        Args:
            result: RetrievalResult from a retrieve() call.
        
        Returns:
            str: Formatted context string for inclusion in LLM prompt.
        """
        if not result.chunks:
            return ""
        
        context_parts = []
        context_parts.append("Relevant context from documents:\n")
        
        for i, chunk in enumerate(result.chunks, 1):
            source = chunk.metadata.get("source", "Unknown")
            chunk_idx = chunk.metadata.get("chunk_index", "?")
            
            context_parts.append(f"--- Source: {source} (chunk {chunk_idx}) ---")
            context_parts.append(chunk.content)
            context_parts.append("")
        
        return "\n".join(context_parts)
