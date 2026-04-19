"""
Vector database operations for MiniChat application.
Provides ChromaDB integration for document embeddings storage and retrieval.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from miniLM.config.settings import get_settings
from miniLM.src.utils.logger import get_logger

logger = get_logger(__name__)

# Collection name for documents
COLLECTION_NAME = "documents"


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with its embedding."""
    id: str
    content: str
    metadata: Dict[str, Any]  # source, page, chunk_index, created_at
    embedding: Optional[List[float]] = None


class VectorDatabase:
    """Vector database operations using ChromaDB."""
    
    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize VectorDatabase.
        
        Args:
            persist_dir: Optional directory for ChromaDB persistence.
                        If not provided, uses the path from settings.
        """
        if persist_dir is None:
            settings = get_settings()
            persist_dir = str(settings.get_absolute_chroma_path())
        
        # Ensure directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        self.persist_dir = persist_dir
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized VectorDatabase at {persist_dir}")

    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector database.
        
        Args:
            chunks: List of DocumentChunk objects to add.
                   Each chunk must have an embedding.
        
        Raises:
            ValueError: If any chunk is missing an embedding.
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        # Validate all chunks have embeddings
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} is missing embedding")
        
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Added {len(chunks)} chunks to vector database")
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> List[DocumentChunk]:
        """
        Query the vector database for similar chunks.
        
        Args:
            query_embedding: The embedding vector to search with.
            n_results: Maximum number of results to return.
        
        Returns:
            List[DocumentChunk]: List of matching chunks ordered by similarity.
        """
        if self.get_document_count() == 0:
            logger.warning("Vector database is empty")
            return []
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "embeddings"]
        )
        
        chunks = []
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=results['documents'][0][i] if results['documents'] else "",
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                    embedding=results['embeddings'][0][i] if results['embeddings'] else None
                )
                chunks.append(chunk)
        
        logger.debug(f"Query returned {len(chunks)} results")
        return chunks

    
    def delete_document(self, source: str) -> None:
        """
        Delete all chunks from a specific document.
        
        Args:
            source: The source filename to delete chunks for.
        """
        # Get all chunks with this source
        results = self._collection.get(
            where={"source": source},
            include=["metadatas"]
        )
        
        if results['ids']:
            self._collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks from source: {source}")
        else:
            logger.warning(f"No chunks found for source: {source}")
    
    def get_document_count(self) -> int:
        """
        Get the total number of document chunks in the database.
        
        Returns:
            int: Number of chunks stored.
        """
        return self._collection.count()
    
    def get_all_sources(self) -> List[str]:
        """
        Get all unique document sources in the database.
        
        Returns:
            List[str]: List of unique source filenames.
        """
        results = self._collection.get(include=["metadatas"])
        sources = set()
        if results['metadatas']:
            for metadata in results['metadatas']:
                if metadata and 'source' in metadata:
                    sources.add(metadata['source'])
        return list(sources)
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate the collection
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared all documents from vector database")
