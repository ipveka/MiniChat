"""
Embedding service using sentence-transformers.

Provides text embedding capabilities for semantic search.
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.
    
    Uses the all-MiniLM-L6-v2 model by default, which produces 384-dimensional
    embeddings suitable for semantic similarity search.
    """
    
    # Class-level model cache to avoid reloading
    _model_cache: dict = {}
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers package not installed")
    
    def _get_model(self) -> SentenceTransformer:
        """
        Get or load the sentence-transformers model.
        
        Uses class-level caching to avoid reloading the model.
        
        Returns:
            Loaded SentenceTransformer model
            
        Raises:
            RuntimeError: If sentence-transformers is not installed
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Please install it with `pip install sentence-transformers`"
            )
        
        # Check class-level cache first
        if self.model_name in EmbeddingService._model_cache:
            return EmbeddingService._model_cache[self.model_name]
        
        # Load and cache the model
        logger.info(f"Loading embedding model: {self.model_name}")
        model = SentenceTransformer(self.model_name)
        EmbeddingService._model_cache[self.model_name] = model
        
        return model

    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            RuntimeError: If sentence-transformers is not installed
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple text strings.
        
        More efficient than calling embed_text multiple times
        as it batches the encoding.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (each a list of floats)
            
        Raises:
            RuntimeError: If sentence-transformers is not installed
            ValueError: If texts list is empty or contains empty strings
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")
        
        # Validate all texts are non-empty
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Cannot embed empty text at index {i}")
        
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
    
    @property
    def embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Integer dimension of embedding vectors
        """
        model = self._get_model()
        return model.get_sentence_embedding_dimension()
