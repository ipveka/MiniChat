"""
Ollama client for local LLM inference.

Provides connection management and streaming chat/generate capabilities.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator, List, Optional
import logging

try:
    import ollama
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    Client = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message with role and content."""
    role: str  # "user", "assistant", or "system"
    content: str


@dataclass
class OllamaResponse:
    """Represents a response chunk from Ollama."""
    content: str
    done: bool


class OllamaConnectionError(Exception):
    """Raised when Ollama is not available or connection fails."""
    def __init__(self, message: str, user_message: str, recoverable: bool = True):
        super().__init__(message)
        self.user_message = user_message
        self.recoverable = recoverable


class ModelNotFoundError(Exception):
    """Raised when the requested model is not available in Ollama."""
    def __init__(self, message: str, user_message: str, recoverable: bool = True):
        super().__init__(message)
        self.user_message = user_message
        self.recoverable = recoverable


class OllamaClient:
    """
    Client for interacting with local Ollama instance.
    
    Provides methods for checking availability, model status,
    and streaming chat/generate responses.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: URL of the Ollama server
            model: Default model to use for generation
        """
        self.base_url = base_url
        self.model = model
        self._client: Optional[Any] = None
        
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama package not installed")
    
    def _get_client(self) -> Any:
        """Get or create the Ollama client instance."""
        if self._client is None:
            if not OLLAMA_AVAILABLE:
                raise OllamaConnectionError(
                    "Ollama package not installed",
                    "Ollama is not installed. Please install it with `pip install ollama`",
                    recoverable=False
                )
            self._client = Client(host=self.base_url)
        return self._client
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is available and responding.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            client = self._get_client()
            # Try to list models to verify connection
            client.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def check_model(self, model: Optional[str] = None) -> bool:
        """
        Check if a specific model is available in Ollama.
        
        Args:
            model: Model name to check (uses default if not specified)
            
        Returns:
            True if model is available, False otherwise
        """
        model_name = model or self.model
        
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            client = self._get_client()
            models_response = client.list()
            available_models = [m.get('name', m.get('model', '')) for m in models_response.get('models', [])]
            
            # Check for exact match or match without tag
            for available in available_models:
                if available == model_name or available.split(':')[0] == model_name:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Failed to check model {model_name}: {e}")
            return False

    
    def chat(
        self, 
        messages: List[ChatMessage], 
        system_prompt: Optional[str] = None
    ) -> Generator[OllamaResponse, None, None]:
        """
        Send chat messages to Ollama and stream the response.
        
        Args:
            messages: List of chat messages
            system_prompt: Optional system prompt to prepend
            
        Yields:
            OllamaResponse objects with content chunks and done status
            
        Raises:
            OllamaConnectionError: If Ollama is not available
            ModelNotFoundError: If the model is not found
        """
        if not self.is_available():
            raise OllamaConnectionError(
                "Ollama server not available",
                "Ollama is not running. Please start Ollama and refresh.",
                recoverable=True
            )
        
        if not self.check_model():
            raise ModelNotFoundError(
                f"Model {self.model} not found",
                f"Model '{self.model}' not found. Please pull it with `ollama pull {self.model}`",
                recoverable=True
            )
        
        try:
            client = self._get_client()
            
            # Build message list for Ollama
            ollama_messages = []
            
            # Add system prompt if provided
            if system_prompt:
                ollama_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add chat messages
            for msg in messages:
                ollama_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Stream the response
            stream = client.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True
            )
            
            for chunk in stream:
                content = chunk.get('message', {}).get('content', '')
                done = chunk.get('done', False)
                yield OllamaResponse(content=content, done=done)
                
        except OllamaConnectionError:
            raise
        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise OllamaConnectionError(
                f"Chat failed: {e}",
                f"Failed to communicate with Ollama: {str(e)}",
                recoverable=True
            )

    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Generator[OllamaResponse, None, None]:
        """
        Generate a response for a single prompt.
        
        Args:
            prompt: The prompt to generate a response for
            system_prompt: Optional system prompt
            
        Yields:
            OllamaResponse objects with content chunks and done status
            
        Raises:
            OllamaConnectionError: If Ollama is not available
            ModelNotFoundError: If the model is not found
        """
        if not self.is_available():
            raise OllamaConnectionError(
                "Ollama server not available",
                "Ollama is not running. Please start Ollama and refresh.",
                recoverable=True
            )
        
        if not self.check_model():
            raise ModelNotFoundError(
                f"Model {self.model} not found",
                f"Model '{self.model}' not found. Please pull it with `ollama pull {self.model}`",
                recoverable=True
            )
        
        try:
            client = self._get_client()
            
            # Stream the response
            stream = client.generate(
                model=self.model,
                prompt=prompt,
                system=system_prompt if system_prompt else None,
                stream=True
            )
            
            for chunk in stream:
                content = chunk.get('response', '')
                done = chunk.get('done', False)
                yield OllamaResponse(content=content, done=done)
                
        except OllamaConnectionError:
            raise
        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Generate error: {e}")
            raise OllamaConnectionError(
                f"Generate failed: {e}",
                f"Failed to communicate with Ollama: {str(e)}",
                recoverable=True
            )
