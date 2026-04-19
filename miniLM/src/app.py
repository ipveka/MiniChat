"""
Main application entry point for MiniChat.

Sets up Streamlit page configuration, initializes all services,
and provides navigation between Chat, Studio, and Agents pages.
"""
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple

from miniLM.config.settings import get_settings, Settings
from miniLM.src.utils.logger import setup_logger, get_logger
from miniLM.src.database.init_db import init_database, verify_database
from miniLM.src.database.chat_db import ChatDatabase
from miniLM.src.database.vector_db import VectorDatabase
from miniLM.src.llm.ollama_client import OllamaClient, OllamaConnectionError
from miniLM.src.llm.embeddings import EmbeddingService
from miniLM.src.rag.document_processor import DocumentProcessor
from miniLM.src.rag.retriever import Retriever
from miniLM.src.ui.chat import render_chat_page
from miniLM.src.ui.studio import render_studio_page
from miniLM.src.ui.agents import render_agents_page


# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="MiniChat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_logger(settings: Settings) -> None:
    """Initialize the application logger."""
    base_dir = Path(__file__).parent.parent
    log_dir = base_dir / settings.log_dir
    setup_logger("minichat", str(log_dir), settings.log_level)


def init_database_service(settings: Settings) -> Optional[ChatDatabase]:
    """
    Initialize the SQLite database service.
    
    Returns:
        ChatDatabase instance or None if initialization fails.
    """
    logger = get_logger("minichat")
    
    try:
        base_dir = Path(__file__).parent.parent
        db_path = base_dir / settings.sqlite_db_path
        
        # Ensure data directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database tables
        init_database(db_path)
        
        # Verify database is ready
        if not verify_database(db_path):
            logger.error("Database verification failed")
            return None
        
        # Create ChatDatabase instance
        chat_db = ChatDatabase(db_path)
        logger.info("Database service initialized successfully")
        return chat_db
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return None


def init_ollama_client(settings: Settings) -> Tuple[Optional[OllamaClient], Optional[str]]:
    """
    Initialize the Ollama client.
    
    Returns:
        Tuple of (OllamaClient instance or None, error message or None).
    """
    logger = get_logger("minichat")
    
    try:
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.default_model
        )
        
        # Check if Ollama is available
        if not client.is_available():
            error_msg = "Ollama is not running. Please start Ollama and refresh the page."
            logger.warning(error_msg)
            return client, error_msg
        
        # Check if the model is available
        if not client.check_model():
            error_msg = f"Model '{settings.default_model}' not found. Please pull it with `ollama pull {settings.default_model}`"
            logger.warning(error_msg)
            return client, error_msg
        
        logger.info(f"Ollama client initialized with model: {settings.default_model}")
        return client, None
        
    except Exception as e:
        logger.error(f"Failed to initialize Ollama client: {e}")
        return None, f"Failed to connect to Ollama: {str(e)}"


def init_vector_db(settings: Settings) -> Optional[VectorDatabase]:
    """
    Initialize the ChromaDB vector database.
    
    Returns:
        VectorDatabase instance or None if initialization fails.
    """
    logger = get_logger("minichat")
    
    try:
        base_dir = Path(__file__).parent.parent
        chroma_path = base_dir / settings.chroma_persist_dir
        
        # Ensure directory exists
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        vector_db = VectorDatabase(str(chroma_path))
        logger.info("Vector database initialized successfully")
        return vector_db
        
    except Exception as e:
        logger.error(f"Failed to initialize vector database: {e}")
        return None


def init_embedding_service(settings: Settings) -> Optional[EmbeddingService]:
    """
    Initialize the embedding service.
    
    Returns:
        EmbeddingService instance or None if initialization fails.
    """
    logger = get_logger("minichat")
    
    try:
        embedding_service = EmbeddingService(model_name=settings.embedding_model)
        logger.info(f"Embedding service initialized with model: {settings.embedding_model}")
        return embedding_service
        
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        return None


def init_document_processor(settings: Settings) -> Optional[DocumentProcessor]:
    """
    Initialize the document processor.
    
    Returns:
        DocumentProcessor instance or None if initialization fails.
    """
    logger = get_logger("minichat")
    
    try:
        doc_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        logger.info("Document processor initialized successfully")
        return doc_processor
        
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {e}")
        return None


@st.cache_resource
def initialize_services():
    """
    Initialize all application services.
    
    Uses Streamlit caching to avoid reinitializing on every rerun.
    
    Returns:
        Dictionary containing all initialized services and any errors.
    """
    settings = get_settings()
    
    # Initialize logger first
    init_logger(settings)
    logger = get_logger("minichat")
    logger.info("Starting MiniChat application...")
    
    services = {
        "settings": settings,
        "chat_db": None,
        "ollama_client": None,
        "vector_db": None,
        "embedding_service": None,
        "doc_processor": None,
        "retriever": None,
        "errors": []
    }
    
    # Initialize database
    chat_db = init_database_service(settings)
    if chat_db is None:
        services["errors"].append("Database initialization failed. Please check the logs.")
    else:
        services["chat_db"] = chat_db
    
    # Initialize Ollama client
    ollama_client, ollama_error = init_ollama_client(settings)
    services["ollama_client"] = ollama_client
    if ollama_error:
        services["errors"].append(ollama_error)
    
    # Initialize vector database
    vector_db = init_vector_db(settings)
    if vector_db is None:
        services["errors"].append("Vector database initialization failed. RAG features will be unavailable.")
    else:
        services["vector_db"] = vector_db
    
    # Initialize embedding service
    embedding_service = init_embedding_service(settings)
    if embedding_service is None:
        services["errors"].append("Embedding service initialization failed. RAG features will be unavailable.")
    else:
        services["embedding_service"] = embedding_service
    
    # Initialize document processor
    doc_processor = init_document_processor(settings)
    if doc_processor is None:
        services["errors"].append("Document processor initialization failed. Document upload will be unavailable.")
    else:
        services["doc_processor"] = doc_processor
    
    # Initialize retriever (requires vector_db and embedding_service)
    if vector_db and embedding_service:
        services["retriever"] = Retriever(vector_db, embedding_service)
        logger.info("Retriever initialized successfully")
    
    logger.info("Service initialization complete")
    return services


def render_sidebar_navigation() -> str:
    """
    Render the sidebar navigation and return the selected page.
    
    Returns:
        Selected page name: "Chat", "Studio", or "Agents"
    """
    with st.sidebar:
        st.title("💬 MiniChat")
        st.caption("Local-first LLM Assistant")
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            options=["Chat", "Studio", "Agents"],
            index=0,
            label_visibility="collapsed"
        )
        
        return page


def render_error_banner(errors: list) -> None:
    """Display error messages at the top of the page."""
    for error in errors:
        st.error(f"⚠️ {error}")


def main():
    """Main application entry point."""
    # Initialize all services
    services = initialize_services()
    
    # Get selected page from navigation
    page = render_sidebar_navigation()
    
    # Display any initialization errors
    if services["errors"]:
        render_error_banner(services["errors"])
    
    # Check for critical services
    chat_db = services["chat_db"]
    ollama_client = services["ollama_client"]
    vector_db = services["vector_db"]
    embedding_service = services["embedding_service"]
    doc_processor = services["doc_processor"]
    retriever = services["retriever"]
    
    # Route to selected page
    if page == "Chat":
        if chat_db is None:
            st.error("Chat is unavailable: Database not initialized.")
            return
        
        # Get agents for chat page
        agents = chat_db.get_agents() if chat_db else []
        
        render_chat_page(
            chat_db=chat_db,
            ollama_client=ollama_client,
            agents=agents
        )
    
    elif page == "Studio":
        if vector_db is None or retriever is None:
            st.error("Studio is unavailable: Vector database or retriever not initialized.")
            return
        
        if doc_processor is None:
            st.error("Document upload is unavailable: Document processor not initialized.")
            return
        
        if embedding_service is None:
            st.error("Studio is unavailable: Embedding service not initialized.")
            return
        
        # Get agents for studio page
        agents = chat_db.get_agents() if chat_db else []
        
        render_studio_page(
            vector_db=vector_db,
            retriever=retriever,
            ollama_client=ollama_client,
            doc_processor=doc_processor,
            embedding_service=embedding_service,
            agents=agents
        )
    
    elif page == "Agents":
        if chat_db is None:
            st.error("Agents is unavailable: Database not initialized.")
            return
        
        render_agents_page(chat_db=chat_db)


if __name__ == "__main__":
    main()
