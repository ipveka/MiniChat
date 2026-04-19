"""
Studio UI components for MiniChat application.
Provides the RAG interface for document upload and querying.
"""
import tempfile
from pathlib import Path
from typing import List, Optional

import streamlit as st

from miniLM.src.database.chat_db import Agent
from miniLM.src.database.vector_db import VectorDatabase, DocumentChunk
from miniLM.src.llm.ollama_client import OllamaClient, OllamaConnectionError, ModelNotFoundError
from miniLM.src.llm.embeddings import EmbeddingService
from miniLM.src.rag.document_processor import DocumentProcessor, DocumentProcessingError
from miniLM.src.rag.retriever import Retriever


def render_studio_page(
    vector_db: VectorDatabase,
    retriever: Retriever,
    ollama_client: OllamaClient,
    doc_processor: DocumentProcessor,
    embedding_service: EmbeddingService,
    agents: List[Agent]
) -> None:
    """
    Render the Studio (RAG) page.
    
    Args:
        vector_db: VectorDatabase instance for document storage.
        retriever: Retriever instance for semantic search.
        ollama_client: OllamaClient for LLM inference.
        doc_processor: DocumentProcessor for handling uploads.
        embedding_service: EmbeddingService for generating embeddings.
        agents: List of available agents.
    """
    st.header("📚 Studio")
    st.caption("Upload documents and query them using RAG")
    
    # Initialize session state
    if "studio_query_result" not in st.session_state:
        st.session_state.studio_query_result = None
    if "studio_selected_agent_id" not in st.session_state:
        st.session_state.studio_selected_agent_id = None
    
    # Agent selection in sidebar
    with st.sidebar:
        st.subheader("RAG Settings")
        
        agent_options = {"None": None}
        agent_options.update({agent.name: agent.id for agent in agents})
        
        selected_agent_name = st.selectbox(
            "Select Agent for RAG",
            options=list(agent_options.keys()),
            index=0,
            help="Select an agent to apply its system prompt to RAG queries",
            key="studio_agent_select"
        )
        st.session_state.studio_selected_agent_id = agent_options[selected_agent_name]
        
        st.divider()
        
        # Document stats
        doc_count = vector_db.get_document_count()
        st.metric("Document Chunks", doc_count)
        
        sources = vector_db.get_all_sources()
        if sources:
            st.write("**Uploaded Documents:**")
            for source in sources:
                st.write(f"• {source}")
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["📤 Upload", "🔍 Query"])
    
    with tab1:
        render_upload_section(doc_processor, vector_db, embedding_service)
    
    with tab2:
        render_query_section(retriever, ollama_client, agents)


def render_upload_section(
    doc_processor: DocumentProcessor,
    vector_db: VectorDatabase,
    embedding_service: EmbeddingService
) -> None:
    """
    Render the document upload section.
    
    Args:
        doc_processor: DocumentProcessor for handling uploads.
        vector_db: VectorDatabase for storing embeddings.
        embedding_service: EmbeddingService for generating embeddings.
    """
    st.subheader("Upload Documents")
    st.write("Supported formats: PDF, DOCX, TXT, MD")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "md"],
        help="Upload a document to add to the RAG knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file to temp location
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_file.name).suffix
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process the document
                    processed = doc_processor.process_file(tmp_path)
                    
                    # Generate embeddings for chunks
                    st.info(f"Generating embeddings for {len(processed.chunks)} chunks...")
                    
                    chunks_with_embeddings = []
                    for chunk in processed.chunks:
                        embedding = embedding_service.embed_text(chunk.content)
                        chunk.embedding = embedding
                        chunks_with_embeddings.append(chunk)
                    
                    # Store in vector database
                    vector_db.add_chunks(chunks_with_embeddings)
                    
                    st.success(
                        f"✅ Successfully processed '{uploaded_file.name}': "
                        f"{len(processed.chunks)} chunks added"
                    )
                    
                    # Clean up temp file
                    Path(tmp_path).unlink(missing_ok=True)
                    
                except DocumentProcessingError as e:
                    st.error(f"❌ {e.user_message}")
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
    
    # Document management
    st.divider()
    st.subheader("Manage Documents")
    
    sources = vector_db.get_all_sources()
    if sources:
        selected_source = st.selectbox(
            "Select document to remove",
            options=sources,
            key="remove_doc_select"
        )
        
        if st.button("🗑️ Remove Document", type="secondary"):
            vector_db.delete_document(selected_source)
            st.success(f"Removed '{selected_source}' from knowledge base")
            st.rerun()
    else:
        st.info("No documents uploaded yet.")



def render_query_section(
    retriever: Retriever,
    ollama_client: OllamaClient,
    agents: List[Agent]
) -> None:
    """
    Render the RAG query section.
    
    Args:
        retriever: Retriever instance for semantic search.
        ollama_client: OllamaClient for LLM inference.
        agents: List of available agents.
    """
    st.subheader("Query Documents")
    
    # Check if documents are available
    if retriever.vector_db.get_document_count() == 0:
        st.warning("⚠️ No documents available for RAG. Please upload documents first.")
        return
    
    # Query input
    query = st.text_area(
        "Enter your question",
        placeholder="Ask a question about your documents...",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        n_results = st.number_input(
            "Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of document chunks to retrieve"
        )
    
    if st.button("🔍 Search & Generate", type="primary", disabled=not query):
        if query and query.strip():
            _execute_rag_query(query, retriever, ollama_client, agents, n_results)


def _execute_rag_query(
    query: str,
    retriever: Retriever,
    ollama_client: OllamaClient,
    agents: List[Agent],
    n_results: int
) -> None:
    """Execute a RAG query and display results."""
    with st.spinner("Searching documents..."):
        # Retrieve relevant chunks
        result = retriever.retrieve(query, n_results=n_results)
        
        if not result.chunks:
            st.warning("No relevant documents found for your query.")
            return
        
        st.session_state.studio_query_result = result
    
    # Display retrieved sources
    with st.expander("📄 Retrieved Sources", expanded=True):
        render_sources(result.chunks)
    
    # Generate response with context
    st.subheader("Generated Response")
    
    # Get system prompt from selected agent
    system_prompt = None
    agent_id = st.session_state.studio_selected_agent_id
    if agent_id:
        agent = next((a for a in agents if a.id == agent_id), None)
        if agent:
            system_prompt = agent.system_prompt
    
    # Build prompt with context
    context = retriever.format_context(result)
    full_prompt = f"{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"
    
    # Stream the response
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        for chunk in ollama_client.generate(full_prompt, system_prompt=system_prompt):
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌")
            if chunk.done:
                break
        
        response_placeholder.markdown(full_response)
        
    except (OllamaConnectionError, ModelNotFoundError) as e:
        st.error(e.user_message)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")


def render_sources(chunks: List[DocumentChunk]) -> None:
    """
    Render the source document chunks.
    
    Args:
        chunks: List of DocumentChunk objects to display.
    """
    if not chunks:
        st.info("No sources to display.")
        return
    
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "Unknown")
        chunk_idx = chunk.metadata.get("chunk_index", "?")
        
        with st.container():
            st.markdown(f"**Source {i}: {source}** (chunk {chunk_idx})")
            st.text(chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content)
            st.divider()
