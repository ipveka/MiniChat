# MiniChat (MiniLM)

A local-first LLM desktop application that runs entirely offline using Ollama. MiniChat provides persistent chat conversations, document-based RAG (Retrieval-Augmented Generation), and customizable agents—all while keeping your data private on your machine.

## Features

- **Persistent Chat**: Have conversations with local LLMs that are saved to SQLite
- **RAG Studio**: Upload documents (PDF, DOCX, TXT, MD) and query them with AI-assisted answers
- **Custom Agents**: Create and manage system prompts to customize LLM behavior for different tasks
- **100% Offline**: All processing happens locally—no data leaves your machine

## Prerequisites

### Python
- Python 3.9 or higher
- pip (Python package manager)

### Ollama

MiniChat requires [Ollama](https://ollama.ai/) to be installed and running locally.

#### Installing Ollama

**Windows:**
1. Download the installer from https://ollama.ai/download/windows
2. Run the installer and follow the prompts
3. Ollama will start automatically as a background service

**macOS:**
1. Download from https://ollama.ai/download/mac
2. Open the downloaded file and drag Ollama to Applications
3. Launch Ollama from Applications

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Starting Ollama

**Windows:**
- Ollama runs automatically after installation
- Look for the Ollama icon in the system tray
- If not running, search for "Ollama" in the Start menu and launch it

**macOS:**
- Launch Ollama from Applications
- Look for the Ollama icon in the menu bar

**Linux:**
```bash
# Start Ollama service
ollama serve
```

#### Pulling a Model

After Ollama is running, pull the default model:

```bash
ollama pull llama3.2
```

You can verify Ollama is running by visiting http://localhost:11434 in your browser.

## Installation

### Option 1: Using setup.py (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd minichat
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the package:
   ```bash
   # Install with all dependencies
   pip install -e .
   
   # Install with development dependencies (for testing)
   pip install -e ".[dev]"
   ```

### Option 2: Using requirements.txt

1. Clone and create virtual environment (same as above)

2. Install dependencies:
   ```bash
   pip install -r miniLM/requirements.txt
   ```

## Running the Application

### Desktop App (Recommended)

Run MiniChat as a native desktop application:

```bash
python miniLM/src/desktop.py
```

Or if installed via setup.py:
```bash
minichat
```

This opens MiniChat in a native window without needing a browser.

### Browser Mode

To run in your web browser instead:

```bash
streamlit run miniLM/src/app.py
```

Or using Python module syntax:
```bash
python -m streamlit run miniLM/src/app.py
```

Then open http://localhost:8501 in your browser.

## Usage

### Chat
The Chat page allows you to have conversations with the LLM:
- Select an agent from the dropdown to apply a specific system prompt
- Type your message and press Enter to send
- Use the "Regenerate" button to get a new response for the last message
- Conversations are automatically saved

### Studio (RAG)
The Studio page enables document-based Q&A:
1. Upload documents (PDF, DOCX, TXT, or MD files)
2. Documents are automatically chunked and embedded
3. Ask questions about your documents
4. View source chunks used to generate answers

### Agents
The Agents page lets you create custom system prompts:
1. Click "Create New Agent"
2. Enter a name, description, and system prompt
3. Save the agent
4. Select your agent in Chat or Studio to use it

## Creating Custom Agents

Agents are system prompts that define how the LLM behaves. Here's an example:

**Name**: Code Reviewer

**Description**: Reviews code for best practices and potential issues

**System Prompt**:
```
You are an expert code reviewer. When given code:
1. Identify potential bugs or issues
2. Suggest improvements for readability
3. Check for security vulnerabilities
4. Recommend best practices
Be constructive and explain your reasoning.
```

### Prebuilt Agent
MiniChat includes an "Equity Research Analyzer" agent as a template. This agent analyzes investment documents and summarizes information about referenced companies.

## Configuration

Configuration is managed in `miniLM/config/settings.py`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `ollama_base_url` | `http://localhost:11434` | Ollama API endpoint |
| `default_model` | `llama3.2` | Default LLM model |
| `embedding_model` | `all-MiniLM-L6-v2` | Model for document embeddings |
| `chunk_size` | `500` | Document chunk size (characters) |
| `chunk_overlap` | `50` | Overlap between chunks |

## Project Structure

```
miniLM/
├── config/
│   └── settings.py      # Application configuration
├── src/
│   ├── app.py           # Main Streamlit application
│   ├── database/        # SQLite and ChromaDB operations
│   ├── llm/             # Ollama client and embeddings
│   ├── rag/             # Document processing and retrieval
│   ├── ui/              # Streamlit UI components
│   └── utils/           # Logging and helper functions
├── data/                # SQLite database and ChromaDB storage
├── logs/                # Application logs
└── requirements.txt     # Python dependencies
```

## Troubleshooting

### "Ollama is not running"
1. **Windows**: Check the system tray for the Ollama icon. If not there, search "Ollama" in Start menu and launch it.
2. **macOS**: Check the menu bar for Ollama. If not there, launch from Applications.
3. **Linux**: Run `ollama serve` in a terminal.
4. Verify by visiting http://localhost:11434 in your browser.

### "Model not found"
- Pull the required model: `ollama pull llama3.2`
- Or change the default model in `miniLM/config/settings.py`

### "Database error"
- Check that the `data/` directory is writable
- Delete `data/minichat.db` to reset the database

### Installation Issues
- Make sure you have Python 3.9+ installed
- Try upgrading pip: `pip install --upgrade pip`
- On Windows, you may need Visual C++ Build Tools for some dependencies

## Running Tests

```bash
# Run all tests
pytest miniLM/tests/ -v

# Run with coverage
pytest miniLM/tests/ -v --cov=miniLM
```

## Building Standalone Executables

You can build MiniChat as a standalone desktop application that doesn't require Python to be installed.

### Prerequisites
```bash
pip install pyinstaller
```

### Build
```bash
python miniLM/installer/build.py
```

This creates a standalone executable in the `dist/MiniChat/` directory:
- **Windows**: `dist/MiniChat/MiniChat.exe`
- **macOS**: `dist/MiniChat.app`
- **Linux**: `dist/MiniChat/MiniChat`

### Clean Build Artifacts
```bash
python miniLM/installer/build.py clean
```

Note: The built application still requires Ollama to be installed and running on the user's system.

## License

See [LICENSE](LICENSE) for details.
