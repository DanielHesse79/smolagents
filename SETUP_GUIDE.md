# Setup Guide for smolagents

This guide will help you set up everything needed to run the smolagents application.

## Prerequisites

### 1. Python 3.10 or newer

**Windows Installation:**
1. Download Python from https://www.python.org/downloads/
2. **IMPORTANT:** During installation, check the box "Add Python to PATH"
3. Restart your terminal/PowerShell after installation
4. Verify installation:
   ```powershell
   python --version
   # Should show: Python 3.10.x or higher
   ```

**Alternative (Microsoft Store):**
- Search for "Python 3.12" or "Python 3.11" in Microsoft Store
- Install and restart terminal

### 2. Install Dependencies

Once Python is installed, run:

```powershell
# Navigate to the project directory
cd C:\Users\DanielsGPU\Documents\GitHub\smolagents

# Install smolagents with all optional dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

Or use the setup script:
```powershell
.\setup.ps1
```

### 3. Ollama Models (Optional but Recommended)

Ollama is already installed on your system. Pull the required models:

```powershell
ollama pull deepseek-r1:8b
ollama pull mistral:latest
```

Verify models are installed:
```powershell
ollama list
```

### 4. Docker (Optional - for Qdrant)

If you want to use Qdrant for persistent memory:

1. Download Docker Desktop from https://www.docker.com/products/docker-desktop
2. Install and start Docker Desktop
3. Run Qdrant:
   ```powershell
   docker run -p 6333:6333 qdrant/qdrant
   ```

### 5. Environment Variables (Optional)

Create a `.env` file in the project root with your API keys (if using API models):

```env
# Hugging Face API token (optional)
HF_TOKEN=your_token_here
HF_API_KEY=your_token_here

# Mistral API key (optional, if not using Ollama)
MISTRAL_API_KEY=your_key_here

# OpenAI API key (optional)
OPENAI_API_KEY=your_key_here

# Search API keys (optional)
SERPAPI_API_KEY=your_key_here
SERPER_API_KEY=your_key_here
```

## Running the Application

### Gradio UI
```powershell
python examples/gradio_ui.py
```

## Troubleshooting

### Python not found
- Make sure Python is added to PATH
- Restart your terminal after installing Python
- Try using `py` instead of `python` on Windows

### Import errors
- Make sure you installed with: `pip install -e ".[all]"`
- Try: `pip install --upgrade smolagents`

### Ollama connection errors
- Make sure Ollama is running: `ollama serve` (usually runs automatically)
- Check if models are installed: `ollama list`

### Qdrant connection errors
- Make sure Docker is running
- Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
- Or skip Qdrant - it's optional

## What's Already Done

✅ Created `examples/shared_agent_utils.py` - shared utility functions
✅ Fixed function signature mismatches in `gradio_ui.py`
✅ Created setup script (`setup.ps1`)
✅ Ollama is installed (version 0.13.2)

## Next Steps

1. **Install Python 3.10+** (if not already installed)
2. **Run the setup script**: `.\setup.ps1`
3. **Pull Ollama models**: `ollama pull deepseek-r1:8b` and `ollama pull mistral:latest`
4. **Run the app**: `python examples/gradio_ui.py`

