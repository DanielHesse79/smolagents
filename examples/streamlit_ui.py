"""
Daniel's Army of Agents - Multi-Agent System

This setup uses a manager-agent architecture:
- Manager Agent (Mistral): Understands user prompts and delegates tasks
- Programming Agent (DeepSeek R1 8B): Handles all coding and programming tasks

To verify you have the latest DeepSeek R1 8B model, run:
    python examples/check_deepseek_model.py

Setup Options:
1. Using Ollama (local): Set USE_OLLAMA=true environment variable
   - Make sure you have: ollama pull mistral:latest
   - Make sure you have: ollama pull deepseek-r1:8b (or check available models)
   
2. Using API models (default):
   - Set MISTRAL_API_KEY environment variable for Mistral
   - DeepSeek R1 will use Hugging Face Inference API (may need HF_TOKEN)
   - You can specify provider in the code (together, fireworks-ai, etc.)
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

# Add project root and src to path for local imports during development
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from intelcore import CodeAgent, LiteLLMModel, InferenceClientModel, WebSearchTool
from intelcore.default_tools import VisitWebpageTool
from intelcore.streamlit_ui import StreamlitUI

# Import publication tools and helpers
try:
    # Try absolute import first (when run as module)
    try:
        from examples.publication_tools import (
            PubMedSearchTool,
            QdrantUpsertTool,
            SQLUpsertTool,
            MarkdownFileWriterTool,
            ErrorLoggingTool,
            ResearcherUpsertTool,
            ResearcherPublicationLinkTool,
        )
        from examples.publication_helpers import generate_unique_key, format_publication_markdown
    except ImportError:
        # Fall back to relative import (when run directly)
        from publication_tools import (
            PubMedSearchTool,
            QdrantUpsertTool,
            SQLUpsertTool,
            MarkdownFileWriterTool,
            ErrorLoggingTool,
            ResearcherUpsertTool,
            ResearcherPublicationLinkTool,
        )
        from publication_helpers import generate_unique_key, format_publication_markdown
    PUBLICATION_TOOLS_AVAILABLE = True
except ImportError as e:
    PUBLICATION_TOOLS_AVAILABLE = False
    print(f"Warning: Publication tools not available: {e}")

# Try to import QdrantMemoryBackend for persistent memory
try:
    from intelcore.memory_backends import QdrantMemoryBackend
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantMemoryBackend = None


# ============================================================================
# Startup Configuration and Result Classes
# ============================================================================

@dataclass
class StartupConfig:
    """Configuration for startup checks."""
    ollama_base_url: str = "http://localhost:11434"
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    sqlite_db_path: str = "./data/publications.db"
    max_retries: int = 3
    retry_delay: float = 2.0
    required_ollama_models: list[str] = field(default_factory=lambda: (
        [m.strip() for m in os.getenv("OLLAMA_REQUIRED_MODELS", "deepseek-r1:8b,mistral:latest").split(",")]
    ))
    required_qdrant_collections: list[str] = field(default_factory=lambda: ["microsampling_publications", "daniel_army_memory"])


@dataclass
class StartupResult:
    """Results from startup checks."""
    ollama: dict
    qdrant: dict
    sqlite: dict
    publication_tools: bool
    all_critical_services_ready: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ============================================================================
# Retry Helper Function
# ============================================================================

def _retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    delay: float = 2.0,
    *args,
    **kwargs
):
    """Retry function with exponential backoff."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                time.sleep(wait_time)
            else:
                raise
    if last_exception:
        raise last_exception


# ============================================================================
# Health Check Functions
# ============================================================================

def check_ollama_health(
    base_url: str = "http://localhost:11434",
    max_retries: int = 3,
    retry_delay: float = 2.0,
    timeout: float = 5.0,
    required_models: Optional[list[str]] = None
) -> dict:
    """
    Check Ollama availability with retry logic.
    
    Returns:
        dict: {
            "available": bool,
            "url": str,
            "version": str | None,
            "models": list[str],
            "required_models": dict[str, bool],
            "error": str | None
        }
    """
    if required_models is None:
        # Get from environment variable or use defaults
        env_models = os.getenv("OLLAMA_REQUIRED_MODELS", "deepseek-r1:8b,mistral:latest")
        required_models = [m.strip() for m in env_models.split(",")] if env_models else ["deepseek-r1:8b", "mistral:latest"]
    
    result = {
        "available": False,
        "url": base_url,
        "version": None,
        "models": [],
        "required_models": {model: False for model in required_models},
        "error": None
    }
    
    try:
        import requests
        
        def check_connection():
            response = requests.get(f"{base_url}/api/tags", timeout=timeout)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama returned status code {response.status_code}")
            return response
        
        response = _retry_with_backoff(check_connection, max_retries, retry_delay)
        
        # Get models
        models_data = response.json().get("models", [])
        result["models"] = [m.get("name", "") for m in models_data]
        
        # Check required models
        model_names_lower = [m.lower() for m in result["models"]]
        for required_model in required_models:
            # Check for exact match or partial match
            found = False
            for model_name in result["models"]:
                if required_model.lower() in model_name.lower() or model_name.lower() in required_model.lower():
                    found = True
                    break
            result["required_models"][required_model] = found
        
        # Get version
        try:
            version_response = requests.get(f"{base_url}/api/version", timeout=timeout)
            if version_response.status_code == 200:
                result["version"] = version_response.json().get("version", "unknown")
        except Exception:
            pass  # Version is optional
        
        result["available"] = True
        
    except ImportError:
        result["error"] = "requests library not installed"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_qdrant_health(
    url: str = "localhost",
    port: int = 6333,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    timeout: float = 5.0,
    required_collections: Optional[list[str]] = None
) -> dict:
    """
    Check Qdrant availability with retry logic.
    
    Returns:
        dict: {
            "available": bool,
            "url": str,
            "port": int,
            "collections": list[str],
            "required_collections": dict[str, bool],
            "error": str | None
        }
    """
    if required_collections is None:
        required_collections = ["microsampling_publications", "daniel_army_memory"]
    
    result = {
        "available": False,
        "url": url,
        "port": port,
        "collections": [],
        "required_collections": {coll: False for coll in required_collections},
        "error": None
    }
    
    if not QDRANT_AVAILABLE:
        result["error"] = "Qdrant dependencies not installed. Install with: pip install 'intelcore[qdrant]'"
        return result
    
    try:
        from qdrant_client import QdrantClient
        
        def check_connection():
            client = QdrantClient(url=url, port=port, timeout=timeout)
            collections = client.get_collections()
            return client, collections
        
        client, collections_response = _retry_with_backoff(check_connection, max_retries, retry_delay)
        
        # Get collection names
        result["collections"] = [c.name for c in collections_response.collections]
        
        # Check required collections
        for required_coll in required_collections:
            result["required_collections"][required_coll] = required_coll in result["collections"]
        
        result["available"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_sqlite_health(db_path: str = "./data/publications.db") -> dict:
    """
    Check SQLite database availability and schema.
    
    Returns:
        dict: {
            "available": bool,
            "path": str,
            "tables": list[str],
            "required_tables": dict[str, bool],
            "error": str | None
        }
    """
    import sqlite3
    
    result = {
        "available": False,
        "path": db_path,
        "tables": [],
        "required_tables": {
            "publications": False,
            "agent_errors": False,
            "researchers": False,
            "researcher_publications": False,
            "agent_learning_patterns": False
        },
        "error": None
    }
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        result["tables"] = [row[0] for row in cursor.fetchall()]
        
        # Check required tables
        for table_name in result["required_tables"].keys():
            result["required_tables"][table_name] = table_name in result["tables"]
        
        conn.close()
        result["available"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ============================================================================
# Startup Checks Orchestration
# ============================================================================

def run_startup_checks(config: StartupConfig) -> StartupResult:
    """
    Run all startup checks with retry logic.
    
    Returns:
        StartupResult with status of all services
    """
    warnings = []
    errors = []
    
    # Check Ollama
    print("Checking Ollama...")
    ollama_status = check_ollama_health(
        base_url=config.ollama_base_url,
        max_retries=config.max_retries,
        retry_delay=config.retry_delay,
        required_models=config.required_ollama_models
    )
    
    if not ollama_status["available"]:
        warnings.append(f"Ollama not available: {ollama_status.get('error', 'Unknown error')}")
    else:
        missing_models = [m for m, found in ollama_status["required_models"].items() if not found]
        if missing_models:
            warnings.append(f"Missing Ollama models: {', '.join(missing_models)}")
    
    # Check Qdrant
    print("Checking Qdrant...")
    qdrant_status = check_qdrant_health(
        url=config.qdrant_url,
        port=config.qdrant_port,
        max_retries=config.max_retries,
        retry_delay=config.retry_delay,
        required_collections=config.required_qdrant_collections
    )
    
    if not qdrant_status["available"]:
        warnings.append(f"Qdrant not available: {qdrant_status.get('error', 'Unknown error')}")
    else:
        missing_collections = [c for c, found in qdrant_status["required_collections"].items() if not found]
        if missing_collections:
            warnings.append(f"Missing Qdrant collections (will be created): {', '.join(missing_collections)}")
    
    # Check SQLite
    print("Checking SQLite...")
    sqlite_status = check_sqlite_health(db_path=config.sqlite_db_path)
    
    if not sqlite_status["available"]:
        errors.append(f"SQLite not available: {sqlite_status.get('error', 'Unknown error')}")
    else:
        missing_tables = [t for t, found in sqlite_status["required_tables"].items() if not found]
        if missing_tables:
            warnings.append(f"Missing SQLite tables (will be created): {', '.join(missing_tables)}")
    
    # Check publication tools
    publication_tools_available = PUBLICATION_TOOLS_AVAILABLE
    if not publication_tools_available:
        warnings.append("Publication tools not available (some features may be limited)")
    
    # Determine if critical services are ready
    # SQLite is critical, Ollama and Qdrant are optional
    all_critical_ready = sqlite_status["available"]
    
    return StartupResult(
        ollama=ollama_status,
        qdrant=qdrant_status,
        sqlite=sqlite_status,
        publication_tools=publication_tools_available,
        all_critical_services_ready=all_critical_ready,
        warnings=warnings,
        errors=errors
    )


# ============================================================================
# Startup Screen (Streamlit UI)
# ============================================================================

def render_startup_screen(startup_result: StartupResult) -> bool:
    """
    Render startup status screen in Streamlit.
    
    Returns:
        bool: True if user wants to continue, False to retry
    """
    import streamlit as st
    
    st.title("🚀 Daniel's Army of Agents - System Startup")
    st.markdown("Checking system dependencies...")
    
    # Overall status
    if startup_result.all_critical_services_ready:
        st.success("✅ All critical services are ready!")
    else:
        st.error("❌ Some critical services are not available")
    
    # Ollama status
    with st.expander("🦙 Ollama Status", expanded=True):
        if startup_result.ollama["available"]:
            st.success("✅ Ollama is running")
            if startup_result.ollama.get("version"):
                st.info(f"Version: {startup_result.ollama['version']}")
            
            st.markdown("**Available Models:**")
            if startup_result.ollama["models"]:
                for model in startup_result.ollama["models"]:
                    st.markdown(f"- {model}")
            else:
                st.warning("No models found")
            
            st.markdown("**Required Models:**")
            for model, found in startup_result.ollama["required_models"].items():
                status = "✅" if found else "❌"
                st.markdown(f"{status} {model}")
                if not found:
                    st.caption(f"Install with: `ollama pull {model}`")
        else:
            st.error("❌ Ollama is not available")
            if startup_result.ollama.get("error"):
                st.error(f"Error: {startup_result.ollama['error']}")
            st.info("💡 Tip: Start Ollama or use API models instead")
    
    # Qdrant status
    with st.expander("🗄️ Qdrant Status", expanded=True):
        if startup_result.qdrant["available"]:
            st.success("✅ Qdrant is running")
            st.info(f"URL: {startup_result.qdrant['url']}:{startup_result.qdrant['port']}")
            
            st.markdown("**Available Collections:**")
            if startup_result.qdrant["collections"]:
                for coll in startup_result.qdrant["collections"]:
                    st.markdown(f"- {coll}")
            else:
                st.info("No collections yet (will be created as needed)")
            
            st.markdown("**Required Collections:**")
            for coll, found in startup_result.qdrant["required_collections"].items():
                status = "✅" if found else "⚠️"
                st.markdown(f"{status} {coll}")
                if not found:
                    st.caption("Will be created automatically")
        else:
            st.warning("⚠️ Qdrant is not available")
            if startup_result.qdrant.get("error"):
                st.error(f"Error: {startup_result.qdrant['error']}")
            st.info("💡 Tip: Start Qdrant with: `docker run -p 6333:6333 qdrant/qdrant`")
    
    # SQLite status
    with st.expander("💾 SQLite Status", expanded=True):
        if startup_result.sqlite["available"]:
            st.success("✅ SQLite database is ready")
            st.info(f"Path: {startup_result.sqlite['path']}")
            
            st.markdown("**Available Tables:**")
            if startup_result.sqlite["tables"]:
                for table in startup_result.sqlite["tables"]:
                    st.markdown(f"- {table}")
            else:
                st.info("No tables yet (will be created)")
            
            st.markdown("**Required Tables:**")
            for table, found in startup_result.sqlite["required_tables"].items():
                status = "✅" if found else "⚠️"
                st.markdown(f"{status} {table}")
                if not found:
                    st.caption("Will be created automatically")
        else:
            st.error("❌ SQLite is not available")
            if startup_result.sqlite.get("error"):
                st.error(f"Error: {startup_result.sqlite['error']}")
    
    # Publication tools status
    with st.expander("📚 Publication Tools Status", expanded=False):
        if startup_result.publication_tools:
            st.success("✅ Publication tools are available")
        else:
            st.warning("⚠️ Publication tools are not available")
            st.info("Some features may be limited")
    
    # Warnings and errors
    if startup_result.warnings:
        st.warning("⚠️ **Warnings:**")
        for warning in startup_result.warnings:
            st.markdown(f"- {warning}")
    
    if startup_result.errors:
        st.error("❌ **Errors:**")
        for error in startup_result.errors:
            st.markdown(f"- {error}")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Retry Checks", use_container_width=True):
            st.session_state.startup_retry = True
            st.rerun()
    
    with col2:
        if startup_result.all_critical_services_ready:
            if st.button("✅ Continue to App", type="primary", use_container_width=True):
                return True
        else:
            st.button("❌ Continue Anyway", disabled=True, use_container_width=True)
            st.caption("Fix errors before continuing")
    
    return False


# ============================================================================
# Service Initialization Functions
# ============================================================================

def initialize_memory_backend(qdrant_status: dict):
    """Initialize Qdrant memory backend based on health check status."""
    if not qdrant_status.get("available", False):
        return None
    
    if not QDRANT_AVAILABLE:
        return None
    
    try:
        backend = QdrantMemoryBackend(
            url=qdrant_status.get("url", "localhost"),
            port=qdrant_status.get("port", 6333),
            collection_name="daniel_army_memory",
            embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
        )
        print("✅ Qdrant memory backend initialized successfully")
        return backend
    except Exception as exc:
        print(f"⚠️  Warning: Could not initialize Qdrant memory backend: {exc}")
        print("Agent will continue without persistent memory")
        return None


def initialize_qdrant_client(qdrant_status: dict):
    """Initialize Qdrant client and create collections if needed."""
    if not qdrant_status.get("available", False):
        return None
    
    if not QDRANT_AVAILABLE:
        return None
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from sentence_transformers import SentenceTransformer
        
        client = QdrantClient(
            url=qdrant_status.get("url", "localhost"),
            port=qdrant_status.get("port", 6333)
        )
        collection_name = "microsampling_publications"
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            # Create collection
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embedding_dim = embedder.get_sentence_embedding_dimension()
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Created Qdrant collection '{collection_name}'")
        else:
            print(f"✅ Qdrant collection '{collection_name}' already exists")
        
        return client
    except Exception as e:
        print(f"⚠️  Warning: Could not setup Qdrant collection: {e}")
        return None


def initialize_sqlite_db(sqlite_status: dict):
    """Initialize SQLite database and create all tables if needed."""
    db_path = sqlite_status.get("path", "./data/publications.db")
    
    try:
        import sqlite3
        from datetime import datetime
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Publications table (existing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS publications (
                unique_key TEXT PRIMARY KEY,
                title TEXT,
                year INTEGER,
                doi TEXT,
                pmid TEXT,
                url TEXT,
                device_workflow TEXT,
                brand TEXT,
                sample_type TEXT,
                application TEXT,
                evidence_snippet TEXT,
                authors TEXT,
                corresponding_author TEXT,
                affiliation TEXT,
                source TEXT,
                retrieved_at TEXT,
                abstract TEXT,
                journal TEXT
            )
        """)
        
        # 2. Agent errors table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_errors (
                error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                code_snippet TEXT NOT NULL,
                step_number INTEGER,
                task_context TEXT,
                corrected_code TEXT,
                fix_attempts INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                agent_name TEXT,
                session_id TEXT,
                resolved BOOLEAN DEFAULT 0,
                resolution_notes TEXT
            )
        """)
        
        # 3. Researchers table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS researchers (
                researcher_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                affiliation TEXT,
                research_fields TEXT,
                relevance_reason TEXT,
                first_encountered TEXT,
                publications_count INTEGER DEFAULT 0,
                contact_info TEXT,
                notes TEXT,
                last_updated TEXT,
                UNIQUE(name, email)
            )
        """)
        
        # 4. Researcher-publications junction table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS researcher_publications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                researcher_id INTEGER NOT NULL,
                publication_key TEXT NOT NULL,
                role TEXT,
                author_position INTEGER,
                FOREIGN KEY (researcher_id) REFERENCES researchers(researcher_id),
                FOREIGN KEY (publication_key) REFERENCES publications(unique_key),
                UNIQUE(researcher_id, publication_key)
            )
        """)
        
        # 5. Agent learning patterns table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_learning_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT,
                common_cause TEXT,
                solution_pattern TEXT,
                frequency INTEGER DEFAULT 1,
                last_seen TEXT,
                effectiveness_score REAL
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ SQLite database initialized with all tables: {db_path}")
        return db_path
    except Exception as e:
        print(f"⚠️  Warning: Could not setup SQL database: {e}")
        return None


# ============================================================================
# Model Setup Functions
# ============================================================================

def setup_ollama_models(ollama_status: dict):
    """Setup Ollama models based on health check status."""
    base_url = ollama_status.get("url", "http://localhost:11434")
    
    # Default model names
    programming_model_name = "deepseek-r1:8b"
    manager_model_name = "mistral:latest"
    
    # Try to find models from available models
    available_models = ollama_status.get("models", [])
    required_models = ollama_status.get("required_models", {})
    
    # Find DeepSeek model
    for model in available_models:
        if "deepseek" in model.lower() and "r1" in model.lower():
            programming_model_name = model
            break
        elif "deepseek" in model.lower() and programming_model_name == "deepseek-r1:8b":
            programming_model_name = model
    
    # Find Mistral model
    for model in available_models:
        if "mistral" in model.lower():
            manager_model_name = model
            break
    
    # Use required models if found
    for model, found in required_models.items():
        if found and "deepseek" in model.lower():
            programming_model_name = model
        elif found and "mistral" in model.lower():
            manager_model_name = model
    
    programming_model = LiteLLMModel(
        model_id=f"ollama_chat/{programming_model_name}",
        api_base=base_url,
        api_key="ollama",
        timeout=120,
        max_tokens=4096,
    )
    
    manager_model = LiteLLMModel(
        model_id=f"ollama_chat/{manager_model_name}",
        api_base=base_url,
        api_key="ollama",
        timeout=120,
        max_tokens=4096,
    )
    
    print(f"✅ Using Ollama models: {programming_model_name} (programming), {manager_model_name} (manager)")
    
    return programming_model, manager_model


def setup_api_models():
    """Setup API models as fallback."""
    print("Using API models")
    
    programming_model = InferenceClientModel(
        model_id="deepseek-ai/DeepSeek-R1",
        provider="together",
    )
    
    manager_model = LiteLLMModel(
        model_id="mistral/mistral-large-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
    
    print("✅ Using API models: DeepSeek-R1 (programming), Mistral Large (manager)")
    
    return programming_model, manager_model


# ============================================================================
# Agent Creation Functions
# ============================================================================

def create_programming_agent(
    model,
    memory_backend,
    db_path: Optional[str] = None,
    qdrant_collection_name: str = "microsampling_publications"
):
    """Create the programming agent with appropriate tools."""
    print("Creating programming agent (DeepSeek R1 8B)...")
    
    # Prepare tools for programming agent
    programming_tools = []
    if PUBLICATION_TOOLS_AVAILABLE:
        db_path_final = db_path or "./data/publications.db"
        programming_tools = [
            QdrantUpsertTool(collection_name=qdrant_collection_name),
            SQLUpsertTool(db_path=db_path_final),
            MarkdownFileWriterTool(output_dir="./data"),
            ErrorLoggingTool(db_path=db_path_final),
            ResearcherUpsertTool(db_path=db_path_final),
            ResearcherPublicationLinkTool(db_path=db_path_final),
        ]
        print("✅ Publication tools and error/researcher tools added to programming agent")
    
    programming_agent = CodeAgent(
        tools=programming_tools,
        model=model,
        name="programmer",
        description="A specialized programming agent that writes, debugs, and executes code. Use this agent when you need to write Python code, solve programming problems, create scripts, or perform computational tasks.",
        verbosity_level=1,
        stream_outputs=True,
        max_steps=50,
        memory_backend=memory_backend,
        enable_experience_retrieval=True,
        additional_authorized_imports=[
            "os",
            "csv",
            "json",
            "hashlib",
            "sqlite3",
            "datetime",
            "pathlib",
        ],
        instructions="""You are an expert Python programming agent. Your code must be syntactically correct and follow Python best practices.

CRITICAL PYTHON SYNTAX RULES:
1. NEVER use '...' (ellipsis) in variable unpacking. This causes SyntaxError.
   WRONG: title, authors, ... = data.split('\\n')
   CORRECT: title, authors, *rest = data.split('\\n')  # Use *rest to capture remaining
   CORRECT: title, authors = data.split('\\n')[:2]     # Unpack only what you need
   CORRECT: parts = data.split('\\n'); title = parts[0]  # Use indexing

2. Variable unpacking patterns:
   - For known number: a, b, c = items[:3]
   - For variable number: first, *rest = items
   - For specific positions: parts = items.split(); title = parts[0]

3. Always validate your code syntax mentally before writing it.

4. Use defensive programming:
   - Check list length before indexing: if len(items) > 0: first = items[0]
   - Use try/except for error handling
   - Validate inputs before processing

5. Code quality:
   - Write clear, readable code
   - Use meaningful variable names
   - Add comments for complex logic
   - Break complex operations into steps

6. Error handling:
   - Always handle potential errors (KeyError, IndexError, etc.)
   - Provide meaningful error messages
   - Use try/except blocks appropriately

PUBLICATION MINING TASKS:
When working with publication mining tasks:
- Use qdrant_upsert_publication to store publications in vector database
- Use sql_upsert_publication to store publications in SQL database
- Use write_markdown_file to save structured output
- Always generate unique keys using DOI, PMID, or hash
- Deduplicate publications before storing
- Structure markdown output clearly with headers and sections

ERROR HANDLING:
- When you encounter an error, use log_agent_error to record it for learning
- This helps identify patterns and improve future performance
- Include error_type, error_message, code_snippet, and context

RESEARCHER MANAGEMENT:
- When you find relevant researchers, use upsert_researcher to store their information
- Include their research fields, affiliation, and why they're relevant
- Use link_researcher_publication to connect researchers to their publications

If you make a syntax error, immediately fix it in the next step with correct Python syntax.
""",
    )
    print("✅ Programming agent created successfully")
    return programming_agent


def create_manager_agent(
    model,
    programming_agent,
    memory_backend
):
    """Create the manager agent with appropriate tools."""
    print("Creating manager agent (Mistral)...")
    
    # Prepare tools for manager agent
    manager_tools = [
        WebSearchTool(max_results=20),
        VisitWebpageTool(max_output_length=30_000),
    ]
    
    # Add PubMed search tool and researcher tools if available
    if PUBLICATION_TOOLS_AVAILABLE:
        manager_tools.append(PubMedSearchTool())
        # Manager can also log errors and track researchers
        db_path_final = "./data/publications.db"  # Default path for manager
        manager_tools.append(ErrorLoggingTool(db_path=db_path_final))
        manager_tools.append(ResearcherUpsertTool(db_path=db_path_final))
        print("✅ PubMed search tool and error/researcher tools added to manager agent")
    
    manager_agent = CodeAgent(
        tools=manager_tools,
        model=model,
        managed_agents=[programming_agent],
        verbosity_level=1,
        planning_interval=3,
        stream_outputs=True,
        max_steps=20,
        memory_backend=memory_backend,
        enable_experience_retrieval=True,
        additional_authorized_imports=[
            "os",
            "csv",
            "json",
            "hashlib",
            "sqlite3",
            "datetime",
            "pathlib",
        ],
        instructions="""You are a manager agent that understands user requests and delegates programming tasks.

When delegating to the 'programmer' agent, provide:
- Clear, detailed task descriptions
- Expected inputs and outputs
- Any constraints or requirements
- Examples if helpful

IMPORTANT: When the programmer makes syntax errors (especially with '...' in unpacking), 
in your next delegation, remind them to use '*rest' or indexing instead.

When a user asks for:
- Code writing, debugging, or execution
- Programming problems or algorithms
- Script creation or automation
- Data processing or analysis
- Computational tasks
- Publication mining, searching, or storage
- File writing (markdown, CSV, etc.)
- Database operations

Delegate these to the 'programmer' agent by calling it with a detailed task description.

PUBLICATION MINING TASKS:
When users request publication mining (e.g., finding microsampling publications):
1. Use pubmed_search or web_search to find publications
2. Use visit_webpage to get detailed information
3. Delegate to 'programmer' agent for:
   - Parsing and structuring publication data
   - Deduplication using unique keys (DOI, PMID, or hash)
   - Storing in Qdrant (vector database) using qdrant_upsert_publication
   - Storing in SQL database using sql_upsert_publication
   - Writing markdown files using write_markdown_file
   - Extracting and storing researcher information using upsert_researcher
   - Linking researchers to publications using link_researcher_publication
4. Ensure the programmer generates unique keys and handles deduplication
5. Verify output format matches requirements (30+ publications, structured markdown)

ERROR HANDLING:
- If the programmer makes errors, use log_agent_error to record them
- This helps build a knowledge base of common mistakes and solutions

RESEARCHER TRACKING:
- When you find relevant researchers, use upsert_researcher to store their information
- Track why they're relevant (relevance_reason field)
- Link them to publications using link_researcher_publication

For other tasks (web search, information gathering), handle them yourself using your tools.
""",
    )
    print("✅ Manager agent created successfully")
    return manager_agent


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main startup sequence."""
    import streamlit as st
    
    # Initialize session state
    if "startup_complete" not in st.session_state:
        st.session_state.startup_complete = False
    if "startup_result" not in st.session_state:
        st.session_state.startup_result = None
    if "agents_initialized" not in st.session_state:
        st.session_state.agents_initialized = False
    
    # Run startup checks if not already done
    if not st.session_state.startup_complete:
        # Check if retry was requested
        if st.session_state.get("startup_retry", False):
            st.session_state.startup_retry = False
            st.session_state.startup_result = None
        
        # Run checks if not cached
        if st.session_state.startup_result is None:
            config = StartupConfig()
            st.session_state.startup_result = run_startup_checks(config)
        
        # Show startup screen
        if render_startup_screen(st.session_state.startup_result):
            st.session_state.startup_complete = True
            st.rerun()
        else:
            st.stop()
    
    # Initialize services and agents only once
    if not st.session_state.agents_initialized:
        startup_result = st.session_state.startup_result
        
        print("Initializing services...")
        memory_backend = initialize_memory_backend(startup_result.qdrant)
        qdrant_client = initialize_qdrant_client(startup_result.qdrant)
        db_path = initialize_sqlite_db(startup_result.sqlite)
        
        # Setup models based on Ollama status
        if startup_result.ollama["available"]:
            programming_model, manager_model = setup_ollama_models(startup_result.ollama)
        else:
            programming_model, manager_model = setup_api_models()
        
        # Create agents
        programming_agent = create_programming_agent(
            model=programming_model,
            memory_backend=memory_backend,
            db_path=db_path,
            qdrant_collection_name="microsampling_publications"
        )
        
        manager_agent = create_manager_agent(
            model=manager_model,
            programming_agent=programming_agent,
            memory_backend=memory_backend
        )
        
        # Store agents in session state
        st.session_state.manager_agent = manager_agent
        st.session_state.agents_initialized = True
    
    # Initialize and run Streamlit UI
    streamlit_ui = StreamlitUI(st.session_state.manager_agent, file_upload_folder="./data")
    streamlit_ui.name = "Daniel's Army of Agents"
    streamlit_ui.run()


# ============================================================================
# Legacy Functions (kept for backward compatibility)
# ============================================================================

def build_memory_backend():
    """Create a Qdrant backend when optional deps are installed."""
    if not QDRANT_AVAILABLE:
        print("Warning: Qdrant dependencies not installed. Agent will continue without persistent memory.")
        print("Install with: pip install 'intelcore[qdrant]'")
        return None

    try:
        backend = QdrantMemoryBackend(
            url="localhost",
            port=6333,
            collection_name="daniel_army_memory",
            embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
        )
        print("Qdrant memory backend initialized successfully")
        return backend
    except Exception as exc:  # pragma: no cover - defensive logging only
        print(f"Warning: Could not initialize Qdrant memory backend: {exc}")
        print("Agent will continue without persistent memory")
        return None


def setup_publication_collection():
    """Create Qdrant collection for publications if it doesn't exist."""
    if not QDRANT_AVAILABLE:
        return None
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from sentence_transformers import SentenceTransformer
        
        client = QdrantClient(url="localhost", port=6333)
        collection_name = "microsampling_publications"
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            # Create collection
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embedding_dim = embedder.get_sentence_embedding_dimension()
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Created Qdrant collection '{collection_name}'")
        else:
            print(f"✅ Qdrant collection '{collection_name}' already exists")
        
        return client
    except Exception as e:
        print(f"⚠️  Warning: Could not setup Qdrant collection: {e}")
        return None


def setup_publications_db():
    """Create SQLite database and tables for publications."""
    import sqlite3
    
    db_path = "./data/publications.db"
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS publications (
                unique_key TEXT PRIMARY KEY,
                title TEXT,
                year INTEGER,
                doi TEXT,
                pmid TEXT,
                url TEXT,
                device_workflow TEXT,
                brand TEXT,
                sample_type TEXT,
                application TEXT,
                evidence_snippet TEXT,
                authors TEXT,
                corresponding_author TEXT,
                affiliation TEXT,
                source TEXT,
                retrieved_at TEXT,
                abstract TEXT,
                journal TEXT
            )
        """)
        conn.commit()
        conn.close()
        print(f"✅ SQLite database initialized: {db_path}")
        return db_path
    except Exception as e:
        print(f"⚠️  Warning: Could not setup SQL database: {e}")
        return None


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
else:
    # When run via streamlit run, call main directly
    main()
