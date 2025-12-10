"""
Shared utility functions for agent setup and initialization.

This module provides common functions used by both streamlit_ui.py and gradio_ui.py
for health checks, service initialization, and agent creation.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

# Try to import QdrantMemoryBackend for persistent memory
try:
    from smolagents.memory_backends import QdrantMemoryBackend
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantMemoryBackend = None

# Check if publication tools are available
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
    PUBLICATION_TOOLS_AVAILABLE = True
except ImportError:
    try:
        from publication_tools import (
            PubMedSearchTool,
            QdrantUpsertTool,
            SQLUpsertTool,
            MarkdownFileWriterTool,
            ErrorLoggingTool,
            ResearcherUpsertTool,
            ResearcherPublicationLinkTool,
        )
        PUBLICATION_TOOLS_AVAILABLE = True
    except ImportError:
        PUBLICATION_TOOLS_AVAILABLE = False


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
    required_ollama_models: list[str] = field(default_factory=lambda: ["deepseek-r1:8b", "mistral:latest"])
    required_qdrant_collections: list[str] = field(default_factory=lambda: ["microsampling_publications", "daniel_army_memory"])
    # Additional config attributes for gradio_ui.py
    ollama_timeout: int = 120
    ollama_max_tokens: int = 4096
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.7
    ollama_top_p: float = 0.9


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
        required_models = ["deepseek-r1:8b", "mistral:latest"]
    
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
        result["error"] = "Qdrant dependencies not installed. Install with: pip install 'smolagents[qdrant]'"
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
        print("[OK] Qdrant memory backend initialized successfully")
        return backend
    except Exception as exc:
        print(f"[WARN] Could not initialize Qdrant memory backend: {exc}")
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
            print(f"[OK] Created Qdrant collection '{collection_name}'")
        else:
            print(f"[OK] Qdrant collection '{collection_name}' already exists")
        
        return client
    except Exception as e:
        print(f"[WARN] Could not setup Qdrant collection: {e}")
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
        print(f"[OK] SQLite database initialized with all tables: {db_path}")
        return db_path
    except Exception as e:
        print(f"[WARN] Could not setup SQL database: {e}")
        return None


# ============================================================================
# Model Setup Functions
# ============================================================================

def setup_ollama_models(ollama_status: dict, config: Optional[StartupConfig] = None):
    """Setup Ollama models based on health check status."""
    # #region debug log
    import json; f=open(r'c:\Users\DanielsGPU\Documents\GitHub\smolagents\.cursor\debug.log','a',encoding='utf-8'); f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"shared_agent_utils.py:571","message":"setup_ollama_models entry","data":{"config_type":type(config).__name__ if config else None,"ollama_url":ollama_status.get("url")},"timestamp":int(__import__('time').time()*1000)})+'\n'); f.close()
    # #endregion
    base_url = ollama_status.get("url", "http://localhost:11434")
    
    # Default config values
    timeout = 120
    max_tokens = 4096
    num_ctx = 8192
    temperature = 0.7
    top_p = 0.9
    
    if config:
        timeout = config.ollama_timeout
        max_tokens = config.ollama_max_tokens
        num_ctx = config.ollama_num_ctx
        temperature = config.ollama_temperature
        top_p = config.ollama_top_p
    
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
    
    from smolagents import LiteLLMModel
    
    # #region debug log
    f=open(r'c:\Users\DanielsGPU\Documents\GitHub\smolagents\.cursor\debug.log','a',encoding='utf-8'); f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"shared_agent_utils.py:620","message":"Before LiteLLMModel init (programming)","data":{"model_name":programming_model_name,"base_url":base_url},"timestamp":int(__import__('time').time()*1000)})+'\n'); f.close()
    # #endregion
    try:
        programming_model = LiteLLMModel(
                model_id=f"ollama_chat/{programming_model_name}",
                api_base=base_url,
                api_key="ollama",
                timeout=timeout,
                max_tokens=max_tokens,
                num_ctx=num_ctx,
                temperature=temperature,
                top_p=top_p,
            )
        # #region debug log
        f=open(r'c:\Users\DanielsGPU\Documents\GitHub\smolagents\.cursor\debug.log','a',encoding='utf-8'); f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"shared_agent_utils.py:631","message":"After LiteLLMModel init (programming)","data":{"model_type":type(programming_model).__name__},"timestamp":int(__import__('time').time()*1000)})+'\n'); f.close()
        # #endregion
    except Exception as e:
        # #region debug log
        f=open(r'c:\Users\DanielsGPU\Documents\GitHub\smolagents\.cursor\debug.log','a',encoding='utf-8'); f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"shared_agent_utils.py:634","message":"LiteLLMModel init exception (programming)","data":{"error_type":type(e).__name__,"error_msg":str(e)},"timestamp":int(__import__('time').time()*1000)})+'\n'); f.close()
        # #endregion
        raise
    
    # #region debug log
    f=open(r'c:\Users\DanielsGPU\Documents\GitHub\smolagents\.cursor\debug.log','a',encoding='utf-8'); f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"shared_agent_utils.py:637","message":"Before LiteLLMModel init (manager)","data":{"model_name":manager_model_name},"timestamp":int(__import__('time').time()*1000)})+'\n'); f.close()
    # #endregion
    try:
        manager_model = LiteLLMModel(
                model_id=f"ollama_chat/{manager_model_name}",
                api_base=base_url,
                api_key="ollama",
                timeout=timeout,
                max_tokens=max_tokens,
                num_ctx=num_ctx,
                temperature=temperature,
                top_p=top_p,
            )
        # #region debug log
        f=open(r'c:\Users\DanielsGPU\Documents\GitHub\smolagents\.cursor\debug.log','a',encoding='utf-8'); f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"shared_agent_utils.py:649","message":"After LiteLLMModel init (manager)","data":{"model_type":type(manager_model).__name__},"timestamp":int(__import__('time').time()*1000)})+'\n'); f.close()
        # #endregion
    except Exception as e:
        # #region debug log
        f=open(r'c:\Users\DanielsGPU\Documents\GitHub\smolagents\.cursor\debug.log','a',encoding='utf-8'); f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"shared_agent_utils.py:652","message":"LiteLLMModel init exception (manager)","data":{"error_type":type(e).__name__,"error_msg":str(e)},"timestamp":int(__import__('time').time()*1000)})+'\n'); f.close()
        # #endregion
        raise
    
    print(f"[OK] Using Ollama models: {programming_model_name} (programming), {manager_model_name} (manager)")
    
    return programming_model, manager_model


def setup_api_models():
    """Setup API models as fallback."""
    from smolagents import InferenceClientModel, LiteLLMModel
    
    print("Using API models")
    
    programming_model = InferenceClientModel(
        model_id="deepseek-ai/DeepSeek-R1",
        provider="together",
    )
    
    manager_model = LiteLLMModel(
        model_id="mistral/mistral-large-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
    
    print("[OK] Using API models: DeepSeek-R1 (programming), Mistral Large (manager)")
    
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
    # #region debug log
    import json; f=open(r'c:\Users\DanielsGPU\Documents\GitHub\smolagents\.cursor\debug.log','a',encoding='utf-8'); f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"shared_agent_utils.py:672","message":"create_programming_agent entry","data":{"model_type":type(model).__name__,"memory_backend_type":type(memory_backend).__name__ if memory_backend else None},"timestamp":int(__import__('time').time()*1000)})+'\n'); f.close()
    # #endregion
    from smolagents import CodeAgent
    
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
        print("[OK] Publication tools and error/researcher tools added to programming agent")
    
    # #region debug log
    f=open(r'c:\Users\DanielsGPU\Documents\GitHub\smolagents\.cursor\debug.log','a',encoding='utf-8'); f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"shared_agent_utils.py:727","message":"Before CodeAgent init","data":{"tools_count":len(programming_tools),"model_type":type(model).__name__,"has_memory_backend":memory_backend is not None},"timestamp":int(__import__('time').time()*1000)})+'\n'); f.close()
    # #endregion
    try:
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
        # #region debug log
        f=open(r'c:\Users\DanielsGPU\Documents\GitHub\smolagents\.cursor\debug.log','a',encoding='utf-8'); f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"shared_agent_utils.py:790","message":"After CodeAgent init","data":{"agent_type":type(programming_agent).__name__},"timestamp":int(__import__('time').time()*1000)})+'\n'); f.close()
        # #endregion
    except Exception as e:
        # #region debug log
        f=open(r'c:\Users\DanielsGPU\Documents\GitHub\smolagents\.cursor\debug.log','a',encoding='utf-8'); f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"shared_agent_utils.py:793","message":"CodeAgent init exception","data":{"error_type":type(e).__name__,"error_msg":str(e)},"timestamp":int(__import__('time').time()*1000)})+'\n'); f.close()
        # #endregion
        raise
    print("[OK] Programming agent created successfully")
    return programming_agent

