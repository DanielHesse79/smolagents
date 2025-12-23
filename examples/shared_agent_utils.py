"""
Shared utility functions for agent setup and initialization.

This module provides common functions used by gradio_ui.py
for health checks, service initialization, and agent creation.
"""

import os
import time
import json
import subprocess
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
        PDFFileWriterTool,
        ErrorLoggingTool,
        ResearcherUpsertTool,
        ResearcherPublicationLinkTool,
        AtomicFactStorageTool,
        RelationshipGraphTool,
    )
    PUBLICATION_TOOLS_AVAILABLE = True
except ImportError:
    try:
        from publication_tools import (
            PubMedSearchTool,
            QdrantUpsertTool,
            SQLUpsertTool,
            MarkdownFileWriterTool,
            PDFFileWriterTool,
            ErrorLoggingTool,
            ResearcherUpsertTool,
            ResearcherPublicationLinkTool,
            AtomicFactStorageTool,
            RelationshipGraphTool,
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
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    sqlite_db_path: str = field(default_factory=lambda: os.getenv("SQLITE_DB_PATH", "./data/publications.db"))
    max_retries: int = 3
    retry_delay: float = 2.0
    required_ollama_models: list[str] = field(default_factory=lambda: (
        [m.strip() for m in os.getenv("OLLAMA_REQUIRED_MODELS", "").split(",") if m.strip()]
        # Empty by default - system will auto-select from available models
    ))
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

def detect_gpu() -> tuple[bool, Optional[str]]:
    """
    Detect GPU availability using multiple methods.
    
    Returns:
        tuple: (gpu_available: bool, gpu_info: str | None)
    """
    gpu_info = None
    gpu_available = False
    
    # Method 1: Check PyTorch/CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_info = f"{gpu_name} (CUDA {torch.version.cuda}, {gpu_count} device(s))"
    except ImportError:
        pass
    except Exception:
        pass
    
    # Method 2: Check nvidia-smi (works even without PyTorch)
    if not gpu_available:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_available = True
                gpu_lines = result.stdout.strip().split('\n')
                gpu_names = [line.split(',')[0].strip() for line in gpu_lines]
                driver_version = gpu_lines[0].split(',')[1].strip() if len(gpu_lines[0].split(',')) > 1 else "Unknown"
                gpu_info = f"{', '.join(gpu_names)} (Driver: {driver_version})"
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
    
    # Method 3: Check Ollama's GPU info via API (if available)
    if not gpu_available:
        try:
            import requests
            # Ollama might expose GPU info in some endpoints
            # This is a fallback - Ollama usually uses GPU if available
            pass
        except Exception:
            pass
    
    return gpu_available, gpu_info


def check_ollama_health(
    base_url: str = "http://localhost:11434",
    max_retries: int = 3,
    retry_delay: float = 2.0,
    timeout: float = 5.0,
    required_models: Optional[list[str]] = None
) -> dict:
    """
    Check Ollama availability with retry logic.
    
    Args:
        base_url: Ollama base URL
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        timeout: Request timeout in seconds
        required_models: Optional list of model names to check for (if None, no specific models required)
    
    Returns:
        dict: {
            "available": bool,
            "url": str,
            "version": str | None,
            "models": list[str],
            "required_models": dict[str, bool],  # Only populated if required_models provided
            "gpu_available": bool,
            "gpu_info": str | None,
            "recommended_num_ctx": int,
            "error": str | None
        }
    """
    # If required_models not provided, check environment variable (but don't require them)
    if required_models is None:
        env_models = os.getenv("OLLAMA_REQUIRED_MODELS", "")
        if env_models:
            required_models = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            required_models = []  # No specific models required - auto-select from available
    
    # Detect GPU
    gpu_available, gpu_info = detect_gpu()
    
    # Calculate recommended context length based on GPU
    if gpu_available:
        # With GPU, can use larger context
        recommended_num_ctx = 16384
    else:
        # CPU mode, use smaller context
        recommended_num_ctx = 8192
    
    result = {
        "available": False,
        "url": base_url,
        "version": None,
        "models": [],
        "required_models": {model: False for model in required_models} if required_models else {},
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "recommended_num_ctx": recommended_num_ctx,
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
        
        # Check required models (if any specified)
        if required_models:
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
# Docker/Qdrant Startup Functions
# ============================================================================

def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def check_qdrant_container_running(container_name: str = "qdrant") -> bool:
    """Check if Qdrant container is already running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return container_name in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def start_qdrant_container(
    container_name: str = "qdrant",
    port: int = 6333,
    port_grpc: int = 6334,
    image: str = "qdrant/qdrant"
) -> tuple[bool, str]:
    """
    Start Qdrant Docker container if not already running.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    # Check if Docker is available
    if not check_docker_available():
        return False, "Docker is not available. Please install Docker Desktop and ensure it's running."
    
    # Check if container is already running
    if check_qdrant_container_running(container_name):
        return True, f"Qdrant container '{container_name}' is already running."
    
    # Check if container exists but is stopped
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if container_name in result.stdout:
            # Container exists but is stopped, start it
            print(f"[STARTUP] Starting existing Qdrant container '{container_name}'...")
            subprocess.run(
                ["docker", "start", container_name],
                capture_output=True,
                timeout=30
            )
            # Wait a moment for container to start
            time.sleep(2)
            return True, f"Started existing Qdrant container '{container_name}'."
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        pass
    
    # Container doesn't exist, create and start it
    try:
        print(f"[STARTUP] Creating and starting Qdrant container '{container_name}'...")
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "-p", f"{port}:6333",
                "-p", f"{port_grpc}:6334",
                "--name", container_name,
                image
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            # Wait a moment for container to start
            time.sleep(3)
            return True, f"Successfully started Qdrant container '{container_name}'."
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return False, f"Failed to start Qdrant container: {error_msg}"
    except subprocess.TimeoutExpired:
        return False, "Timeout while starting Qdrant container."
    except FileNotFoundError:
        return False, "Docker command not found. Please install Docker Desktop."
    except Exception as e:
        return False, f"Error starting Qdrant container: {str(e)}"


# ============================================================================
# Startup Checks Orchestration
# ============================================================================

def run_startup_checks(config: StartupConfig) -> StartupResult:
        """
        Run all startup checks with retry logic.
        
        Returns:
            StartupResult with status of all services
        """
        import sys
        warnings = []
        errors = []
        
        # Check Ollama
        print("[STARTUP] Checking Ollama...")
        sys.stdout.flush()
        ollama_status = check_ollama_health(
            base_url=config.ollama_base_url,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            required_models=config.required_ollama_models
        )
        
        if not ollama_status["available"]:
            warnings.append(f"Ollama not available: {ollama_status.get('error', 'Unknown error')}")
        else:
            # Only warn about missing models if specific models were required
            if ollama_status.get("required_models"):
                missing_models = [m for m, found in ollama_status["required_models"].items() if not found]
                if missing_models:
                    warnings.append(f"Missing Ollama models: {', '.join(missing_models)}")
            # If no models available at all, warn
            elif not ollama_status.get("models"):
                warnings.append("No Ollama models found. Please install at least one model using `ollama pull <model_name>`")
        
        # Try to start Qdrant if not available
        print("[STARTUP] Checking Qdrant...")
        sys.stdout.flush()
        
        # First check if Qdrant is already running
        qdrant_status = check_qdrant_health(
            url=config.qdrant_url,
            port=config.qdrant_port,
            max_retries=1,  # Quick check first
            retry_delay=1.0,
            required_collections=config.required_qdrant_collections
        )
        
        # If Qdrant is not available, try to start it
        if not qdrant_status["available"]:
            print("[STARTUP] Qdrant not available, attempting to start Qdrant container...")
            sys.stdout.flush()
            success, message = start_qdrant_container(
                container_name="qdrant",
                port=config.qdrant_port,
                port_grpc=6334,
                image="qdrant/qdrant"
            )
            
            if success:
                print(f"[STARTUP] {message}")
                sys.stdout.flush()
                # Wait a bit more for Qdrant to fully initialize
                time.sleep(2)
                # Re-check Qdrant health
                qdrant_status = check_qdrant_health(
                    url=config.qdrant_url,
                    port=config.qdrant_port,
                    max_retries=config.max_retries,
                    retry_delay=config.retry_delay,
                    required_collections=config.required_qdrant_collections
                )
            else:
                print(f"[STARTUP] Could not start Qdrant: {message}")
                sys.stdout.flush()
        
        if not qdrant_status["available"]:
            warnings.append(f"Qdrant not available: {qdrant_status.get('error', 'Unknown error')}")
        else:
            missing_collections = [c for c, found in qdrant_status["required_collections"].items() if not found]
            if missing_collections:
                warnings.append(f"Missing Qdrant collections (will be created): {', '.join(missing_collections)}")
        
        # Check SQLite
        print("[STARTUP] Checking SQLite...")
        sys.stdout.flush()
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
        
        # 6. Atomic facts table (OSINT evidence tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS atomic_facts (
                fact_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT,
                qualifiers TEXT,
                evidence_url TEXT NOT NULL,
                evidence_snippet TEXT NOT NULL,
                page_locator TEXT,
                published_date TEXT,
                retrieved_at TEXT NOT NULL,
                confidence TEXT,
                target_company TEXT,
                mode TEXT
            )
        """)
        
        # 7. Relationship edges table (OSINT relationship graph)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationship_edges (
                edge_id TEXT PRIMARY KEY,
                from_entity TEXT NOT NULL,
                to_entity TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                strength TEXT,
                evidence_urls TEXT,
                evidence_snippets TEXT,
                target_company TEXT,
                mode TEXT,
                retrieved_at TEXT NOT NULL
            )
        """)
        
        # 8. Evidence sources table (deduplication)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evidence_sources (
                source_url TEXT PRIMARY KEY,
                source_type TEXT,
                published_date TEXT,
                first_retrieved_at TEXT,
                last_retrieved_at TEXT,
                title TEXT,
                domain TEXT
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

def get_model_config_path() -> str:
    """Get the path to the model configuration file."""
    config_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "model_config.json")


def load_model_preferences() -> dict:
    """Load saved model preferences from config file."""
    config_path = get_model_config_path()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load model preferences: {e}")
    return {}


def save_model_preferences(programming_model: str, manager_model: str):
    """Save model preferences to config file."""
    config_path = get_model_config_path()
    try:
        preferences = {
            "programming_model": programming_model,
            "manager_model": manager_model,
            "last_updated": time.time()
        }
        with open(config_path, 'w') as f:
            json.dump(preferences, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save model preferences: {e}")


def select_best_programming_model(available_models: list[str]) -> Optional[str]:
    """Select the best programming model from available models.
    
    Priority:
    1. Models with 'code' or 'coder' in name
    2. DeepSeek models (especially R1)
    3. CodeLlama models
    4. WizardCoder models
    5. StarCoder models
    6. Other models with 'code' related keywords
    7. Largest model (by parameter count in name)
    8. First available model
    """
    if not available_models:
        return None
    
    model_lower = [m.lower() for m in available_models]
    
    # Priority 1: Explicit code models
    for i, model in enumerate(model_lower):
        if any(keyword in model for keyword in ['code', 'coder', 'coding']):
            return available_models[i]
    
    # Priority 2: DeepSeek (especially R1)
    deepseek_r1 = None
    deepseek_any = None
    for i, model in enumerate(model_lower):
        if 'deepseek' in model and 'r1' in model:
            deepseek_r1 = available_models[i]
        elif 'deepseek' in model and not deepseek_any:
            deepseek_any = available_models[i]
    
    if deepseek_r1:
        return deepseek_r1
    if deepseek_any:
        return deepseek_any
    
    # Priority 3: CodeLlama
    for i, model in enumerate(model_lower):
        if 'codellama' in model or 'code-llama' in model:
            return available_models[i]
    
    # Priority 4: WizardCoder
    for i, model in enumerate(model_lower):
        if 'wizardcoder' in model or 'wizard-coder' in model:
            return available_models[i]
    
    # Priority 5: StarCoder
    for i, model in enumerate(model_lower):
        if 'starcoder' in model or 'star-coder' in model:
            return available_models[i]
    
    # Priority 6: Other code-related
    for i, model in enumerate(model_lower):
        if any(keyword in model for keyword in ['python', 'programming', 'dev']):
            return available_models[i]
    
    # Priority 7: Largest model (by parameter count in name)
    # Try to extract parameter count and pick largest
    def extract_params(model_name: str) -> int:
        import re
        # Look for patterns like "7b", "13b", "34b", "70b", etc.
        matches = re.findall(r'(\d+)[bm]', model_name.lower())
        if matches:
            return max(int(m) for m in matches)
        return 0
    
    sorted_models = sorted(available_models, key=extract_params, reverse=True)
    return sorted_models[0]


def select_best_manager_model(available_models: list[str], exclude_model: Optional[str] = None) -> Optional[str]:
    """Select the best manager/general-purpose model from available models.
    
    Priority:
    1. Mistral models
    2. Llama models (especially 3.x)
    3. Qwen models
    4. Other general-purpose models
    5. Largest model (by parameter count)
    6. First available (excluding programming model)
    """
    if not available_models:
        return None
    
    # Filter out excluded model
    candidates = [m for m in available_models if m != exclude_model]
    if not candidates:
        candidates = available_models  # Fallback if all excluded
    
    model_lower = [m.lower() for m in candidates]
    
    # Priority 1: Mistral
    for i, model in enumerate(model_lower):
        if 'mistral' in model:
            return candidates[i]
    
    # Priority 2: Llama (especially 3.x)
    llama3 = None
    llama_any = None
    for i, model in enumerate(model_lower):
        if 'llama' in model:
            if '3' in model or 'llama-3' in model:
                llama3 = candidates[i]
            elif not llama_any:
                llama_any = candidates[i]
    
    if llama3:
        return llama3
    if llama_any:
        return llama_any
    
    # Priority 3: Qwen
    for i, model in enumerate(model_lower):
        if 'qwen' in model:
            return candidates[i]
    
    # Priority 4: Other general-purpose models
    for i, model in enumerate(model_lower):
        if any(keyword in model for keyword in ['chat', 'instruct', 'general']):
            return candidates[i]
    
    # Priority 5: Largest model
    def extract_params(model_name: str) -> int:
        import re
        matches = re.findall(r'(\d+)[bm]', model_name.lower())
        if matches:
            return max(int(m) for m in matches)
        return 0
    
    sorted_models = sorted(candidates, key=extract_params, reverse=True)
    return sorted_models[0]


def setup_ollama_models(
    ollama_status: dict, 
    config: Optional[StartupConfig] = None,
    programming_model_name: Optional[str] = None,
    manager_model_name: Optional[str] = None
):
    """Setup Ollama models based on health check status.
    
    Automatically selects the most suitable models from available models if not specified.
    
    Args:
        ollama_status: Status dict from health check
        config: Optional startup configuration
        programming_model_name: Optional specific programming model name (if None, will auto-select)
        manager_model_name: Optional specific manager model name (if None, will auto-select)
    """
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
    
    # Get available models
    available_models = ollama_status.get("models", [])
    
    if not available_models:
        raise ValueError("No Ollama models available. Please install at least one model.")
    
    # Auto-select programming model if not provided
    if not programming_model_name:
        preferences = load_model_preferences()
        programming_model_name = preferences.get("programming_model")
        
        # Validate saved preference is still available
        if programming_model_name not in available_models:
            programming_model_name = None
        
        # Auto-select best programming model
        if not programming_model_name:
            programming_model_name = select_best_programming_model(available_models)
    
    # Auto-select manager model if not provided
    if not manager_model_name:
        preferences = load_model_preferences()
        manager_model_name = preferences.get("manager_model")
        
        # Validate saved preference is still available
        if manager_model_name not in available_models:
            manager_model_name = None
        
        # Auto-select best manager model (different from programming)
        if not manager_model_name:
            manager_model_name = select_best_manager_model(available_models, exclude_model=programming_model_name)
        
        # Fallback: if only one model, use it for both
        if not manager_model_name:
            manager_model_name = programming_model_name
    
    # Validate models exist
    if programming_model_name not in available_models:
        raise ValueError(f"Programming model '{programming_model_name}' not found in available models: {available_models}")
    if manager_model_name not in available_models:
        raise ValueError(f"Manager model '{manager_model_name}' not found in available models: {available_models}")
    
    # Save preferences for next time
    save_model_preferences(programming_model_name, manager_model_name)
    
    from smolagents import LiteLLMModel
    
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
    except Exception as e:
        raise
    
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
    except Exception as e:
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

def create_unicode_safe_logger(verbosity_level=1):
    """Create an AgentLogger with Unicode-safe console for Windows compatibility.
    
    This logger uses a console that writes to a custom file object on Windows
    that Rich won't detect as a terminal, avoiding Unicode encoding errors.
    Since we're streaming to Gradio, we don't need Rich's console output anyway.
    """
    import sys
    import io
    from smolagents.monitoring import AgentLogger, LogLevel
    from rich.console import Console
    
    # Create a console that handles Unicode properly on Windows
    if sys.platform == "win32":
        # Use os.devnull opened with UTF-8 encoding - this is a real file, not a terminal
        # Rich won't try to use Windows terminal rendering on a file
        null_file = open(os.devnull, 'w', encoding='utf-8', errors='replace')
        
        # Create console that won't use Windows terminal rendering
        # Environment variables are already set globally in gradio_ui.py
        console = Console(
            file=null_file,
            highlight=False,
            force_terminal=False,  # Disable terminal-specific features
            legacy_windows=False,  # Disable legacy Windows rendering
            width=None,  # Auto-detect width
            _environ={},  # Don't use environment variables for terminal detection
        )
        
        # Monkey-patch Rich's Windows rendering to prevent Unicode errors
        # Patch the _write_buffer method to catch and ignore Unicode errors
        import types
        original_write_buffer = console._write_buffer
        
        def safe_write_buffer(self):
            try:
                # Try to write, but catch Unicode errors
                original_write_buffer()
            except (UnicodeEncodeError, AttributeError, OSError):
                # If Unicode error occurs, just skip the write
                # We don't need the output anyway since we're streaming to Gradio
                # Clear the buffer to prevent retries
                if hasattr(self, '_buffer'):
                    self._buffer = []
        
        console._write_buffer = types.MethodType(safe_write_buffer, console)
        
        # Note: Rich's legacy_windows_render is already patched globally in gradio_ui.py
        # This ensures it's patched before any Console objects are created
    else:
        console = Console(highlight=False)
    
    return AgentLogger(level=verbosity_level, console=console)

def create_programming_agent(
    model,
    memory_backend,
    db_path: Optional[str] = None,
    qdrant_collection_name: str = "microsampling_publications"
):
    """Create the programming agent with appropriate tools."""
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
            PDFFileWriterTool(output_dir="./data"),
            ErrorLoggingTool(db_path=db_path_final),
            ResearcherUpsertTool(db_path=db_path_final),
            ResearcherPublicationLinkTool(db_path=db_path_final),
            AtomicFactStorageTool(db_path=db_path_final),
            RelationshipGraphTool(db_path=db_path_final),
        ]
        print("[OK] Publication tools, PDF writer, OSINT evidence tools, and error/researcher tools added to programming agent")
    
    try:
        # Create Unicode-safe logger for Windows compatibility
        logger = create_unicode_safe_logger(verbosity_level=1)
        
        programming_agent = CodeAgent(
            tools=programming_tools,
            model=model,
            name="programmer",
            description="A specialized programming agent that writes, debugs, and executes code. Use this agent when you need to write Python code, solve programming problems, create scripts, or perform computational tasks.",
            verbosity_level=1,
            logger=logger,  # Use custom logger with Unicode-safe console
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
    except Exception as e:
        raise
    print("[OK] Programming agent created successfully")
    return programming_agent

