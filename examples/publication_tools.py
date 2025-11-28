"""
Custom tools for publication mining tasks.

This module provides specialized tools for:
- Searching PubMed/Europe PMC
- Storing publications in Qdrant vector database
- Storing publications in SQL database
- Writing markdown files
"""

import os
import json
from typing import Optional, Dict, Any

from smolagents import Tool

# Try to import Qdrant dependencies
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    PointStruct = None
    SentenceTransformer = None


class PubMedSearchTool(Tool):
    """Tool for searching PubMed/Europe PMC for academic publications."""
    
    name = "pubmed_search"
    description = (
        "Searches PubMed/Europe PMC for academic publications. "
        "Returns a list of publications with title, authors, DOI, PMID, abstract, and URL. "
        "Use this tool to find scientific publications on specific topics. "
        "The tool searches both PubMed and Europe PMC databases."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query (e.g., 'volumetric absorptive microsampling', 'VAMS AND pharmacokinetics')"
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default: 20, max: 100)",
            "nullable": True
        }
    }
    output_type = "string"
    
    def forward(self, query: str, max_results: int = 20) -> str:
        """Search PubMed/Europe PMC."""
        try:
            import requests
            
            # Limit max_results to prevent excessive API calls
            max_results = min(max_results, 100)
            
            # Use Europe PMC REST API (free, no API key required)
            # Documentation: https://europepmc.org/RestfulWebService
            base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            
            params = {
                "query": query,
                "format": "json",
                "pageSize": max_results,
                "resultType": "core"
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "resultList" in data and "result" in data["resultList"]:
                for pub in data["resultList"]["result"]:
                    pub_info = {
                        "title": pub.get("title", "N/A"),
                        "authors": pub.get("authorString", "N/A"),
                        "doi": pub.get("doi", ""),
                        "pmid": pub.get("pmid", ""),
                        "pmcid": pub.get("pmcid", ""),
                        "year": pub.get("pubYear", ""),
                        "journal": pub.get("journalTitle", ""),
                        "abstract": pub.get("abstractText", ""),
                        "url": f"https://europepmc.org/article/MED/{pub.get('pmid', '')}" if pub.get("pmid") else ""
                    }
                    results.append(pub_info)
            
            if not results:
                return f"No publications found for query: {query}"
            
            # Format results as readable string
            formatted_results = []
            for i, pub in enumerate(results, 1):
                formatted_results.append(
                    f"\n{i}. {pub['title']}\n"
                    f"   Authors: {pub['authors']}\n"
                    f"   Year: {pub['year']}\n"
                    f"   Journal: {pub['journal']}\n"
                    f"   DOI: {pub['doi']}\n"
                    f"   PMID: {pub['pmid']}\n"
                    f"   URL: {pub['url']}\n"
                    f"   Abstract: {pub['abstract'][:500]}..." if len(pub['abstract']) > 500 else f"   Abstract: {pub['abstract']}"
                )
            
            return f"Found {len(results)} publications:\n" + "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching PubMed: {str(e)}"


class QdrantUpsertTool(Tool):
    """Tool for storing publications in Qdrant vector database."""
    
    name = "qdrant_upsert_publication"
    description = (
        "Stores a publication in Qdrant vector database with embeddings. "
        "The publication text will be embedded and stored for semantic search. "
        "Use this tool to persist publications for later retrieval."
    )
    inputs = {
        "collection": {
            "type": "string",
            "description": "Collection name (default: 'microsampling_publications')",
            "nullable": True
        },
        "point_id": {
            "type": "string",
            "description": "Unique point ID (e.g., DOI, PMID, or hash)"
        },
        "vector_text": {
            "type": "string",
            "description": "Text to embed (typically title + abstract or full text)"
        },
        "payload": {
            "type": "object",
            "description": "Metadata payload as dictionary (title, authors, doi, pmid, year, etc.)"
        }
    }
    output_type = "string"
    
    def __init__(self, collection_name: str = "microsampling_publications", 
                 url: str = "localhost", port: int = 6333,
                 embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant dependencies not available. Install with: pip install 'smolagents[qdrant]'"
            )
        self.collection_name = collection_name
        self.url = url
        self.port = port
        self.embedding_model_id = embedding_model_id
        self._client = None
        self._embedder = None
    
    def _get_client(self):
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(url=self.url, port=self.port)
            # Ensure collection exists
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.collection_name not in collection_names:
                # Need embedding dimension to create collection
                if self._embedder is None:
                    self._embedder = SentenceTransformer(self.embedding_model_id)
                embedding_dim = self._embedder.get_sentence_embedding_dimension()
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
        return self._client
    
    def _get_embedder(self):
        """Lazy initialization of embedding model."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_id)
        return self._embedder
    
    def forward(self, collection: str | None, point_id: str, vector_text: str, payload: Dict[str, Any]) -> str:
        """Upsert a publication to Qdrant."""
        try:
            # Use provided collection or default
            collection_name = collection or self.collection_name
            
            client = self._get_client()
            embedder = self._get_embedder()
            
            # Generate embedding
            vector = embedder.encode(vector_text).tolist()
            
            # Ensure collection exists (in case different collection name provided)
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]
            if collection_name not in collection_names:
                embedding_dim = len(vector)
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
            
            # Upsert point
            # Convert point_id to integer hash if it's a string (Qdrant requires integer IDs)
            if isinstance(point_id, str):
                # Generate consistent integer ID from string
                point_id_int = abs(hash(point_id)) % (2**63)
            else:
                point_id_int = point_id
            
            point = PointStruct(
                id=point_id_int,
                vector=vector,
                payload=payload
            )
            
            client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            return f"Successfully stored publication '{point_id}' in Qdrant collection '{collection_name}'"
            
        except Exception as e:
            return f"Error storing publication in Qdrant: {str(e)}"


class SQLUpsertTool(Tool):
    """Tool for storing publications in SQL database."""
    
    name = "sql_upsert_publication"
    description = (
        "Stores or updates a publication in SQL database. "
        "Uses INSERT OR REPLACE to handle duplicates based on unique_key. "
        "Use this tool to persist publication metadata in a relational database."
    )
    inputs = {
        "table": {
            "type": "string",
            "description": "Table name (default: 'publications')",
            "nullable": True
        },
        "record": {
            "type": "object",
            "description": "Publication record as dictionary with fields: unique_key, title, year, doi, pmid, url, etc."
        }
    }
    output_type = "string"
    
    def __init__(self, db_path: str = "./data/publications.db"):
        super().__init__()
        self.db_path = db_path
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        # Initialize database schema
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create table if it doesn't exist."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
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
    
    def forward(self, table: str | None, record: Dict[str, Any]) -> str:
        """Upsert a publication record."""
        try:
            import sqlite3
            from datetime import datetime
            
            table_name = table or "publications"
            
            # Ensure schema exists
            self._ensure_schema()
            
            # Add timestamp if not present
            if "retrieved_at" not in record:
                record["retrieved_at"] = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get column names from table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Filter record to only include valid columns
            filtered_record = {k: v for k, v in record.items() if k in columns}
            
            # Build INSERT OR REPLACE query
            placeholders = ", ".join(["?"] * len(filtered_record))
            columns_str = ", ".join(filtered_record.keys())
            values = list(filtered_record.values())
            
            query = f"""
                INSERT OR REPLACE INTO {table_name} ({columns_str})
                VALUES ({placeholders})
            """
            
            cursor.execute(query, values)
            conn.commit()
            conn.close()
            
            unique_key = record.get("unique_key", "unknown")
            return f"Successfully stored publication '{unique_key}' in SQL table '{table_name}'"
            
        except Exception as e:
            return f"Error storing publication in SQL: {str(e)}"


class MarkdownFileWriterTool(Tool):
    """Tool for writing markdown files."""
    
    name = "write_markdown_file"
    description = (
        "Writes content to a markdown file. "
        "Use this tool to save structured output, summaries, or publication lists as markdown files."
    )
    inputs = {
        "filename": {
            "type": "string",
            "description": "Output filename (e.g., 'microsampling_publications.md')"
        },
        "content": {
            "type": "string",
            "description": "Markdown content to write to the file"
        }
    }
    output_type = "string"
    
    def __init__(self, output_dir: str = "./data"):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def forward(self, filename: str, content: str) -> str:
        """Write markdown content to file."""
        try:
            # Ensure filename ends with .md
            if not filename.endswith(".md"):
                filename += ".md"
            
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            return f"Successfully wrote markdown file: {filepath}"
            
        except Exception as e:
            return f"Error writing markdown file: {str(e)}"


class ErrorLoggingTool(Tool):
    """Tool for automatically logging agent errors to database."""
    
    name = "log_agent_error"
    description = (
        "Logs an agent error to the database for learning purposes. "
        "Use this tool when an error occurs to track patterns and improve future performance."
    )
    inputs = {
        "error_type": {
            "type": "string",
            "description": "Type of error (e.g., 'SyntaxError', 'ValueError', 'TypeError')"
        },
        "error_message": {
            "type": "string",
            "description": "Full error message"
        },
        "code_snippet": {
            "type": "string",
            "description": "The code that caused the error"
        },
        "step_number": {
            "type": "integer",
            "description": "Step number where error occurred",
            "nullable": True
        },
        "task_context": {
            "type": "string",
            "description": "What the agent was trying to do",
            "nullable": True
        },
        "agent_name": {
            "type": "string",
            "description": "Name of agent ('programmer' or 'manager')",
            "nullable": True
        },
        "session_id": {
            "type": "string",
            "description": "Session identifier",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, db_path: str = "./data/publications.db"):
        super().__init__()
        self.db_path = db_path
    
    def forward(
        self,
        error_type: str,
        error_message: str,
        code_snippet: str,
        step_number: int | None = None,
        task_context: str | None = None,
        agent_name: str | None = None,
        session_id: str | None = None
    ) -> str:
        """Log an error to the database."""
        try:
            import sqlite3
            from datetime import datetime
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO agent_errors (
                    error_type, error_message, code_snippet, step_number,
                    task_context, timestamp, agent_name, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                error_type,
                error_message,
                code_snippet,
                step_number,
                task_context,
                datetime.now().isoformat(),
                agent_name,
                session_id
            ))
            
            conn.commit()
            error_id = cursor.lastrowid
            conn.close()
            
            return f"Error logged with ID {error_id}"
        except Exception as e:
            return f"Error logging failed: {str(e)}"


class ResearcherUpsertTool(Tool):
    """Tool for storing researcher information."""
    
    name = "upsert_researcher"
    description = (
        "Stores or updates researcher information in the database. "
        "Use this to track relevant researchers, their fields, and contact information."
    )
    inputs = {
        "name": {
            "type": "string",
            "description": "Researcher name"
        },
        "email": {
            "type": "string",
            "description": "Email address",
            "nullable": True
        },
        "affiliation": {
            "type": "string",
            "description": "Institution or organization",
            "nullable": True
        },
        "research_fields": {
            "type": "array",
            "description": "List of research fields (e.g., ['microsampling', 'pharmacokinetics'])",
            "items": {"type": "string"},
            "nullable": True
        },
        "relevance_reason": {
            "type": "string",
            "description": "Why this researcher is relevant to the project",
            "nullable": True
        },
        "contact_info": {
            "type": "object",
            "description": "Additional contact information (phone, address, ORCID, etc.)",
            "nullable": True
        },
        "notes": {
            "type": "string",
            "description": "Additional notes about the researcher",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, db_path: str = "./data/publications.db"):
        super().__init__()
        self.db_path = db_path
    
    def forward(
        self,
        name: str,
        email: str | None = None,
        affiliation: str | None = None,
        research_fields: list[str] | None = None,
        relevance_reason: str | None = None,
        contact_info: Dict[str, Any] | None = None,
        notes: str | None = None
    ) -> str:
        """Upsert researcher information."""
        try:
            import sqlite3
            from datetime import datetime
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if researcher exists
            if email:
                cursor.execute("SELECT researcher_id, first_encountered FROM researchers WHERE name = ? AND email = ?", (name, email))
            else:
                cursor.execute("SELECT researcher_id, first_encountered FROM researchers WHERE name = ? AND email IS NULL", (name,))
            
            existing = cursor.fetchone()
            
            now = datetime.now().isoformat()
            
            if existing:
                researcher_id, first_encountered = existing
                # Update existing researcher
                cursor.execute("""
                    UPDATE researchers SET
                        affiliation = COALESCE(?, affiliation),
                        research_fields = COALESCE(?, research_fields),
                        relevance_reason = COALESCE(?, relevance_reason),
                        contact_info = COALESCE(?, contact_info),
                        notes = COALESCE(?, notes),
                        last_updated = ?
                    WHERE researcher_id = ?
                """, (
                    affiliation,
                    json.dumps(research_fields) if research_fields else None,
                    relevance_reason,
                    json.dumps(contact_info) if contact_info else None,
                    notes,
                    now,
                    researcher_id
                ))
                action = "updated"
            else:
                # Insert new researcher
                cursor.execute("""
                    INSERT INTO researchers (
                        name, email, affiliation, research_fields,
                        relevance_reason, first_encountered, contact_info, notes, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    name,
                    email,
                    affiliation,
                    json.dumps(research_fields) if research_fields else None,
                    relevance_reason,
                    now,
                    json.dumps(contact_info) if contact_info else None,
                    notes,
                    now
                ))
                researcher_id = cursor.lastrowid
                action = "created"
            
            conn.commit()
            conn.close()
            
            return f"Researcher {action} with ID {researcher_id}"
        except Exception as e:
            return f"Error upserting researcher: {str(e)}"


class ResearcherPublicationLinkTool(Tool):
    """Tool for linking researchers to publications."""
    
    name = "link_researcher_publication"
    description = (
        "Links a researcher to a publication, recording their role and position. "
        "Use this to track which researchers authored which publications."
    )
    inputs = {
        "researcher_id": {
            "type": "integer",
            "description": "ID of the researcher"
        },
        "publication_key": {
            "type": "string",
            "description": "Unique key of the publication"
        },
        "role": {
            "type": "string",
            "description": "Role in publication ('author', 'corresponding_author', 'first_author', etc.)",
            "nullable": True
        },
        "author_position": {
            "type": "integer",
            "description": "Position in author list (1 = first author)",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, db_path: str = "./data/publications.db"):
        super().__init__()
        self.db_path = db_path
    
    def forward(
        self,
        researcher_id: int,
        publication_key: str,
        role: str | None = None,
        author_position: int | None = None
    ) -> str:
        """Link researcher to publication."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO researcher_publications (
                    researcher_id, publication_key, role, author_position
                ) VALUES (?, ?, ?, ?)
            """, (researcher_id, publication_key, role, author_position))
            
            # Update publications_count for researcher
            cursor.execute("""
                UPDATE researchers SET
                    publications_count = (
                        SELECT COUNT(*) FROM researcher_publications
                        WHERE researcher_id = ?
                    )
                WHERE researcher_id = ?
            """, (researcher_id, researcher_id))
            
            conn.commit()
            conn.close()
            
            return f"Linked researcher {researcher_id} to publication {publication_key}"
        except Exception as e:
            return f"Error linking researcher to publication: {str(e)}"

