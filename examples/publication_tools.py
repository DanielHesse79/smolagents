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

# Try to import Scrapy dependencies
try:
    from scrapy.http import HtmlResponse
    from scrapy.selector import Selector
    SCRAPY_AVAILABLE = True
except ImportError:
    SCRAPY_AVAILABLE = False
    HtmlResponse = None
    Selector = None

# Try to import Playwright dependencies
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    sync_playwright = None


class PubMedSearchTool(Tool):
    """Tool for searching PubMed/Europe PMC for academic publications."""
    
    name = "pubmed_search"
    description = (
        "Searches PubMed/Europe PMC for academic publications. "
        "Returns formatted text with publication details (title, authors, DOI, PMID, abstract, affiliations, URL). "
        "The output includes a [STRUCTURED_DATA] section with JSON that can be parsed programmatically. "
        "Each publication includes an 'affiliations' list that can be filtered by country/institution. "
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
            import json
            
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
                    # Extract affiliations from authorList if available
                    affiliations = []
                    if "authorList" in pub and "author" in pub["authorList"]:
                        for author in pub["authorList"]["author"]:
                            if "affiliation" in author:
                                affil_text = author["affiliation"]
                                if isinstance(affil_text, list):
                                    affiliations.extend([a.get("$", "") for a in affil_text if isinstance(a, dict) and "$" in a])
                                elif isinstance(affil_text, str):
                                    affiliations.append(affil_text)
                    
                    pub_info = {
                        "title": pub.get("title", "N/A"),
                        "authors": pub.get("authorString", "N/A"),
                        "doi": pub.get("doi", ""),
                        "pmid": pub.get("pmid", ""),
                        "pmcid": pub.get("pmcid", ""),
                        "year": pub.get("pubYear", ""),
                        "journal": pub.get("journalTitle", ""),
                        "abstract": pub.get("abstractText", ""),
                        "url": f"https://europepmc.org/article/MED/{pub.get('pmid', '')}" if pub.get("pmid") else "",
                        "affiliations": affiliations
                    }
                    results.append(pub_info)
            
            if not results:
                return f"No publications found for query: {query}"
            
            # Return JSON string that can be parsed by the agent
            # This allows the agent to programmatically filter by affiliations
            json_str = json.dumps(results, ensure_ascii=False, indent=2)
            
            # Also provide a human-readable summary
            formatted_results = []
            for i, pub in enumerate(results, 1):
                affil_str = "; ".join(pub['affiliations'][:3]) if pub['affiliations'] else "N/A"
                if len(pub['affiliations']) > 3:
                    affil_str += f" (+{len(pub['affiliations']) - 3} more)"
                formatted_results.append(
                    f"\n{i}. {pub['title']}\n"
                    f"   Authors: {pub['authors']}\n"
                    f"   Year: {pub['year']}\n"
                    f"   Journal: {pub['journal']}\n"
                    f"   Affiliations: {affil_str}\n"
                    f"   DOI: {pub['doi']}\n"
                    f"   PMID: {pub['pmid']}\n"
                    f"   URL: {pub['url']}\n"
                    f"   Abstract: {pub['abstract'][:500]}..." if len(pub['abstract']) > 500 else f"   Abstract: {pub['abstract']}"
                )
            
            # Return both formatted text and JSON for programmatic access
            return f"Found {len(results)} publications:\n" + "\n".join(formatted_results) + f"\n\n[STRUCTURED_DATA]\n{json_str}\n[/STRUCTURED_DATA]"
            
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
            # Don't raise - allow tool to be created but it will fail gracefully when used
            # This allows the agent to continue even if Qdrant is optional
            pass
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
        if not QDRANT_AVAILABLE:
            return "Qdrant is not available. Install with: pip install 'smolagents[qdrant]'. Publication will be stored in SQLite only."
        
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
        with sqlite3.connect(self.db_path) as conn:
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
            
            with sqlite3.connect(self.db_path) as conn:
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


class PDFFileWriterTool(Tool):
    """Tool for writing PDF files from text or markdown content."""
    
    name = "write_pdf_file"
    description = (
        "Writes content to a PDF file. "
        "Use this tool to save structured output, summaries, research reports, or publication lists as PDF files. "
        "The content can be plain text or markdown - it will be formatted appropriately in the PDF."
    )
    inputs = {
        "filename": {
            "type": "string",
            "description": "Output filename (e.g., 'telimmune_research_report.pdf')"
        },
        "content": {
            "type": "string",
            "description": "Text or markdown content to write to the PDF file"
        },
        "title": {
            "type": "string",
            "description": "Optional title for the PDF document",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, output_dir: str = "./data"):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Try to import reportlab
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.enums import TA_LEFT, TA_CENTER
            import re
            self.reportlab_available = True
            self.letter = letter
            self.A4 = A4
            self.getSampleStyleSheet = getSampleStyleSheet
            self.ParagraphStyle = ParagraphStyle
            self.inch = inch
            self.SimpleDocTemplate = SimpleDocTemplate
            self.Paragraph = Paragraph
            self.Spacer = Spacer
            self.PageBreak = PageBreak
            self.TA_LEFT = TA_LEFT
            self.TA_CENTER = TA_CENTER
            self.re = re
        except ImportError:
            self.reportlab_available = False
    
    def _markdown_to_paragraphs(self, content: str, styles):
        """Convert markdown content to reportlab Paragraph objects."""
        paragraphs = []
        lines = content.split('\n')
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append(self.Paragraph(' '.join(current_paragraph), styles['Normal']))
                    paragraphs.append(self.Spacer(1, 0.1 * self.inch))
                    current_paragraph = []
                continue
            
            # Handle headers
            if line.startswith('# '):
                if current_paragraph:
                    paragraphs.append(self.Paragraph(' '.join(current_paragraph), styles['Normal']))
                    current_paragraph = []
                paragraphs.append(self.Paragraph(line[2:], styles['Heading1']))
                paragraphs.append(self.Spacer(1, 0.2 * self.inch))
            elif line.startswith('## '):
                if current_paragraph:
                    paragraphs.append(self.Paragraph(' '.join(current_paragraph), styles['Normal']))
                    current_paragraph = []
                paragraphs.append(self.Paragraph(line[3:], styles['Heading2']))
                paragraphs.append(self.Spacer(1, 0.15 * self.inch))
            elif line.startswith('### '):
                if current_paragraph:
                    paragraphs.append(self.Paragraph(' '.join(current_paragraph), styles['Normal']))
                    current_paragraph = []
                paragraphs.append(self.Paragraph(line[4:], styles['Heading3']))
                paragraphs.append(self.Spacer(1, 0.1 * self.inch))
            elif line.startswith('**') and line.endswith('**'):
                # Bold text
                if current_paragraph:
                    paragraphs.append(self.Paragraph(' '.join(current_paragraph), styles['Normal']))
                    current_paragraph = []
                bold_text = line[2:-2]
                paragraphs.append(self.Paragraph(f"<b>{bold_text}</b>", styles['Normal']))
                paragraphs.append(self.Spacer(1, 0.05 * self.inch))
            elif line.startswith('- ') or line.startswith('* '):
                # Bullet point
                if current_paragraph:
                    paragraphs.append(self.Paragraph(' '.join(current_paragraph), styles['Normal']))
                    current_paragraph = []
                bullet_text = line[2:]
                paragraphs.append(self.Paragraph(f"• {bullet_text}", styles['Normal']))
                paragraphs.append(self.Spacer(1, 0.05 * self.inch))
            else:
                # Regular text
                current_paragraph.append(line)
        
        if current_paragraph:
            paragraphs.append(self.Paragraph(' '.join(current_paragraph), styles['Normal']))
        
        return paragraphs
    
    def forward(self, filename: str, content: str, title: Optional[str] = None) -> str:
        """Write content to PDF file."""
        try:
            # Ensure filename ends with .pdf
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            
            filepath = os.path.join(self.output_dir, filename)
            
            if self.reportlab_available:
                # Use reportlab to create PDF
                doc = self.SimpleDocTemplate(filepath, pagesize=self.A4)
                styles = self.getSampleStyleSheet()
                
                # Custom styles
                styles.add(self.ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=18,
                    textColor='#333333',
                    spaceAfter=30,
                    alignment=self.TA_CENTER
                ))
                
                # Build PDF content
                story = []
                
                # Add title if provided
                if title:
                    story.append(self.Paragraph(title, styles['CustomTitle']))
                    story.append(self.Spacer(1, 0.3 * self.inch))
                elif filename:
                    # Use filename as title
                    title_text = filename.replace('.pdf', '').replace('_', ' ').title()
                    story.append(self.Paragraph(title_text, styles['CustomTitle']))
                    story.append(self.Spacer(1, 0.3 * self.inch))
                
                # Convert content to paragraphs
                paragraphs = self._markdown_to_paragraphs(content, styles)
                story.extend(paragraphs)
                
                # Build PDF
                doc.build(story)
                
                return f"Successfully wrote PDF file: {filepath}"
            else:
                # Fallback: save as text file if reportlab is not available
                # User should install reportlab for proper PDF generation
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"PDF generation requires reportlab. Content saved as text file: {filepath}. Install reportlab with: pip install reportlab"
            
        except Exception as e:
            return f"Error writing PDF file: {str(e)}"


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
            
            with sqlite3.connect(self.db_path) as conn:
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
            
            with sqlite3.connect(self.db_path) as conn:
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
            
            with sqlite3.connect(self.db_path) as conn:
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
            
            return f"Linked researcher {researcher_id} to publication {publication_key}"
        except Exception as e:
            return f"Error linking researcher to publication: {str(e)}"


class ResearcherQueryTool(Tool):
    """Tool for querying and filtering researchers with various criteria."""
    
    name = "query_researchers"
    description = (
        "Söker och filtrerar forskare baserat på olika kriterier. "
        "Returnerar en formaterad lista med forskare och deras publikationer, år, ämnen och affiliations. "
        "Använd detta verktyg för att hitta forskare baserat på namn, affiliation, forskningsområden, publikationsår eller antal publikationer."
    )
    inputs = {
        "name": {
            "type": "string",
            "description": "Forskares namn (partial match stöds)",
            "nullable": True
        },
        "affiliation": {
            "type": "string",
            "description": "Affiliation/institution att filtrera på (partial match stöds)",
            "nullable": True
        },
        "research_field": {
            "type": "string",
            "description": "Forskningsområde/ämne att filtrera på (t.ex. 'microsampling', 'pharmacokinetics')",
            "nullable": True
        },
        "min_year": {
            "type": "integer",
            "description": "Minsta publikationsår att inkludera",
            "nullable": True
        },
        "max_year": {
            "type": "integer",
            "description": "Högsta publikationsår att inkludera",
            "nullable": True
        },
        "min_publications": {
            "type": "integer",
            "description": "Minsta antal publikationer som forskaren ska ha",
            "nullable": True
        },
        "limit": {
            "type": "integer",
            "description": "Maximalt antal resultat att returnera (default: 50, max: 200)",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, db_path: str = "./data/publications.db"):
        super().__init__()
        self.db_path = db_path
    
    def forward(
        self,
        name: str | None = None,
        affiliation: str | None = None,
        research_field: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        min_publications: int | None = None,
        limit: int | None = None
    ) -> str:
        """Query researchers with various filters."""
        try:
            import sqlite3
            
            limit = min(limit or 50, 200)  # Cap at 200
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.cursor()
                
                # Build WHERE clause dynamically
                where_clauses = []
                params = []
                
                if name:
                    where_clauses.append("r.name LIKE ?")
                    params.append(f"%{name}%")
                
                if affiliation:
                    where_clauses.append("r.affiliation LIKE ?")
                    params.append(f"%{affiliation}%")
                
                if research_field:
                    where_clauses.append("r.research_fields LIKE ?")
                    params.append(f"%{research_field}%")
                
                if min_publications is not None:
                    where_clauses.append("r.publications_count >= ?")
                    params.append(min_publications)
                
                where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
                
                # Main query to get researchers
                query = f"""
                    SELECT DISTINCT
                        r.researcher_id,
                        r.name,
                        r.email,
                        r.affiliation,
                        r.research_fields,
                        r.publications_count,
                        r.first_encountered,
                        r.last_updated
                    FROM researchers r
                    WHERE {where_sql}
                    ORDER BY r.publications_count DESC, r.name ASC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor.execute(query, params)
                researchers = cursor.fetchall()
                
                if not researchers:
                    return "Inga forskare hittades med de angivna kriterierna."
                
                # For each researcher, get their publications with year filtering
                results = []
                for researcher in researchers:
                    researcher_id = researcher["researcher_id"]
                    
                    # Get publications for this researcher
                    pub_query = """
                        SELECT 
                            p.title,
                            p.year,
                            p.journal,
                            p.doi,
                            p.pmid,
                            rp.role,
                            rp.author_position
                        FROM researcher_publications rp
                        JOIN publications p ON rp.publication_key = p.unique_key
                        WHERE rp.researcher_id = ?
                    """
                    pub_params = [researcher_id]
                    
                    # Add year filtering if specified
                    if min_year is not None or max_year is not None:
                        if min_year is not None and max_year is not None:
                            pub_query += " AND p.year >= ? AND p.year <= ?"
                            pub_params.extend([min_year, max_year])
                        elif min_year is not None:
                            pub_query += " AND p.year >= ?"
                            pub_params.append(min_year)
                        elif max_year is not None:
                            pub_query += " AND p.year <= ?"
                            pub_params.append(max_year)
                    
                    pub_query += " ORDER BY p.year DESC, rp.author_position ASC"
                    
                    cursor.execute(pub_query, pub_params)
                    publications = cursor.fetchall()
                    
                    # Filter: if year filter is specified and no publications match, skip this researcher
                    if (min_year is not None or max_year is not None) and not publications:
                        continue
                    
                    # Parse research_fields JSON
                    research_fields = []
                    if researcher["research_fields"]:
                        try:
                            research_fields = json.loads(researcher["research_fields"])
                        except:
                            pass
                    
                    # Format researcher info
                    researcher_info = {
                        "id": researcher_id,
                        "name": researcher["name"],
                        "email": researcher["email"] or "N/A",
                        "affiliation": researcher["affiliation"] or "N/A",
                        "research_fields": research_fields,
                        "publications_count": researcher["publications_count"] or 0,
                        "first_encountered": researcher["first_encountered"],
                        "last_updated": researcher["last_updated"],
                        "publications": []
                    }
                    
                    # Add publications
                    for pub in publications:
                        researcher_info["publications"].append({
                            "title": pub["title"] or "N/A",
                            "year": pub["year"],
                            "journal": pub["journal"] or "N/A",
                            "doi": pub["doi"] or "",
                            "pmid": pub["pmid"] or "",
                            "role": pub["role"] or "author",
                            "author_position": pub["author_position"]
                        })
                    
                    results.append(researcher_info)
            
            # Format output
            if not results:
                return "Inga forskare hittades med de angivna kriterierna (efter årfiltrering)."
            
            output_lines = [f"Hittade {len(results)} forskare:\n"]
            
            for i, res in enumerate(results, 1):
                output_lines.append(f"\n{i}. {res['name']} (ID: {res['id']})")
                output_lines.append(f"   Email: {res['email']}")
                output_lines.append(f"   Affiliation: {res['affiliation']}")
                if res['research_fields']:
                    fields_str = ", ".join(res['research_fields'])
                    output_lines.append(f"   Forskningsområden: {fields_str}")
                output_lines.append(f"   Antal publikationer: {res['publications_count']}")
                output_lines.append(f"   Först hittad: {res['first_encountered']}")
                
                if res['publications']:
                    output_lines.append(f"   Publikationer:")
                    for pub in res['publications']:
                        role_str = f" ({pub['role']})" if pub['role'] else ""
                        pos_str = f" [Position {pub['author_position']}]" if pub['author_position'] else ""
                        output_lines.append(f"     - ({pub['year']}) {pub['title']}{role_str}{pos_str}")
                        if pub['doi']:
                            output_lines.append(f"       DOI: {pub['doi']}")
                else:
                    output_lines.append(f"   Publikationer: Inga matchande publikationer med årfiltrering")
            
            return "\n".join(output_lines)
            
        except Exception as e:
            return f"Error querying researchers: {str(e)}"


class ScrapyVisionTool(Tool):
    """Advanced web scraping tool with Scrapy and automatic vision model support for complex pages."""
    
    name = "scrapy_extract"
    description = (
        "Advanced web scraper that extracts content from web pages using Scrapy's powerful selectors. "
        "Automatically uses vision model (qwen2.5vl:7b) when pages contain images, charts, or complex layouts. "
        "Supports multi-page crawling with configurable depth and link patterns. "
        "Use this tool for robust extraction from websites, especially when standard text extraction fails."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL to scrape (required for single page extraction)"
        },
        "css_selector": {
            "type": "string",
            "description": "CSS selector to extract specific content (e.g., 'article', '.content', '#main')",
            "nullable": True
        },
        "xpath_selector": {
            "type": "string",
            "description": "XPath selector as alternative to CSS selector",
            "nullable": True
        },
        "start_url": {
            "type": "string",
            "description": "Starting URL for multi-page crawling",
            "nullable": True
        },
        "max_pages": {
            "type": "integer",
            "description": "Maximum number of pages to crawl (default: 1, max: 10)",
            "nullable": True
        },
        "link_pattern": {
            "type": "string",
            "description": "URL pattern to follow when crawling (e.g., 'https://example.com/article/*')",
            "nullable": True
        },
        "use_vision": {
            "type": "boolean",
            "description": "Force use of vision model even if auto-detect doesn't trigger",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(
        self,
        vision_model_id: str = "ollama_chat/qwen2.5vl:7b",
        ollama_base_url: str = "http://localhost:11434",
        max_output_length: int = 40000,
        vision_threshold: int = 3
    ):
        super().__init__()
        if not SCRAPY_AVAILABLE:
            raise ImportError(
                "Scrapy is required for ScrapyVisionTool. Install with: pip install scrapy"
            )
        self.vision_model_id = vision_model_id
        self.ollama_base_url = ollama_base_url
        self.max_output_length = max_output_length
        self.vision_threshold = vision_threshold  # Number of images/charts to trigger vision
        self._vision_model = None
        self._playwright = None
    
    def _get_vision_model(self):
        """Lazy initialization of vision model."""
        if self._vision_model is None:
            try:
                from smolagents import LiteLLMModel
                self._vision_model = LiteLLMModel(
                    model_id=self.vision_model_id,
                    api_base=self.ollama_base_url,
                    api_key="ollama",
                    timeout=300,
                )
            except Exception as e:
                print(f"[WARNING] Could not initialize vision model: {e}")
                return None
        return self._vision_model
    
    def _needs_vision(self, response: HtmlResponse) -> bool:
        """Auto-detect if page needs vision model analysis."""
        if not SCRAPY_AVAILABLE:
            return False
        
        try:
            selector = Selector(response=response)
            
            # Count significant images (exclude icons, logos, small images)
            images = selector.css("img").getall()
            # Filter out small images (likely icons/logos)
            significant_images = [
                img for img in images
                if not any(skip in img.lower() for skip in ["icon", "logo", "button", "sprite"])
            ]
            
            # Check for canvas elements (charts/graphs)
            canvas_count = len(selector.css("canvas").getall())
            
            # Check for SVG charts
            svg_charts = len(selector.css("svg[class*='chart'], svg[class*='graph']").getall())
            
            # Check for PDF embeds
            pdf_embeds = len(selector.css("embed[type='application/pdf'], object[type='application/pdf']").getall())
            
            # Check for data visualization libraries indicators
            has_chart_libs = any(
                selector.css(f"script[src*='{lib}']").getall()
                for lib in ["chart.js", "d3", "plotly", "highcharts", "echarts"]
            )
            
            # Trigger vision if:
            # - More than threshold images
            # - Has canvas elements (likely charts)
            # - Has chart libraries
            # - Has PDF embeds
            needs_vision = (
                len(significant_images) > self.vision_threshold or
                canvas_count > 0 or
                svg_charts > 0 or
                pdf_embeds > 0 or
                has_chart_libs
            )
            
            return needs_vision
            
        except Exception as e:
            print(f"[WARNING] Error detecting vision needs: {e}")
            return False
    
    def _extract_with_vision(self, url: str) -> str:
        """Extract content using vision model via Playwright screenshot."""
        if not PLAYWRIGHT_AVAILABLE:
            return "Playwright not available. Install with: pip install playwright && playwright install chromium"
        
        vision_model = self._get_vision_model()
        if vision_model is None:
            return "Vision model not available. Falling back to text extraction."
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Take screenshot
                screenshot_bytes = page.screenshot(full_page=True)
                
                # Convert to PIL Image for vision model
                from PIL import Image
                from io import BytesIO
                image = Image.open(BytesIO(screenshot_bytes))
                
                browser.close()
                
                # Use vision model to analyze
                try:
                    # Create message with image for vision model
                    # LiteLLM/Ollama expects images in message content
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Analyze this webpage screenshot from {url}. Extract all visible text content, describe any charts, graphs, or images, and summarize the key information in a structured format."
                                },
                                {
                                    "type": "image",
                                    "image": image
                                }
                            ]
                        }
                    ]
                    
                    # Call vision model
                    response = vision_model.generate(messages)
                    vision_analysis = response.content if hasattr(response, 'content') else str(response)
                    
                    return f"[Vision Analysis of {url}]\n{vision_analysis}"
                    
                except Exception as e:
                    return f"Error processing image with vision model: {str(e)}. Falling back to text extraction."
                    
        except Exception as e:
            return f"Error capturing screenshot: {str(e)}"
    
    def _crawl_links(self, start_url: str, max_pages: int, link_pattern: str | None = None) -> list:
        """Crawl multiple pages following link patterns."""
        if not SCRAPY_AVAILABLE:
            return []
        
        import requests
        from urllib.parse import urljoin, urlparse
        
        visited = set()
        to_visit = [start_url]
        results = []
        
        while to_visit and len(results) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            
            visited.add(url)
            
            try:
                response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                
                scrapy_response = HtmlResponse(url=url, body=response.content)
                selector = Selector(response=scrapy_response)
                
                # Extract content from this page
                text_content = " ".join(selector.css("body *::text").getall())
                results.append({
                    "url": url,
                    "content": text_content[:5000]  # Limit per page
                })
                
                # Find links to follow
                if link_pattern:
                    # Simple pattern matching - could be enhanced
                    links = selector.css("a::attr(href)").getall()
                    for link in links:
                        full_url = urljoin(url, link)
                        if link_pattern in full_url and full_url not in visited:
                            to_visit.append(full_url)
                            
            except Exception as e:
                print(f"[WARNING] Error crawling {url}: {e}")
                continue
        
        return results
    
    def forward(
        self,
        url: str | None = None,
        css_selector: str | None = None,
        xpath_selector: str | None = None,
        start_url: str | None = None,
        max_pages: int = 1,
        link_pattern: str | None = None,
        use_vision: bool | None = None
    ) -> str:
        """Extract content from webpage(s) using Scrapy."""
        if not SCRAPY_AVAILABLE:
            return "Scrapy is not available. Install with: pip install scrapy"
        
        try:
            import requests
            
            # Determine if this is a crawl or single page
            target_url = start_url or url
            if not target_url:
                return "Error: Either 'url' or 'start_url' must be provided"
            
            # Multi-page crawling
            if start_url and max_pages > 1:
                results = self._crawl_links(start_url, max_pages, link_pattern)
                if not results:
                    return f"No content extracted from {start_url}"
                
                formatted = []
                for i, result in enumerate(results, 1):
                    formatted.append(f"\n--- Page {i}: {result['url']} ---\n{result['content']}")
                return "\n".join(formatted)[:self.max_output_length]
            
            # Single page extraction
            response = requests.get(target_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            
            scrapy_response = HtmlResponse(url=target_url, body=response.content)
            selector = Selector(response=scrapy_response)
            
            # Check if vision is needed
            needs_vision = use_vision or (use_vision is None and self._needs_vision(scrapy_response))
            
            if needs_vision and PLAYWRIGHT_AVAILABLE:
                vision_result = self._extract_with_vision(target_url)
                # For now, combine with text extraction
                # In full implementation, vision would be primary
                text_result = self._extract_text(selector, css_selector, xpath_selector)
                return f"{vision_result}\n\n--- Text Content ---\n{text_result}"[:self.max_output_length]
            
            # Standard text extraction
            return self._extract_text(selector, css_selector, xpath_selector)
            
        except requests.exceptions.Timeout:
            return f"Request to {target_url} timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"Error fetching {target_url}: {str(e)}"
        except Exception as e:
            return f"Error extracting content: {str(e)}"
    
    def _extract_text(self, selector: Selector, css_selector: str | None, xpath_selector: str | None) -> str:
        """Extract text using CSS or XPath selectors."""
        if css_selector:
            elements = selector.css(css_selector)
            text = " ".join(elements.css("*::text").getall())
        elif xpath_selector:
            elements = selector.xpath(xpath_selector)
            text = " ".join(elements.css("*::text").getall())
        else:
            # Extract all text from body
            text = " ".join(selector.css("body *::text").getall())
        
        # Clean up whitespace
        import re
        text = re.sub(r"\s+", " ", text).strip()
        
        return text[:self.max_output_length]


class ResearcherRegisterTool(Tool):
    """Tool for generating and exporting complete researcher registers."""
    
    name = "generate_researcher_register"
    description = (
        "Genererar ett komplett personregister med alla kopplingar mellan forskare, publikationer, år, ämnen och affiliations. "
        "Kan exportera i markdown, CSV eller JSON-format. Använd detta verktyg för att skapa rapporter och översikter över forskare."
    )
    inputs = {
        "format": {
            "type": "string",
            "description": "Exportformat: 'markdown', 'csv', eller 'json' (default: 'markdown')",
            "nullable": True
        },
        "include_publications": {
            "type": "boolean",
            "description": "Inkludera detaljerad publikationslista för varje forskare (default: True)",
            "nullable": True
        },
        "filter_affiliation": {
            "type": "string",
            "description": "Filtrera endast forskare från specifik affiliation",
            "nullable": True
        },
        "filter_year": {
            "type": "integer",
            "description": "Filtrera endast publikationer från specifikt år",
            "nullable": True
        },
        "output_filename": {
            "type": "string",
            "description": "Filnamn för export (utan extension, default: 'researcher_register')",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, db_path: str = "./data/publications.db"):
        super().__init__()
        self.db_path = db_path
    
    def forward(
        self,
        format: str | None = None,
        include_publications: bool | None = None,
        filter_affiliation: str | None = None,
        filter_year: int | None = None,
        output_filename: str | None = None
    ) -> str:
        """Generate researcher register in specified format."""
        try:
            import sqlite3
            from datetime import datetime
            
            format = format or "markdown"
            include_publications = include_publications if include_publications is not None else True
            output_filename = output_filename or "researcher_register"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get all researchers
                query = "SELECT * FROM researchers WHERE 1=1"
                params = []
                
                if filter_affiliation:
                    query += " AND affiliation LIKE ?"
                    params.append(f"%{filter_affiliation}%")
                
                query += " ORDER BY publications_count DESC, name ASC"
                
                cursor.execute(query, params)
                researchers = cursor.fetchall()
                
                if not researchers:
                    return "Inga forskare hittades i databasen."
                
                # Build register data
                register_data = []
                
                for researcher in researchers:
                    researcher_id = researcher["researcher_id"]
                    
                    # Get publications
                    pub_query = """
                        SELECT 
                            p.title,
                            p.year,
                            p.journal,
                            p.doi,
                            p.pmid,
                            rp.role,
                            rp.author_position
                        FROM researcher_publications rp
                        JOIN publications p ON rp.publication_key = p.unique_key
                        WHERE rp.researcher_id = ?
                    """
                    pub_params = [researcher_id]
                    
                    if filter_year:
                        pub_query += " AND p.year = ?"
                        pub_params.append(filter_year)
                    
                    pub_query += " ORDER BY p.year DESC, rp.author_position ASC"
                    
                    cursor.execute(pub_query, pub_params)
                    publications = cursor.fetchall()
                    
                    # Parse research_fields
                    research_fields = []
                    if researcher["research_fields"]:
                        try:
                            research_fields = json.loads(researcher["research_fields"])
                        except:
                            pass
                    
                    register_data.append({
                        "id": researcher_id,
                        "name": researcher["name"],
                        "email": researcher["email"] or "",
                        "affiliation": researcher["affiliation"] or "",
                        "research_fields": research_fields,
                        "publications_count": researcher["publications_count"] or 0,
                        "first_encountered": researcher["first_encountered"] or "",
                        "last_updated": researcher["last_updated"] or "",
                        "publications": [
                            {
                                "title": pub["title"] or "",
                                "year": pub["year"],
                                "journal": pub["journal"] or "",
                                "doi": pub["doi"] or "",
                                "pmid": pub["pmid"] or "",
                                "role": pub["role"] or "author",
                                "author_position": pub["author_position"]
                            }
                            for pub in publications
                        ]
                    })
            
            # Generate output based on format
            if format == "markdown":
                content = self._generate_markdown(register_data, include_publications, filter_affiliation, filter_year)
            elif format == "csv":
                content = self._generate_csv(register_data, include_publications)
            elif format == "json":
                content = self._generate_json(register_data)
            else:
                return f"Okänt format: {format}. Använd 'markdown', 'csv' eller 'json'."
            
            # Save to file
            import os
            extension = "md" if format == "markdown" else ("csv" if format == "csv" else "json")
            filepath = os.path.join("./data", f"{output_filename}.{extension}")
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Register sparad till {filepath}\n\n{len(register_data)} forskare inkluderade."
            
        except Exception as e:
            return f"Error generating register: {str(e)}"
    
    def _generate_markdown(self, register_data: list, include_publications: bool, filter_affiliation: str | None, filter_year: int | None) -> str:
        """Generate markdown format register."""
        lines = ["# Personregister\n"]
        lines.append(f"*Genererat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        if filter_affiliation:
            lines.append(f"*Filtrerat på affiliation: {filter_affiliation}*\n")
        if filter_year:
            lines.append(f"*Filtrerat på år: {filter_year}*\n")
        
        lines.append(f"\n**Totalt antal forskare:** {len(register_data)}\n")
        
        for res in register_data:
            lines.append(f"\n## {res['name']}")
            if res['email']:
                lines.append(f"- **Email:** {res['email']}")
            if res['affiliation']:
                lines.append(f"- **Affiliation:** {res['affiliation']}")
            if res['research_fields']:
                fields_str = ", ".join(res['research_fields'])
                lines.append(f"- **Forskningsområden:** {fields_str}")
            lines.append(f"- **Antal publikationer:** {res['publications_count']}")
            if res['first_encountered']:
                lines.append(f"- **Först hittad:** {res['first_encountered']}")
            if res['last_updated']:
                lines.append(f"- **Senast uppdaterad:** {res['last_updated']}")
            
            if include_publications and res['publications']:
                lines.append(f"\n### Publikationer ({len(res['publications'])}):")
                for i, pub in enumerate(res['publications'], 1):
                    role_str = f" - {pub['role']}" if pub['role'] else ""
                    pos_str = f" [Position {pub['author_position']}]" if pub['author_position'] else ""
                    lines.append(f"{i}. **({pub['year']})** {pub['title']}{role_str}{pos_str}")
                    if pub['journal']:
                        lines.append(f"   *Journal:* {pub['journal']}")
                    if pub['doi']:
                        lines.append(f"   *DOI:* {pub['doi']}")
                    if pub['pmid']:
                        lines.append(f"   *PMID:* {pub['pmid']}")
            elif include_publications:
                lines.append(f"\n### Publikationer: Inga")
        
        return "\n".join(lines)
    
    def _generate_csv(self, register_data: list, include_publications: bool) -> str:
        """Generate CSV format register."""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        if include_publications:
            writer.writerow([
                "Name", "Email", "Affiliation", "Research Fields", 
                "Publication Count", "First Encountered", "Last Updated",
                "Publications (Year:Title:Role:Position)"
            ])
        else:
            writer.writerow([
                "Name", "Email", "Affiliation", "Research Fields",
                "Publication Count", "First Encountered", "Last Updated"
            ])
        
        # Data rows
        for res in register_data:
            fields_str = ", ".join(res['research_fields']) if res['research_fields'] else ""
            
            if include_publications:
                # Format publications as semicolon-separated list
                pub_list = []
                for pub in res['publications']:
                    pub_str = f"{pub['year']}:{pub['title']}:{pub['role'] or 'author'}:{pub['author_position'] or ''}"
                    pub_list.append(pub_str)
                pubs_str = ";".join(pub_list)
                
                writer.writerow([
                    res['name'],
                    res['email'],
                    res['affiliation'],
                    fields_str,
                    res['publications_count'],
                    res['first_encountered'],
                    res['last_updated'],
                    pubs_str
                ])
            else:
                writer.writerow([
                    res['name'],
                    res['email'],
                    res['affiliation'],
                    fields_str,
                    res['publications_count'],
                    res['first_encountered'],
                    res['last_updated']
                ])
        
        return output.getvalue()
    
    def _generate_json(self, register_data: list) -> str:
        """Generate JSON format register."""
        return json.dumps(register_data, indent=2, ensure_ascii=False)

