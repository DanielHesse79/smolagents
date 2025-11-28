"""
Helper functions for publication mining tasks.

This module provides utility functions for:
- Generating unique keys for deduplication
- Parsing publication metadata from web content
"""

import hashlib
from typing import Optional, Dict, Any


def generate_unique_key(
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    title: Optional[str] = None,
    year: Optional[int] = None,
    first_author: Optional[str] = None
) -> str:
    """
    Generate unique key for deduplication.
    
    Priority:
    1. DOI (most reliable)
    2. PMID (reliable)
    3. Hash of title + year + first_author (fallback)
    
    Args:
        doi: Digital Object Identifier
        pmid: PubMed ID
        title: Publication title
        year: Publication year
        first_author: First author name
        
    Returns:
        Unique key string (format: "doi:...", "pmid:...", or "hash:...")
    """
    if doi:
        # Normalize DOI (remove https://doi.org/ prefix if present)
        doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        return f"doi:{doi_clean}"
    elif pmid:
        return f"pmid:{pmid}"
    else:
        # Generate hash from available metadata
        key_parts = []
        if title:
            key_parts.append(str(title).strip().lower())
        if year:
            key_parts.append(str(year))
        if first_author:
            key_parts.append(str(first_author).strip().lower())
        
        if not key_parts:
            # Fallback: generate random hash if no metadata available
            import uuid
            return f"hash:{uuid.uuid4().hex}"
        
        key_str = "_".join(key_parts)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"hash:{key_hash}"


def parse_publication_from_webpage(url: str, content: str) -> Dict[str, Any]:
    """
    Extract publication metadata from webpage content.
    
    This is a basic parser that looks for common patterns in publication pages.
    For more robust parsing, consider using specialized libraries or APIs.
    
    Args:
        url: URL of the webpage
        content: HTML/text content of the webpage
        
    Returns:
        Dictionary with extracted metadata (title, authors, doi, etc.)
    """
    import re
    
    metadata = {
        "url": url,
        "title": None,
        "authors": None,
        "doi": None,
        "pmid": None,
        "year": None,
        "abstract": None,
    }
    
    # Try to extract DOI
    doi_pattern = r'10\.\d{4,}/[^\s<>"]+'
    doi_match = re.search(doi_pattern, content)
    if doi_match:
        metadata["doi"] = doi_match.group(0)
    
    # Try to extract PMID
    pmid_pattern = r'PMID[:\s]*(\d+)'
    pmid_match = re.search(pmid_pattern, content, re.IGNORECASE)
    if pmid_match:
        metadata["pmid"] = pmid_match.group(1)
    
    # Try to extract year
    year_pattern = r'\b(19|20)\d{2}\b'
    year_matches = re.findall(year_pattern, content)
    if year_matches:
        # Take the most recent year found
        years = [int(match[0] + match[1]) for match in year_matches if len(match) == 2]
        if years:
            metadata["year"] = max(years)
    
    # Try to extract title (look for <title> tag or h1)
    title_patterns = [
        r'<title[^>]*>([^<]+)</title>',
        r'<h1[^>]*>([^<]+)</h1>',
    ]
    for pattern in title_patterns:
        title_match = re.search(pattern, content, re.IGNORECASE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
            break
    
    return metadata


def format_publication_markdown(publications: list[Dict[str, Any]]) -> str:
    """
    Format a list of publications as markdown.
    
    Args:
        publications: List of publication dictionaries
        
    Returns:
        Formatted markdown string
    """
    if not publications:
        return "# Publications\n\nNo publications found."
    
    markdown_lines = [
        "# Microsampling Publications",
        "",
        f"Total publications: {len(publications)}",
        "",
        "---",
        ""
    ]
    
    for i, pub in enumerate(publications, 1):
        markdown_lines.append(f"## {i}. {pub.get('title', 'Untitled')}")
        markdown_lines.append("")
        
        if pub.get("authors"):
            markdown_lines.append(f"**Authors:** {pub['authors']}")
        if pub.get("year"):
            markdown_lines.append(f"**Year:** {pub['year']}")
        if pub.get("journal"):
            markdown_lines.append(f"**Journal:** {pub['journal']}")
        if pub.get("doi"):
            markdown_lines.append(f"**DOI:** {pub['doi']}")
        if pub.get("pmid"):
            markdown_lines.append(f"**PMID:** {pub['pmid']}")
        if pub.get("url"):
            markdown_lines.append(f"**URL:** {pub['url']}")
        if pub.get("device_workflow"):
            markdown_lines.append(f"**Device/Workflow:** {pub['device_workflow']}")
        if pub.get("brand"):
            markdown_lines.append(f"**Brand:** {pub['brand']}")
        if pub.get("sample_type"):
            markdown_lines.append(f"**Sample Type:** {pub['sample_type']}")
        if pub.get("application"):
            markdown_lines.append(f"**Application:** {pub['application']}")
        if pub.get("evidence_snippet"):
            markdown_lines.append(f"**Evidence:** {pub['evidence_snippet']}")
        if pub.get("abstract"):
            markdown_lines.append(f"**Abstract:** {pub['abstract'][:500]}..." if len(pub.get('abstract', '')) > 500 else f"**Abstract:** {pub['abstract']}")
        
        markdown_lines.append("")
        markdown_lines.append("---")
        markdown_lines.append("")
    
    return "\n".join(markdown_lines)


# ============================================================================
# Error Pattern Analysis Functions
# ============================================================================

def analyze_error_patterns(db_path: str = "./data/publications.db") -> list[dict]:
    """
    Analyze error patterns from database.
    
    Returns:
        list[dict]: List of error patterns with frequency information
    """
    import sqlite3
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Find most common error types
        cursor.execute("""
            SELECT error_type, COUNT(*) as count
            FROM agent_errors
            GROUP BY error_type
            ORDER BY count DESC
            LIMIT 10
        """)
        
        patterns = []
        for row in cursor.fetchall():
            error_type, count = row
            patterns.append({"error_type": error_type, "frequency": count})
        
        conn.close()
        return patterns
    except Exception as e:
        print(f"Error analyzing patterns: {e}")
        return []


def get_learning_suggestions(db_path: str, error_type: str) -> str | None:
    """
    Get learning suggestions based on error patterns.
    
    Args:
        db_path: Path to SQLite database
        error_type: Type of error to get suggestions for
    
    Returns:
        str | None: Solution pattern if found, None otherwise
    """
    import sqlite3
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT solution_pattern, effectiveness_score
            FROM agent_learning_patterns
            WHERE error_type = ?
            ORDER BY effectiveness_score DESC, frequency DESC
            LIMIT 1
        """, (error_type,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]  # solution_pattern
        return None
    except Exception as e:
        print(f"Error getting learning suggestions: {e}")
        return None


def update_learning_pattern(
    db_path: str,
    error_type: str,
    common_cause: str | None = None,
    solution_pattern: str | None = None,
    effectiveness_score: float | None = None
):
    """
    Update or create a learning pattern in the database.
    
    Args:
        db_path: Path to SQLite database
        error_type: Type of error
        common_cause: Common cause of this error type
        solution_pattern: Pattern for solving this error
        effectiveness_score: How effective the solution is (0-1)
    """
    import sqlite3
    from datetime import datetime
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if pattern exists
        cursor.execute("SELECT pattern_id, frequency FROM agent_learning_patterns WHERE error_type = ?", (error_type,))
        existing = cursor.fetchone()
        
        now = datetime.now().isoformat()
        
        if existing:
            pattern_id, frequency = existing
            # Update existing pattern
            cursor.execute("""
                UPDATE agent_learning_patterns SET
                    common_cause = COALESCE(?, common_cause),
                    solution_pattern = COALESCE(?, solution_pattern),
                    frequency = frequency + 1,
                    last_seen = ?,
                    effectiveness_score = COALESCE(?, effectiveness_score)
                WHERE pattern_id = ?
            """, (common_cause, solution_pattern, now, effectiveness_score, pattern_id))
        else:
            # Insert new pattern
            cursor.execute("""
                INSERT INTO agent_learning_patterns (
                    error_type, common_cause, solution_pattern, frequency, last_seen, effectiveness_score
                ) VALUES (?, ?, ?, 1, ?, ?)
            """, (error_type, common_cause, solution_pattern, now, effectiveness_score))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating learning pattern: {e}")


def get_researcher_by_name(db_path: str, name: str, email: str | None = None) -> dict | None:
    """
    Get researcher information by name and optionally email.
    
    Returns:
        dict | None: Researcher information if found, None otherwise
    """
    import sqlite3
    import json
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if email:
            cursor.execute("""
                SELECT researcher_id, name, email, affiliation, research_fields,
                       relevance_reason, first_encountered, publications_count,
                       contact_info, notes, last_updated
                FROM researchers
                WHERE name = ? AND email = ?
            """, (name, email))
        else:
            cursor.execute("""
                SELECT researcher_id, name, email, affiliation, research_fields,
                       relevance_reason, first_encountered, publications_count,
                       contact_info, notes, last_updated
                FROM researchers
                WHERE name = ? AND email IS NULL
            """, (name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "researcher_id": row[0],
                "name": row[1],
                "email": row[2],
                "affiliation": row[3],
                "research_fields": json.loads(row[4]) if row[4] else [],
                "relevance_reason": row[5],
                "first_encountered": row[6],
                "publications_count": row[7],
                "contact_info": json.loads(row[8]) if row[8] else {},
                "notes": row[9],
                "last_updated": row[10]
            }
        return None
    except Exception as e:
        print(f"Error getting researcher: {e}")
        return None

