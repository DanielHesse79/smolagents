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
        with sqlite3.connect(db_path) as conn:
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
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT solution_pattern, effectiveness_score
                FROM agent_learning_patterns
                WHERE error_type = ?
                ORDER BY effectiveness_score DESC, frequency DESC
                LIMIT 1
            """, (error_type,))
            
            result = cursor.fetchone()
            
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
        with sqlite3.connect(db_path) as conn:
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
        with sqlite3.connect(db_path) as conn:
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


# ============================================================================
# Code Validation Functions
# ============================================================================

def validate_code_for_executor(code: str) -> tuple[bool, str, list[str]]:
    """
    Validate code for common issues before execution in the restricted Python executor.
    
    This function checks for:
    - Forbidden imports (sys, subprocess, builtins, io, etc.)
    - Ellipsis in unpacking patterns
    - Dunder attribute access
    - Dangerous function calls (eval, exec, compile)
    - Nested function definitions (security risk)
    - Class definitions (can be used to bypass restrictions)
    - Lambda functions with dangerous operations
    - List/dict comprehensions with side effects
    - Basic syntax validation
    
    Args:
        code: Python code string to validate
        
    Returns:
        tuple[bool, str, list[str]]: (is_valid, error_message, warnings)
        - is_valid: True if code passes validation, False otherwise
        - error_message: Empty string if valid, descriptive error message with line numbers if invalid
        - warnings: List of warning messages for potentially problematic code
    """
    import ast
    import re
    
    warnings = []
    errors = []
    
    # Parse AST first for detailed analysis
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        error_msg = f"Syntax error on line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f"\n{e.text}"
            if e.offset:
                error_msg += f"\n{' ' * (e.offset - 1)}^"
        return False, error_msg, []
    
    # List of forbidden modules (from DANGEROUS_MODULES in local_python_executor.py)
    forbidden_modules = [
        'sys', 'subprocess', 'builtins', 'io', 'multiprocessing', 
        'pty', 'shutil', 'socket', 'eval', 'exec', 'compile'
    ]
    
    # AST visitor to check for various issues
    class CodeValidator(ast.NodeVisitor):
        def __init__(self):
            self.errors = []
            self.warnings = []
            self.line_numbers = {}
            
        def visit_Import(self, node):
            for alias in node.names:
                module_name = alias.name.split('.')[0]  # Get base module name
                if module_name in forbidden_modules:
                    self.errors.append(
                        f"Line {node.lineno}: Forbidden import '{alias.name}'. "
                        f"This module is not allowed in the restricted executor."
                    )
            self.generic_visit(node)
            
        def visit_ImportFrom(self, node):
            if node.module:
                module_name = node.module.split('.')[0]  # Get base module name
                if module_name in forbidden_modules:
                    self.errors.append(
                        f"Line {node.lineno}: Forbidden import 'from {node.module}'. "
                        f"This module is not allowed in the restricted executor."
                    )
            self.generic_visit(node)
            
        def visit_Call(self, node):
            # Check for dangerous function calls
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in ['eval', 'exec', 'compile']:
                    self.errors.append(
                        f"Line {node.lineno}: Dangerous function '{func_name}()' is forbidden. "
                        f"Use alternative approaches that don't require dynamic code execution."
                    )
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id in forbidden_modules:
                    func_name = f"{node.func.value.id}.{node.func.attr}"
                    self.errors.append(
                        f"Line {node.lineno}: Dangerous function call '{func_name}()' is forbidden."
                    )
            self.generic_visit(node)
            
        def visit_Attribute(self, node):
            # Check for forbidden dunder attribute access
            if isinstance(node.attr, str) and node.attr.startswith('__') and node.attr.endswith('__'):
                allowed_dunders = ['__init__', '__str__', '__repr__']
                if node.attr not in allowed_dunders:
                    suggestions = {
                        '__class__': "Use 'type(obj)' instead",
                        '__dict__': "Use 'vars(obj)' or 'getattr()' instead",
                        '__module__': "This attribute is not accessible in restricted executor",
                        '__name__': "This attribute is not accessible in restricted executor"
                    }
                    suggestion = suggestions.get(node.attr, "Only __init__, __str__, and __repr__ are allowed")
                    self.errors.append(
                        f"Line {node.lineno}: Dunder attribute access '{node.attr}' is forbidden. {suggestion}."
                    )
            self.generic_visit(node)
            
        def visit_FunctionDef(self, node):
            # Warn about nested function definitions (security risk)
            if any(isinstance(parent, (ast.FunctionDef, ast.ClassDef, ast.Lambda)) 
                   for parent in ast.walk(node) if parent != node):
                self.warnings.append(
                    f"Line {node.lineno}: Nested function definition '{node.name}' detected. "
                    f"Nested functions can be a security risk in restricted executors."
                )
            self.generic_visit(node)
            
        def visit_ClassDef(self, node):
            # Warn about class definitions (can be used to bypass restrictions)
            self.warnings.append(
                f"Line {node.lineno}: Class definition '{node.name}' detected. "
                f"Classes can potentially be used to bypass executor restrictions."
            )
            self.generic_visit(node)
            
        def visit_Lambda(self, node):
            # Check lambda for dangerous operations
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id in ['eval', 'exec', 'compile']:
                        self.errors.append(
                            f"Line {node.lineno}: Lambda function contains dangerous operation '{child.func.id}()'."
                        )
            self.generic_visit(node)
            
        def visit_ListComp(self, node):
            # Warn about list comprehensions with side effects
            for generator in node.generators:
                for if_expr in generator.ifs:
                    # Check if comprehension has side effects (calls, assignments, etc.)
                    for child in ast.walk(if_expr):
                        if isinstance(child, ast.Call):
                            self.warnings.append(
                                f"Line {node.lineno}: List comprehension contains function calls. "
                                f"Side effects in comprehensions can be problematic."
                            )
                            return
            self.generic_visit(node)
            
        def visit_DictComp(self, node):
            # Warn about dict comprehensions with side effects
            for generator in node.generators:
                for if_expr in generator.ifs:
                    for child in ast.walk(if_expr):
                        if isinstance(child, ast.Call):
                            self.warnings.append(
                                f"Line {node.lineno}: Dict comprehension contains function calls. "
                                f"Side effects in comprehensions can be problematic."
                            )
                            return
            self.generic_visit(node)
    
    # Check for ellipsis in unpacking using regex (AST doesn't capture this well)
    if re.search(r'\.\.\s*=', code):
        # Try to find line number
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if re.search(r'\.\.\s*=', line):
                errors.append(
                    f"Line {i}: Ellipsis (...) in unpacking is not allowed. "
                    f"Use '*rest' or indexing instead. Example: 'a, b, *rest = items' instead of 'a, b, ... = items'"
                )
                break
    
    # Run AST visitor
    validator = CodeValidator()
    validator.visit(tree)
    errors.extend(validator.errors)
    warnings.extend(validator.warnings)
    
    # Return results
    if errors:
        error_msg = "\n".join(errors)
        if len(errors) > 1:
            error_msg = f"Found {len(errors)} validation errors:\n" + error_msg
        return False, error_msg, warnings
    
    return True, "", warnings

