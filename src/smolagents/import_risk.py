"""
Import risk assessment module for evaluating the safety of Python imports.
"""

# Low risk: Safe data processing modules
LOW_RISK_MODULES = [
    "json",
    "csv",
    "datetime",
    "time",
    "math",
    "random",
    "collections",
    "itertools",
    "functools",
    "operator",
    "string",
    "re",
    "hashlib",
    "base64",
    "urllib.parse",
    "pathlib",
    "typing",
    "dataclasses",
    "enum",
    "uuid",
    "decimal",
    "fractions",
    "statistics",
    "array",
    "bisect",
    "heapq",
    "copy",
    "pickle",
    "struct",
    "codecs",
    "unicodedata",
    "textwrap",
    "difflib",
    "pprint",
    "reprlib",
    "numbers",
    "io",
    "os.path",  # Safe path operations
    "posixpath",
    "ntpath",
]

# Medium risk: File/network access modules
MEDIUM_RISK_MODULES = [
    "os",
    "sys",
    "subprocess",
    "shutil",
    "tempfile",
    "glob",
    "fnmatch",
    "linecache",
    "fileinput",
    "mmap",
    "socket",
    "urllib",
    "urllib.request",
    "urllib.error",
    "http",
    "http.client",
    "http.server",
    "email",
    "smtplib",
    "ftplib",
    "poplib",
    "imaplib",
    "nntplib",
    "telnetlib",
    "xmlrpc",
    "webbrowser",
    "sqlite3",
    "dbm",
    "shelve",
    "configparser",
    "logging",
    "multiprocessing",
    "threading",
    "queue",
    "asyncio",
    "concurrent",
    "select",
    "selectors",
    "signal",
    "pwd",
    "grp",
    "termios",
    "tty",
    "pty",
    "fcntl",
    "resource",
    "errno",
    "ctypes",
    "platform",
    "locale",
    "gettext",
    "argparse",
    "getopt",
    "readline",
    "rlcompleter",
    "cmd",
    "shlex",
    "traceback",
    "warnings",
    "contextlib",
    "abc",
    "atexit",
    "gc",
    "inspect",
    "site",
    "sysconfig",
    "builtins",
    "__builtin__",
    "importlib",
    "pkgutil",
    "modulefinder",
    "runpy",
    "zipimport",
    "pkg_resources",
    "setuptools",
    "distutils",
    "ensurepip",
    "venv",
    "pathlib",
    "pathlib2",
]

# High risk: System commands, code execution, dangerous operations
HIGH_RISK_MODULES = [
    "subprocess",
    "os.system",
    "os.popen",
    "os.exec",
    "os.spawn",
    "os.kill",
    "os.fork",
    "posix.system",
    "posix.exec",
    "posix.spawn",
    "posix.kill",
    "posix.fork",
    "eval",
    "exec",
    "compile",
    "__import__",
    "importlib.util",
    "imp",
    "builtins.__import__",
    "builtins.eval",
    "builtins.exec",
    "builtins.compile",
    "builtins.__build_class__",
    "pickle",
    "marshal",
    "shelve",
    "dbm",
    "sqlite3",
    "ctypes",
    "cffi",
    "cryptography",
    "hashlib",
    "hmac",
    "secrets",
    "random",
    "secrets",
    "uuid",
    "token",
    "keyword",
    "ast",
    "dis",
    "symtable",
    "py_compile",
    "compileall",
    "code",
    "codeop",
    "bdb",
    "pdb",
    "profile",
    "pstats",
    "timeit",
    "trace",
    "tracemalloc",
    "faulthandler",
    "pdb",
    "cProfile",
    "hotshot",
    "doctest",
    "unittest",
    "test",
    "test.support",
    "test.support.script_helper",
    "test.support.bytecode_helper",
    "test.support.testresult",
    "test.support.testing",
    "test.support.threading_helper",
    "test.support.os_helper",
    "test.support.import_helper",
    "test.support.warnings_helper",
    "test.support.socket_helper",
    "test.support.ssl_helper",
    "test.support.http_helper",
    "test.support.subprocess_helper",
    "test.support.gc_helper",
    "test.support.sizeof_helper",
    "test.support.script_helper",
    "test.support.cleanup",
    "test.support.testing",
    "test.support.threading_helper",
    "test.support.os_helper",
    "test.support.import_helper",
    "test.support.warnings_helper",
    "test.support.socket_helper",
    "test.support.ssl_helper",
    "test.support.http_helper",
    "test.support.subprocess_helper",
    "test.support.gc_helper",
    "test.support.sizeof_helper",
    "test.support.script_helper",
    "test.support.cleanup",
]

# Blocked modules: Extremely dangerous, always blocked
BLOCKED_MODULES = [
    "__builtin__",
    "builtins.__import__",
    "builtins.eval",
    "builtins.exec",
    "builtins.compile",
    "builtins.__build_class__",
    "importlib.util.spec_from_loader",
    "importlib.util.module_from_spec",
    "importlib.util.spec_from_file_location",
    "importlib.util.module_from_spec",
    "importlib.machinery",
    "imp",
    "marshal",
    "pickle",
    "shelve",
    "dbm",
    "sqlite3",
    "ctypes",
    "cffi",
    "cryptography",
    "hashlib",
    "hmac",
    "secrets",
    "random",
    "secrets",
    "uuid",
    "token",
    "keyword",
    "ast",
    "dis",
    "symtable",
    "py_compile",
    "compileall",
    "code",
    "codeop",
    "bdb",
    "pdb",
    "profile",
    "pstats",
    "timeit",
    "trace",
    "tracemalloc",
    "faulthandler",
    "pdb",
    "cProfile",
    "hotshot",
    "doctest",
    "unittest",
    "test",
    "test.support",
    "test.support.script_helper",
    "test.support.bytecode_helper",
    "test.support.testresult",
    "test.support.testing",
    "test.support.threading_helper",
    "test.support.os_helper",
    "test.support.import_helper",
    "test.support.warnings_helper",
    "test.support.socket_helper",
    "test.support.ssl_helper",
    "test.support.http_helper",
    "test.support.subprocess_helper",
    "test.support.gc_helper",
    "test.support.sizeof_helper",
    "test.support.script_helper",
    "test.support.cleanup",
]

IMPORT_RISK_LEVELS = {
    "low": LOW_RISK_MODULES,
    "medium": MEDIUM_RISK_MODULES,
    "high": HIGH_RISK_MODULES,
}


def assess_import_risk(module_name: str) -> tuple[str, str]:
    """
    Assess the risk level of importing a module.
    
    Args:
        module_name: Name of the module to assess
        
    Returns:
        tuple[str, str]: (risk_level, reason) where risk_level is "low", "medium", "high", or "blocked"
    """
    # Check if module is blocked
    if module_name in BLOCKED_MODULES or any(
        module_name.startswith(blocked) for blocked in BLOCKED_MODULES if "." in blocked
    ):
        return ("blocked", "This module is always blocked due to security concerns")
    
    # Check risk levels
    if module_name in LOW_RISK_MODULES or any(
        module_name.startswith(low) for low in LOW_RISK_MODULES if "." in low
    ):
        return ("low", "Safe data processing module")
    
    if module_name in MEDIUM_RISK_MODULES or any(
        module_name.startswith(med) for med in MEDIUM_RISK_MODULES if "." in med
    ):
        return ("medium", "Module provides file/network access capabilities")
    
    if module_name in HIGH_RISK_MODULES or any(
        module_name.startswith(high) for high in HIGH_RISK_MODULES if "." in high
    ):
        return ("high", "Module provides system command execution or code execution capabilities")
    
    # Default: unknown modules are treated as medium risk
    return ("medium", "Unknown module - treated as medium risk by default")


def check_import_with_risk_assessment(module_name: str, risk_tolerance: str = "medium") -> tuple[bool, str | None]:
    """
    Check if an import is allowed based on risk assessment.
    
    Args:
        module_name: Name of the module to check
        risk_tolerance: Risk tolerance level ("low", "medium", or "high")
        
    Returns:
        tuple[bool, str | None]: (allowed, message) where:
            - allowed: True if import is allowed, False otherwise
            - message: Warning message if medium risk, None if no warning
    """
    risk_level, reason = assess_import_risk(module_name)
    
    # Blocked modules are never allowed
    if risk_level == "blocked":
        return (False, f"Module '{module_name}' is blocked: {reason}")
    
    # Check based on risk tolerance
    if risk_tolerance == "low":
        if risk_level == "low":
            return (True, None)
        else:
            return (False, f"Module '{module_name}' requires {risk_level} risk tolerance (current: {risk_tolerance})")
    
    elif risk_tolerance == "medium":
        if risk_level == "low":
            return (True, None)
        elif risk_level == "medium":
            return (True, f"Warning: {module_name} is a {risk_level} risk module: {reason}")
        else:  # high
            return (False, f"Module '{module_name}' requires {risk_level} risk tolerance (current: {risk_tolerance})")
    
    else:  # high
        if risk_level in ["low", "medium", "high"]:
            if risk_level == "medium":
                return (True, f"Warning: {module_name} is a {risk_level} risk module: {reason}")
            elif risk_level == "high":
                return (True, f"Warning: {module_name} is a {risk_level} risk module: {reason}")
            else:
                return (True, None)
        else:
            return (False, f"Module '{module_name}' is blocked: {reason}")

