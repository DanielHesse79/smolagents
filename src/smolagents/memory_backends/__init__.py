"""Memory backends for persistent agent memory storage.

This package provides backends for storing and retrieving agent memory steps,
enabling semantic search and cross-session learning.
"""

from smolagents.memory_backends.memory_backends import MemoryBackend

__all__ = ["MemoryBackend"]

try:
    from smolagents.memory_backends.qdrant_backend import QdrantMemoryBackend
    __all__.append("QdrantMemoryBackend")
except ImportError:
    # qdrant optional dependency not installed
    pass

