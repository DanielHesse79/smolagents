"""Abstract base class for memory backends.

This module provides the abstraction layer for implementing different memory storage backends,
such as vector databases, that can store and retrieve agent memory steps semantically.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents.memory import MemoryStep


class MemoryBackend(ABC):
    """Abstract base class for memory backends.
    
    Memory backends provide persistent storage and semantic search capabilities
    for agent memory steps. They enable cross-session learning and experience retrieval.
    """
    
    @abstractmethod
    def add_step(self, step: "MemoryStep", agent_id: str, run_id: str) -> str:
        """Store a memory step and return its ID.
        
        Args:
            step: The memory step to store (ActionStep, PlanningStep, TaskStep, etc.)
            agent_id: Unique identifier for the agent instance
            run_id: Unique identifier for the current agent run/session
            
        Returns:
            A string identifier for the stored step
        """
        pass
    
    @abstractmethod
    def search_similar(self, query: str, k: int = 5, 
                      exclude_run_id: str | None = None) -> list["MemoryStep"]:
        """Semantic search over stored steps.
        
        Args:
            query: Search query string
            k: Number of similar steps to retrieve
            exclude_run_id: Optional run_id to exclude from results (e.g., current run)
            
        Returns:
            List of similar MemoryStep objects, ordered by relevance
        """
        pass
    
    @abstractmethod
    def get_by_id(self, step_id: str) -> "MemoryStep":
        """Retrieve a step by ID.
        
        Args:
            step_id: The identifier of the step to retrieve
            
        Returns:
            The MemoryStep object
            
        Raises:
            KeyError: If step_id is not found
        """
        pass

