"""Qdrant vector database backend for agent memory storage.

This module provides a Qdrant-based implementation of the MemoryBackend interface,
enabling semantic search over stored agent memory steps.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents.memory import MemoryStep

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from smolagents.memory_backends import MemoryBackend
from smolagents.memory import ActionStep, PlanningStep, TaskStep, SystemPromptStep, FinalAnswerStep
from smolagents.monitoring import Timing, TokenUsage
from smolagents.models import ChatMessage
from smolagents.utils import AgentError


class QdrantMemoryBackend(MemoryBackend):
    """Qdrant-based memory backend for semantic search over agent memory steps.
    
    Args:
        url: Qdrant server URL (default: "localhost")
        port: Qdrant server port (default: 6333)
        collection_name: Name of the Qdrant collection (default: "smolagents_memory")
        embedding_model_id: Hugging Face model ID for embeddings (default: "sentence-transformers/all-MiniLM-L6-v2")
        api_key: Optional API key for Qdrant Cloud
        
    Raises:
        ImportError: If qdrant-client or sentence-transformers are not installed
    """
    
    def __init__(
        self,
        url: str = "localhost",
        port: int = 6333,
        collection_name: str = "smolagents_memory",
        embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: str | None = None,
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client not available. Install with: pip install 'smolagents[qdrant]'"
            )
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Sentence transformers not available. Install with: pip install 'smolagents[qdrant]'"
            )
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=url, port=port, api_key=api_key)
        self.collection_name = collection_name
        self.embedding_model_id = embedding_model_id
        
        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model_id)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
    
    def _extract_step_text(self, step: "MemoryStep") -> str:
        """Extract searchable text from a memory step.
        
        Args:
            step: The memory step to extract text from
            
        Returns:
            Concatenated text content for embedding
        """
        texts = []
        
        if isinstance(step, ActionStep):
            if step.model_output:
                texts.append(str(step.model_output))
            if step.observations:
                texts.append(str(step.observations))
            if step.code_action:
                texts.append(str(step.code_action))
            if step.error:
                texts.append(f"Error: {str(step.error)}")
        
        elif isinstance(step, PlanningStep):
            if step.plan:
                texts.append(str(step.plan))
        
        elif isinstance(step, TaskStep):
            if step.task:
                texts.append(str(step.task))
        
        elif isinstance(step, SystemPromptStep):
            if step.system_prompt:
                texts.append(str(step.system_prompt))
        
        elif isinstance(step, FinalAnswerStep):
            if step.output:
                texts.append(str(step.output))
        
        return " ".join(texts)
    
    def add_step(self, step: "MemoryStep", agent_id: str, run_id: str) -> str:
        """Store a memory step in Qdrant.
        
        Args:
            step: The memory step to store
            agent_id: Unique identifier for the agent instance
            run_id: Unique identifier for the current agent run
            
        Returns:
            A string identifier for the stored step
        """
        # Extract text for embedding
        text = self._extract_step_text(step)
        if not text.strip():
            # Skip empty steps
            return ""
        
        # Generate embedding
        embedding = self.embedder.encode(text, normalize_embeddings=True).tolist()
        
        # Generate unique step ID
        step_id = str(uuid.uuid4())
        
        # Store metadata
        step_dict = step.dict()
        payload = {
            "step_id": step_id,
            "step_type": type(step).__name__,
            "step_dict": step_dict,
            "agent_id": agent_id,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Generate point ID (hash of step_id for consistency)
        point_id = abs(hash(step_id)) % (2**63)
        
        # Store in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
        
        return step_id
    
    def search_similar(
        self, query: str, k: int = 5, exclude_run_id: str | None = None
    ) -> list["MemoryStep"]:
        """Semantic search for similar memory steps.
        
        Args:
            query: Search query string
            k: Number of similar steps to retrieve
            exclude_run_id: Optional run_id to exclude from results
            
        Returns:
            List of similar MemoryStep objects ordered by relevance
        """
        if not query.strip():
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
        
        # Build filter to exclude current run if specified
        query_filter = None
        if exclude_run_id:
            query_filter = Filter(
                must_not=[
                    FieldCondition(key="run_id", match=MatchValue(value=exclude_run_id))
                ]
            )
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=query_filter,
        )
        
        # Reconstruct MemoryStep objects
        similar_steps = []
        for result in results:
            try:
                payload = result.payload
                step_type = payload.get("step_type")
                step_dict = payload.get("step_dict", {})
                
                step = self._reconstruct_step(step_type, step_dict)
                if step is not None:
                    similar_steps.append(step)
            except Exception as e:
                # Skip steps that can't be reconstructed
                continue
        
        return similar_steps
    
    def _reconstruct_step(self, step_type: str, step_dict: dict) -> "MemoryStep | None":
        """Reconstruct a MemoryStep object from stored dictionary.
        
        Args:
            step_type: Name of the step class
            step_dict: Dictionary representation of the step
            
        Returns:
            Reconstructed MemoryStep object or None if reconstruction fails
        """
        try:
            if step_type == "ActionStep":
                # Reconstruct Timing
                timing_dict = step_dict.get("timing", {})
                timing = Timing(
                    start_time=timing_dict.get("start_time", 0.0),
                    end_time=timing_dict.get("end_time"),
                )
                
                # Reconstruct TokenUsage if present
                token_usage = None
                if step_dict.get("token_usage"):
                    tu_dict = step_dict["token_usage"]
                    token_usage = TokenUsage(
                        input_tokens=tu_dict.get("input_tokens", 0),
                        output_tokens=tu_dict.get("output_tokens", 0),
                    )
                
                # Reconstruct AgentError if present
                error = None
                if step_dict.get("error"):
                    error_dict = step_dict["error"]
                    error = AgentError(
                        message=error_dict.get("message", ""),
                        logger=None,  # Logger can't be reconstructed
                    )
                
                return ActionStep(
                    step_number=step_dict.get("step_number", 0),
                    timing=timing,
                    model_input_messages=None,  # Not reconstructing full messages
                    tool_calls=None,  # Can be reconstructed if needed
                    error=error,
                    model_output_message=None,
                    model_output=step_dict.get("model_output"),
                    code_action=step_dict.get("code_action"),
                    observations=step_dict.get("observations"),
                    observations_images=None,  # Images not stored in vector DB
                    action_output=step_dict.get("action_output"),
                    token_usage=token_usage,
                    is_final_answer=step_dict.get("is_final_answer", False),
                )
            
            elif step_type == "PlanningStep":
                timing_dict = step_dict.get("timing", {})
                timing = Timing(
                    start_time=timing_dict.get("start_time", 0.0),
                    end_time=timing_dict.get("end_time"),
                )
                
                token_usage = None
                if step_dict.get("token_usage"):
                    tu_dict = step_dict["token_usage"]
                    token_usage = TokenUsage(
                        input_tokens=tu_dict.get("input_tokens", 0),
                        output_tokens=tu_dict.get("output_tokens", 0),
                    )
                
                return PlanningStep(
                    model_input_messages=[],  # Not reconstructing full messages
                    model_output_message=None,
                    plan=step_dict.get("plan", ""),
                    timing=timing,
                    token_usage=token_usage,
                )
            
            elif step_type == "TaskStep":
                return TaskStep(
                    task=step_dict.get("task", ""),
                    task_images=None,  # Images not stored
                )
            
            elif step_type == "SystemPromptStep":
                return SystemPromptStep(
                    system_prompt=step_dict.get("system_prompt", "")
                )
            
            elif step_type == "FinalAnswerStep":
                return FinalAnswerStep(
                    output=step_dict.get("output")
                )
            
        except Exception:
            # Return None if reconstruction fails
            return None
        
        return None
    
    def get_by_id(self, step_id: str) -> "MemoryStep":
        """Retrieve a step by ID.
        
        Args:
            step_id: The identifier of the step to retrieve
            
        Returns:
            The MemoryStep object
            
        Raises:
            KeyError: If step_id is not found
        """
        # Search by step_id in payload
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="step_id", match=MatchValue(value=step_id))
                ]
            ),
            limit=1,
        )
        
        points = results[0]  # First element is points
        if not points:
            raise KeyError(f"Step with id {step_id} not found")
        
        point = points[0]
        payload = point.payload
        step_type = payload.get("step_type")
        step_dict = payload.get("step_dict", {})
        
        step = self._reconstruct_step(step_type, step_dict)
        if step is None:
            raise KeyError(f"Could not reconstruct step with id {step_id}")
        
        return step

