"""
Example demonstrating Qdrant vector memory backend for agent memory storage.

This example shows how to:
1. Set up Qdrant memory backend
2. Create an agent with persistent memory
3. Demonstrate cross-session learning
4. Manually search similar experiences

Prerequisites:
- Qdrant server running (install: docker run -p 6333:6333 qdrant/qdrant)
- Install dependencies: pip install 'smolagents[qdrant]'
"""

import sys
import os

# Add parent directory to path to use local source (for development)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents import CodeAgent, LiteLLMModel
from smolagents.memory_backends import QdrantMemoryBackend

# Initialize Qdrant backend
# Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant
print("Initializing Qdrant memory backend...")
memory_backend = QdrantMemoryBackend(
    url="localhost",
    port=6333,
    collection_name="example_agent_memory",
    embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",  # Fast, lightweight model
)

# Create initial model with Ollama (or use any other model)
# You can also use InferenceClientModel, OpenAIModel, etc.
try:
    model = LiteLLMModel(
        model_id="ollama_chat/mistral:latest",
        api_base="http://localhost:11434",
        api_key="ollama",
        num_ctx=8192,
    )
except Exception as e:
    print(f"Note: Could not initialize Ollama model ({e})")
    print("Using a different model provider or ensure Ollama is running.")
    # Fallback example (commented out - uncomment and configure as needed):
    # from smolagents import InferenceClientModel
    # model = InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct")
    raise

# Create agent with vector memory backend
print("Creating agent with Qdrant memory backend...")
agent = CodeAgent(
    tools=[],
    model=model,
    verbosity_level=1,
    planning_interval=3,
    name="example_agent_with_memory",
    description="An example agent with persistent vector memory.",
    stream_outputs=True,
    max_steps=10,
    # Enable vector memory backend
    memory_backend=memory_backend,
    enable_experience_retrieval=True,  # Automatic retrieval of similar experiences
)

print("\n" + "="*80)
print("Example 1: First run - agent learns and stores experience")
print("="*80)
result1 = agent.run("What is 5+5?")
print(f"\nResult: {result1}")

print("\n" + "="*80)
print("Example 2: Second run - agent can retrieve similar experiences")
print("="*80)
# Create a new agent instance (simulating a different session)
# The memory backend is shared, so it will have access to previous experiences
agent2 = CodeAgent(
    tools=[],
    model=model,
    verbosity_level=1,
    planning_interval=3,
    name="example_agent_with_memory",
    description="An example agent with persistent vector memory.",
    stream_outputs=True,
    max_steps=10,
    memory_backend=memory_backend,  # Same backend instance
    enable_experience_retrieval=True,
)

result2 = agent2.run("What is 10+10?")
print(f"\nResult: {result2}")

print("\n" + "="*80)
print("Example 3: Manual search for similar experiences")
print("="*80)
# Manually search for similar experiences
similar_experiences = agent.memory.search_similar_experiences(
    query="How to calculate basic arithmetic?",
    k=3
)
print(f"\nFound {len(similar_experiences)} similar experiences:")
for i, exp in enumerate(similar_experiences, 1):
    if hasattr(exp, 'model_output'):
        print(f"\nExperience {i}:")
        print(f"  Output: {exp.model_output[:100]}..." if exp.model_output else "  (No output)")
        if hasattr(exp, 'observations') and exp.observations:
            print(f"  Observations: {exp.observations[:100]}...")

print("\n" + "="*80)
print("Example 4: Cross-session learning demonstration")
print("="*80)
# Run a similar but different task - agent should benefit from previous experiences
result3 = agent.run("Calculate 15 + 25 and explain the process")
print(f"\nResult: {result3}")

print("\n" + "="*80)
print("Memory Statistics")
print("="*80)
print(f"Current run ID: {agent.memory.run_id}")
print(f"Agent ID: {agent.memory.agent_id}")
print(f"Total steps in current session: {len(agent.memory.steps)}")
print(f"Memory backend enabled: {agent.memory.backend is not None}")

print("\n" + "="*80)
print("Example completed!")
print("="*80)
print("\nAll agent experiences have been stored in Qdrant and are available for")
print("semantic search in future sessions. The agent can now learn from past runs!")

