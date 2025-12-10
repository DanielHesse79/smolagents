"""
Daniel's Army of Agents - Multi-Agent System (Gradio UI)

This setup uses a manager-agent architecture:
- Manager Agent (Mistral): Understands user prompts and delegates tasks
- Programming Agent (DeepSeek R1 8B): Handles all coding and programming tasks

Gradio 5.50 implementation with ChatInterface for better streaming and UX.

To run:
    python examples/gradio_ui.py

Or with Gradio installed:
    gradio examples/gradio_ui.py
"""

import os
import sys
import time
import traceback
import json
from typing import Optional, Generator, Dict, Any

# Add project root and src to path for local imports during development
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# Force reload of smolagents modules to avoid cached bytecode issues
# This ensures we always use the latest code from the source directory
modules_to_reload = [k for k in sys.modules.keys() if k.startswith('smolagents')]
for module in modules_to_reload:
    del sys.modules[module]

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is required. Install with: pip install 'smolagents[gradio]' or pip install gradio>=5.50.0"
    )

from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel, InferenceClientModel, WebSearchTool, GoogleSearchTool
from smolagents.default_tools import VisitWebpageTool
from smolagents.agent_types import AgentImage, AgentText
from smolagents.agents import PlanningStep, RunResult
from smolagents.memory import ActionStep, FinalAnswerStep
from smolagents.models import ChatMessageStreamDelta, agglomerate_stream_deltas

# Import publication tools and helpers
try:
    from examples.publication_tools import (
        PubMedSearchTool,
        QdrantUpsertTool,
        SQLUpsertTool,
        MarkdownFileWriterTool,
        ErrorLoggingTool,
        ResearcherUpsertTool,
        ResearcherPublicationLinkTool,
        ResearcherQueryTool,
        ResearcherRegisterTool,
        ScrapyVisionTool,
    )
    PUBLICATION_TOOLS_AVAILABLE = True
except ImportError as e:
    try:
        import sys
        import os
        examples_path = os.path.dirname(os.path.abspath(__file__))
        if examples_path not in sys.path:
            sys.path.insert(0, examples_path)
        from publication_tools import (
            PubMedSearchTool,
            QdrantUpsertTool,
            SQLUpsertTool,
            MarkdownFileWriterTool,
            ErrorLoggingTool,
            ResearcherUpsertTool,
            ResearcherPublicationLinkTool,
            ResearcherQueryTool,
            ResearcherRegisterTool,
            ScrapyVisionTool,
        )
        PUBLICATION_TOOLS_AVAILABLE = True
    except ImportError:
        PUBLICATION_TOOLS_AVAILABLE = False
        print(f"Warning: Publication tools not available: {e}")

# Import Open Deep Research components
try:
    sys.path.insert(0, os.path.join(project_root, "examples", "open_deep_research"))
    from scripts.text_web_browser import (
        ArchiveSearchTool,
        FinderTool,
        FindNextTool,
        PageDownTool,
        PageUpTool,
        SimpleTextBrowser,
        VisitTool,
    )
    from scripts.text_inspector_tool import TextInspectorTool
    from scripts.visual_qa import visualizer
    OPEN_DEEP_RESEARCH_AVAILABLE = True
except ImportError as e:
    OPEN_DEEP_RESEARCH_AVAILABLE = False
    print(f"Warning: Open Deep Research tools not available: {e}")

# Try to import QdrantMemoryBackend for persistent memory
try:
    from smolagents.memory_backends import QdrantMemoryBackend
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantMemoryBackend = None


# ============================================================================
# Startup Configuration and Result Classes (from shared_agent_utils.py)
# ============================================================================

from examples.shared_agent_utils import (
    _retry_with_backoff,
    check_ollama_health,
    check_qdrant_health,
    check_sqlite_health,
    run_startup_checks,
    initialize_memory_backend,
    initialize_qdrant_client,
    initialize_sqlite_db,
    setup_ollama_models,
    setup_api_models,
    create_programming_agent,
    StartupConfig,
    StartupResult,
    load_model_preferences,
    save_model_preferences,
)
# Note: We use create_manager_agent_gradio instead of create_manager_agent


# ============================================================================
# Streaming function for Gradio (UI-agnostic)
# ============================================================================

def stream_to_gradio(
    agent,
    task: str,
    task_images: list | None = None,
    reset_agent_memory: bool = False,
    additional_args: dict | None = None,
    max_steps: int | None = None,
) -> Generator:
    """Runs an agent with the given task and streams the messages from the agent for Gradio display."""
    from smolagents.memory import ActionStep, FinalAnswerStep
    from smolagents.agents import PlanningStep
    from smolagents.models import ChatMessageStreamDelta, agglomerate_stream_deltas
    from smolagents.agent_types import AgentImage, AgentText, AgentAudio
    import re
    
    def _clean_model_output(model_output: str) -> str:
        """Clean up model output by removing trailing tags and extra backticks."""
        if not model_output:
            return ""
        model_output = model_output.strip()
        model_output = re.sub(r"```\s*<end_code>", "```", model_output)
        model_output = re.sub(r"<end_code>\s*```", "```", model_output)
        model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)
        return model_output.strip()
    
    def _format_code_content(content: str) -> str:
        """Format code content as Python code block if it's not already formatted."""
        content = content.strip()
        content = re.sub(r"```.*?\n", "", content)
        content = re.sub(r"\s*<end_code>\s*", "", content)
        content = content.strip()
        if not content.startswith("```python"):
            content = f"```python\n{content}\n```"
        return content
    
    def _process_action_step(step_log: ActionStep, skip_model_outputs: bool = False) -> Generator:
        """Process an ActionStep and yield appropriate content for Gradio display."""
        step_number = f"Step {step_log.step_number}"
        if not skip_model_outputs:
            yield {"type": "markdown", "content": f"**{step_number}**"}
        
        if not skip_model_outputs and getattr(step_log, "model_output", ""):
            model_output = _clean_model_output(step_log.model_output)
            yield {"type": "markdown", "content": model_output}
        
        if getattr(step_log, "tool_calls", []):
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()
            if used_code:
                content = _format_code_content(content)
            yield {
                "type": "tool_call",
                "content": content,
                "tool_name": first_tool_call.name,
            }
        
        if getattr(step_log, "observations", "") and step_log.observations.strip():
            log_content = step_log.observations.strip()
            if log_content:
                log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                yield {"type": "code", "content": log_content, "language": "bash"}
        
        if getattr(step_log, "observations_images", []):
            for image in step_log.observations_images:
                yield {"type": "image", "content": image}
        
        if getattr(step_log, "error", None):
            yield {"type": "error", "content": str(step_log.error)}
    
    def _process_planning_step(step_log: PlanningStep, skip_model_outputs: bool = False) -> Generator:
        """Process a PlanningStep and yield appropriate content for Gradio display."""
        if not skip_model_outputs:
            yield {"type": "markdown", "content": "**Planning step**"}
            yield {"type": "markdown", "content": step_log.plan}
    
    def _process_final_answer_step(step_log: FinalAnswerStep) -> Generator:
        """Process a FinalAnswerStep and yield appropriate content for Gradio display."""
        final_answer = step_log.output
        if isinstance(final_answer, AgentText):
            yield {"type": "markdown", "content": f"**Final answer:**\n{final_answer.to_string()}\n"}
        elif isinstance(final_answer, AgentImage):
            yield {"type": "image", "content": final_answer.to_string()}
        elif isinstance(final_answer, AgentAudio):
            yield {"type": "audio", "content": final_answer.to_string()}
        else:
            yield {"type": "markdown", "content": f"**Final answer:** {str(final_answer)}"}
    
    accumulated_events: list[ChatMessageStreamDelta] = []
    for event in agent.run(
        task,
        images=task_images,
        stream=True,
        reset=reset_agent_memory,
        additional_args=additional_args,
        max_steps=max_steps,
    ):
        if isinstance(event, ActionStep | PlanningStep | FinalAnswerStep):
            skip_model_outputs = getattr(agent, "stream_outputs", False)
            if isinstance(event, ActionStep):
                yield from _process_action_step(event, skip_model_outputs)
            elif isinstance(event, PlanningStep):
                yield from _process_planning_step(event, skip_model_outputs)
            elif isinstance(event, FinalAnswerStep):
                yield from _process_final_answer_step(event)
            accumulated_events = []
        elif isinstance(event, ChatMessageStreamDelta):
            accumulated_events.append(event)
            text = agglomerate_stream_deltas(accumulated_events).render_as_markdown()
            yield {"type": "stream", "content": text}


def process_stream_chunk(chunk: Any, current_response: str) -> tuple[str, str]:
    """Update the running response string based on a streamed chunk."""
    if not isinstance(chunk, dict):
        updated = current_response + "\n\n" + str(chunk)
        return updated, updated
    
    chunk_type = chunk.get("type")
    if chunk_type == "stream":
        updated = chunk.get("content", "")
        return updated, updated
    if chunk_type == "markdown":
        content = chunk.get("content", "")
        updated = current_response + "\n\n" + content
        return updated, updated
    if chunk_type == "code":
        content = chunk.get("content", "")
        code_block = f"\n\n```python\n{content}\n```"
        updated = current_response + code_block
        return updated, updated
    if chunk_type == "tool_call":
        tool_name = chunk.get("tool_name", "tool")
        tool_msg = f"\n\n🛠️ **Used tool {tool_name}**\n\n```\n{chunk.get('content', '')}\n```"
        updated = current_response + tool_msg
        return updated, updated
    
    updated = current_response + "\n\n" + str(chunk)
    return updated, updated


# ============================================================================
# Gradio UI Components
# ============================================================================

def format_status_markdown(startup_result: StartupResult) -> str:
    """Format overall status as markdown."""
    if startup_result.all_critical_services_ready:
        return "## ✅ All critical services are ready!"
    else:
        return "## ❌ Some critical services are not available"


def format_ollama_status(ollama_status: dict) -> str:
    """Format Ollama status as markdown."""
    lines = []
    if ollama_status["available"]:
        lines.append("### ✅ Ollama is running")
        if ollama_status.get("version"):
            lines.append(f"**Version:** {ollama_status['version']}")
        
        # GPU Status
        gpu_available = ollama_status.get("gpu_available", False)
        gpu_info = ollama_status.get("gpu_info")
        recommended_num_ctx = ollama_status.get("recommended_num_ctx", 8192)
        
        lines.append("\n**GPU Status:**")
        if gpu_available:
            lines.append(f"✅ GPU Available: {gpu_info if gpu_info else 'GPU detected'}")
            lines.append(f"💡 Recommended context length: {recommended_num_ctx}")
        else:
            lines.append("⚠️ GPU not detected - using CPU (slower)")
            lines.append("💡 Tip: GPU acceleration requires NVIDIA GPU with CUDA support")
        
        lines.append("\n**Available Models:**")
        if ollama_status["models"]:
            for model in ollama_status["models"]:
                lines.append(f"- {model}")
        else:
            lines.append("- No models found")
        
        lines.append("\n**Required Models:**")
        for model, found in ollama_status["required_models"].items():
            status = "✅" if found else "❌"
            lines.append(f"{status} {model}")
            if not found:
                lines.append(f"  Install with: `ollama pull {model}`")
    else:
        lines.append("### ❌ Ollama is not available")
        if ollama_status.get("error"):
            lines.append(f"**Error:** {ollama_status['error']}")
        lines.append("💡 Tip: Start Ollama or use API models instead")
    
    return "\n".join(lines)


def format_qdrant_status(qdrant_status: dict) -> str:
    """Format Qdrant status as markdown."""
    lines = []
    if qdrant_status["available"]:
        lines.append("### ✅ Qdrant is running")
        lines.append(f"**URL:** {qdrant_status['url']}:{qdrant_status['port']}")
        
        lines.append("\n**Available Collections:**")
        if qdrant_status["collections"]:
            for coll in qdrant_status["collections"]:
                lines.append(f"- {coll}")
        else:
            lines.append("- No collections yet (will be created as needed)")
        
        lines.append("\n**Required Collections:**")
        for coll, found in qdrant_status["required_collections"].items():
            status = "✅" if found else "⚠️"
            lines.append(f"{status} {coll}")
            if not found:
                lines.append("  Will be created automatically")
    else:
        lines.append("### ⚠️ Qdrant is not available")
        if qdrant_status.get("error"):
            lines.append(f"**Error:** {qdrant_status['error']}")
        lines.append("💡 Tip: Start Qdrant with: `docker run -p 6333:6333 qdrant/qdrant`")
    
    return "\n".join(lines)


def format_sqlite_status(sqlite_status: dict) -> str:
    """Format SQLite status as markdown."""
    lines = []
    if sqlite_status["available"]:
        lines.append("### ✅ SQLite database is ready")
        lines.append(f"**Path:** {sqlite_status['path']}")
        
        lines.append("\n**Available Tables:**")
        if sqlite_status["tables"]:
            for table in sqlite_status["tables"]:
                lines.append(f"- {table}")
        else:
            lines.append("- No tables yet (will be created)")
        
        lines.append("\n**Required Tables:**")
        for table, found in sqlite_status["required_tables"].items():
            status = "✅" if found else "⚠️"
            lines.append(f"{status} {table}")
            if not found:
                lines.append("  Will be created automatically")
    else:
        lines.append("### ❌ SQLite is not available")
        if sqlite_status.get("error"):
            lines.append(f"**Error:** {sqlite_status['error']}")
    
    return "\n".join(lines)


def create_startup_interface() -> tuple:
    """Create Gradio interface for startup checks.
    
    Returns:
        tuple: (startup_ui, startup_result_state, startup_config_state)
    """
    startup_result_state = gr.State(value=None)
    startup_config_state = gr.State(value=None)
    
    def retry_checks():
        """Retry startup checks."""
        config = StartupConfig()
        result = run_startup_checks(config)
        return (
            result,
            config,
            format_status_markdown(result),
            format_ollama_status(result.ollama),
            format_qdrant_status(result.qdrant),
            format_sqlite_status(result.sqlite),
            "\n".join([f"- {w}" for w in result.warnings]) if result.warnings else "No warnings",
            "\n".join([f"- {e}" for e in result.errors]) if result.errors else "No errors",
            gr.update(interactive=result.all_critical_services_ready),
        )
    
    with gr.Blocks(title="Daniel's Army of Agents - System Startup") as startup_ui:
        gr.Markdown("# 🚀 Daniel's Army of Agents - System Startup")
        gr.Markdown("Checking system dependencies...")
        
        # Overall status
        status_display = gr.Markdown()
        
        # Service status accordions
        with gr.Accordion("🦙 Ollama Status", open=True):
            ollama_display = gr.Markdown()
        
        with gr.Accordion("🗄️ Qdrant Status", open=True):
            qdrant_display = gr.Markdown()
        
        with gr.Accordion("💾 SQLite Status", open=True):
            sqlite_display = gr.Markdown()
        
        # Warnings and errors
        with gr.Accordion("⚠️ Warnings", open=False):
            warnings_display = gr.Markdown()
        
        with gr.Accordion("❌ Errors", open=False):
            errors_display = gr.Markdown()
        
        # Action buttons
        with gr.Row():
            retry_btn = gr.Button("🔄 Retry Checks", variant="secondary")
            continue_btn = gr.Button("✅ Continue to App", variant="primary")
        
        # Initial load
        startup_ui.load(
            fn=retry_checks,
            outputs=[
                startup_result_state,
                startup_config_state,
                status_display,
                ollama_display,
                qdrant_display,
                sqlite_display,
                warnings_display,
                errors_display,
                continue_btn,
            ],
        )
        
        # Retry button
        retry_btn.click(
            fn=retry_checks,
            outputs=[
                startup_result_state,
                startup_config_state,
                status_display,
                ollama_display,
                qdrant_display,
                sqlite_display,
                warnings_display,
                errors_display,
                continue_btn,
            ],
        )
    
    return startup_ui, startup_result_state, startup_config_state


def save_uploaded_file(file, upload_folder: str = "./data") -> Optional[str]:
    """Save uploaded file to disk."""
    if file is None:
        return None
    
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, os.path.basename(file.name))
    
    # Copy file content
    with open(file_path, "wb") as f:
        if hasattr(file, "read"):
            f.write(file.read())
        else:
            # Handle different file types
            import shutil
            shutil.copy(file.name, file_path)
    
    return file_path


# ============================================================================
# Document Reader Tool
# ============================================================================

class SimpleDocumentReaderTool:
    """Simple tool to read PDF and document files."""
    name = "read_document"
    description = """Read the content of a document file (PDF, DOCX, TXT) and extract its text.
    Use this tool when you need to read uploaded files.
    Supported formats: .pdf, .docx, .txt"""
    
    inputs = {
        "file_path": {
            "description": "The path to the document file to read",
            "type": "string",
        }
    }
    output_type = "string"
    
    def __init__(self):
        pass
    
    def forward(self, file_path: str) -> str:
        """Read document and return text content."""
        import os
        
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            
            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(file_path)
                    text_parts = []
                    for page in reader.pages:
                        text_parts.append(page.extract_text() or "")
                    return "\n\n".join(text_parts)
                except ImportError:
                    try:
                        # Fallback to pdfplumber
                        import pdfplumber
                        with pdfplumber.open(file_path) as pdf:
                            text_parts = []
                            for page in pdf.pages:
                                text_parts.append(page.extract_text() or "")
                            return "\n\n".join(text_parts)
                    except ImportError:
                        return "Error: PDF reading requires 'pypdf' or 'pdfplumber'. Install with: pip install pypdf"
            
            elif ext == ".docx":
                try:
                    from docx import Document
                    doc = Document(file_path)
                    paragraphs = [p.text for p in doc.paragraphs]
                    return "\n\n".join(paragraphs)
                except ImportError:
                    return "Error: DOCX reading requires 'python-docx'. Install with: pip install python-docx"
            
            else:
                # Try to read as text
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
                    
        except Exception as e:
            return f"Error reading file: {str(e)}"


# ============================================================================
# Model Management Functions
# ============================================================================

def get_available_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Get list of available Ollama models."""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            return [m.get("name", "") for m in models_data]
    except Exception:
        pass
    return []


def get_vision_capable_models(models: list[str]) -> list[str]:
    """Filter models that support vision/image input."""
    vision_keywords = ["llava", "vision", "vl", "qwen2.5vl", "llama3.2-vision", "bakllava", "moondream"]
    vision_models = []
    for model in models:
        model_lower = model.lower()
        if any(kw in model_lower for kw in vision_keywords):
            vision_models.append(model)
    return vision_models


def create_model_from_name(model_name: str, base_url: str = "http://localhost:11434", config: StartupConfig = None) -> LiteLLMModel:
    """Create a LiteLLMModel from a model name."""
    if config is None:
        config = StartupConfig()
    
    return LiteLLMModel(
        model_id=f"ollama_chat/{model_name}",
        api_base=base_url,
        api_key="ollama",
        timeout=config.ollama_timeout,
        max_tokens=config.ollama_max_tokens,
        num_ctx=config.ollama_num_ctx,
        temperature=config.ollama_temperature,
        top_p=config.ollama_top_p,
    )


# ============================================================================
# Monitoring and Metrics
# ============================================================================

def get_agent_metrics(agent) -> Dict[str, Any]:
    """Extract metrics from agent."""
    metrics = {
        "total_steps": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_duration": 0.0,
        "steps": [],
    }
    
    if hasattr(agent, "memory") and agent.memory:
        steps = agent.memory.steps
        metrics["total_steps"] = len(steps)
        
        for step in steps:
            if hasattr(step, "token_usage") and step.token_usage:
                metrics["input_tokens"] += step.token_usage.input_tokens
                metrics["output_tokens"] += step.token_usage.output_tokens
                metrics["total_tokens"] += step.token_usage.total_tokens
            
            if hasattr(step, "timing") and step.timing:
                duration = step.timing.duration or 0.0
                metrics["total_duration"] += duration
                metrics["steps"].append({
                    "step_number": getattr(step, "step_number", 0),
                    "duration": duration,
                    "tokens": step.token_usage.total_tokens if hasattr(step, "token_usage") and step.token_usage else 0,
                })
    
    return metrics


def format_metrics_markdown(metrics: Dict[str, Any]) -> str:
    """Format metrics as markdown."""
    lines = [
        "## 📊 Agent Metrics",
        "",
        f"**Total Steps:** {metrics['total_steps']}",
        f"**Total Tokens:** {metrics['total_tokens']:,}",
        f"  - Input: {metrics['input_tokens']:,}",
        f"  - Output: {metrics['output_tokens']:,}",
        f"**Total Duration:** {metrics['total_duration']:.2f}s",
    ]
    
    if metrics["total_duration"] > 0:
        tokens_per_sec = metrics["total_tokens"] / metrics["total_duration"]
        lines.append(f"**Tokens/sec:** {tokens_per_sec:.1f}")
    
    return "\n".join(lines)


def create_open_deep_research_agent(model, text_limit=100000):
    """Create Open Deep Research agent with web browsing capabilities (fully local, no API keys required)."""
    if not OPEN_DEEP_RESEARCH_AVAILABLE:
        return None
    
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    browser_config = {
        "viewport_size": 1024 * 5,
        "downloads_folder": "./downloads_folder",
        "request_kwargs": {
            "headers": {"User-Agent": user_agent},
            "timeout": 300,
        },
        # Make serpapi_key optional - will use local search if not provided
        "serpapi_key": os.getenv("SERPAPI_API_KEY") or os.getenv("SERPER_API_KEY") or None,
    }
    
    os.makedirs(browser_config["downloads_folder"], exist_ok=True)
    browser = SimpleTextBrowser(**browser_config)
    
    # Use local DuckDuckGo search instead of GoogleSearchTool (no API key required)
    # If API keys are available, we could use GoogleSearchTool, but for local-only we use DuckDuckGo
    from smolagents import DuckDuckGoSearchTool, WebSearchTool
    
    # Try DuckDuckGoSearchTool first (requires ddgs package), fallback to WebSearchTool
    try:
        search_tool = DuckDuckGoSearchTool(max_results=10, rate_limit=1.0)
    except ImportError:
        # Fallback to WebSearchTool which uses DuckDuckGo lite directly
        search_tool = WebSearchTool(max_results=10, engine="duckduckgo")
    
    web_tools = [
        search_tool,  # Local search, no API key needed
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    
    search_agent = ToolCallingAgent(
        model=model,
        tools=web_tools,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    
    search_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""
    
    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)] if OPEN_DEEP_RESEARCH_AVAILABLE else [],
        max_steps=12,
        verbosity_level=2,
        import_risk_tolerance="high",  # Manager needs flexibility for Open Deep Research
        planning_interval=4,
        managed_agents=[search_agent],
    )
    
    return manager_agent


def create_manager_agent_gradio(
    model,
    programming_agent,
    memory_backend,
    startup_config: StartupConfig,
    include_open_deep_research: bool = False,
):
    """Create manager agent for Gradio."""
    print("Creating manager agent (Mistral)...")
    
    # Prepare tools for manager agent
    manager_tools = [
        WebSearchTool(max_results=20),
        VisitWebpageTool(max_output_length=30_000),
    ]
    
    # Prepare managed agents list
    managed_agents = [programming_agent]
    
    # Add Open Deep Research if requested
    if include_open_deep_research and OPEN_DEEP_RESEARCH_AVAILABLE:
        try:
            odr_agent = create_open_deep_research_agent(model)
            if odr_agent:
                managed_agents.append(odr_agent)
                print("[OK] Open Deep Research agent added")
        except Exception as e:
            print(f"[WARNING] Could not add Open Deep Research: {e}")
    
    # Add PubMed search tool and researcher tools if available
    if PUBLICATION_TOOLS_AVAILABLE:
        manager_tools.append(PubMedSearchTool())
        db_path_final = "./data/publications.db"
        manager_tools.append(ErrorLoggingTool(db_path=db_path_final))
        manager_tools.append(ResearcherUpsertTool(db_path=db_path_final))
        manager_tools.append(ResearcherQueryTool(db_path=db_path_final))
        manager_tools.append(ResearcherRegisterTool(db_path=db_path_final))
        
        # Add ScrapyVisionTool with vision model support (use config directly)
        try:
            ollama_base_url = startup_config.ollama_base_url if startup_config else "http://localhost:11434"
            
            manager_tools.append(ScrapyVisionTool(
                vision_model_id="ollama_chat/qwen2.5vl:7b",
                ollama_base_url=ollama_base_url,
            ))
            print("[OK] ScrapyVisionTool with qwen2.5vl:7b vision support added to manager agent")
        except Exception as e:
            print(f"[WARNING] Could not add ScrapyVisionTool: {e}")
        
        print("[OK] PubMed search tool and error/researcher tools added to manager agent")
    
    manager_agent = CodeAgent(
        tools=manager_tools,
        model=model,
        managed_agents=managed_agents,
        verbosity_level=1,
        planning_interval=3,
        stream_outputs=True,
        max_steps=20,
        memory_backend=memory_backend,
        enable_experience_retrieval=True,
        instructions="""You are a manager agent that delegates programming tasks to the 'programmer' agent.

DELEGATION:
- Provide clear task descriptions with expected inputs/outputs
- Delegate ALL programming tasks (code, data processing, file operations, database queries)
- Handle non-programming tasks yourself (web search, information gathering)

UNDERSTANDING USER INTENT:
- When users ask about database contents (e.g., "how many publications/researchers are stored"), understand they likely want you to:
  1. FIRST search online if the database might be empty
  2. Store the results in the database
  3. THEN query and report the counts
- Don't just query an empty database - be proactive about searching and populating data when needed
- If a user asks "query the database" about publications/researchers, interpret this as "search for publications/researchers, store them, then tell me how many are stored"

CRITICAL RESTRICTIONS (remind programmer):
1. IMPORT RISK ASSESSMENT: Imports are assessed by risk level.
   - LOW RISK (always allowed): math, json, datetime, pandas, numpy, re, collections, PIL
   - MEDIUM RISK (allowed with warnings): os, sys, requests, io, pickle, threading
   - HIGH RISK (always blocked): subprocess, socket, multiprocessing, ctypes, builtins
   The current tolerance level determines which risk levels are allowed.
   ⚠️ CRITICAL: Never create a variable called 'authorized_imports' - this is informational text only!

2. FORBIDDEN OPERATIONS: eval(), exec(), compile(), globals(), locals(), __import__()
   Forbidden dunder access: __class__, __dict__, __module__ (use type(obj), vars(obj) instead)

3. FORBIDDEN SYNTAX: NEVER use '...' (ellipsis) in unpacking. Use '*rest' or indexing instead.

4. CODE FORMAT: Always wrap code in <code>...</code> tags when delegating.

ERROR HANDLING:
- If programmer errors occur, guide them to: log_agent_error, analyze_error_patterns(), get_learning_suggestions(), update_learning_pattern()
- After errors, remind: "Learn from the error. Document it, analyze patterns, build prevention."

PUBLICATION MINING:
- Use pubmed_search/web_search to find publications
- CRITICAL: pubmed_search() returns a STRING, not a list! You MUST parse it to extract the JSON data.
- The output contains [STRUCTURED_DATA]...[/STRUCTURED_DATA] tags with JSON inside.
- CORRECT USAGE PATTERN:
  ```python
  import json
  import re
  
  # Step 1: Call pubmed_search (returns a string)
  search_result = pubmed_search("microsampling AND Mitra", max_results=100)
  
  # Step 2: Extract JSON from [STRUCTURED_DATA] section
  match = re.search(r'\\[STRUCTURED_DATA\\](.*?)\\[/STRUCTURED_DATA\\]', search_result, re.DOTALL)
  if match:
      json_str = match.group(1).strip()
      publications = json.loads(json_str)  # Now you have a list of dicts
  else:
      publications = []
  
  # Step 3: Filter by country/affiliation
  swedish_pubs = []
  for pub in publications:
      affiliations = pub.get('affiliations', [])
      # Check if any affiliation contains country keywords
      if any('Sweden' in aff or 'Swedish' in aff or 'Stockholm' in aff or 'Uppsala' in aff 
             for aff in affiliations):
          swedish_pubs.append(pub)
  
  final_answer(swedish_pubs)
  ```
- WRONG: Don't iterate over pubmed_search() result directly - it's a string, not a list!
- Delegate to programmer for: parsing JSON, filtering by affiliations, deduplication (use DOI/PMID/hash), storing (qdrant_upsert_publication, sql_upsert_publication), writing markdown, researcher tracking (upsert_researcher, link_researcher_publication)

RESEARCHER TRACKING:
- Use upsert_researcher to store researcher info
- Link to publications with link_researcher_publication
""",
    )
    print("[OK] Manager agent created successfully")
    return manager_agent


# ============================================================================
# Monitoring Tab
# ============================================================================

def create_monitoring_tab():
    """Create monitoring and visualization tab content."""
    gr.Markdown("## 📊 Agent Performance Metrics")
    gr.Markdown("*Click Refresh to update metrics after agent runs*")
    
    metrics_display = gr.Markdown()
    refresh_btn = gr.Button("🔄 Refresh All", variant="primary")
    
    def refresh_metrics():
        agent = _global_manager_agent
        if agent:
            metrics = get_agent_metrics(agent)
            return format_metrics_markdown(metrics)
        return "No agent available. Run a task first in the Chat tab."
    
    # Code extraction (for CodeAgent)
    gr.Markdown("## 💻 Generated Code")
    gr.Markdown("*Shows all Python code generated by the agent during execution*")
    code_display = gr.Code(language="python", label="Generated Code", lines=20)
    
    def get_full_code():
        code_parts = []
        
        # Try manager agent first
        agent = _global_manager_agent
        if agent and isinstance(agent, CodeAgent) and hasattr(agent, "memory"):
            try:
                code = agent.memory.return_full_code()
                if code and code.strip():
                    code_parts.append(f"# === Manager Agent Code ===\n{code}")
            except Exception:
                pass
        
        # Also try programming agent
        prog_agent = _global_programming_agent
        if prog_agent and isinstance(prog_agent, CodeAgent) and hasattr(prog_agent, "memory"):
            try:
                code = prog_agent.memory.return_full_code()
                if code and code.strip():
                    code_parts.append(f"# === Programming Agent Code ===\n{code}")
            except Exception:
                pass
        
        if code_parts:
            return "\n\n".join(code_parts)
        return "# No code generated yet.\n# Run a task in the Chat tab, then click Refresh."
    
    code_btn = gr.Button("💻 Get Full Code", variant="secondary")
    
    # Memory inspection
    gr.Markdown("## 📋 Memory Inspection")
    memory_display = gr.Code(language="json", label="Memory Steps (JSON)", lines=15)
    
    def get_memory_json():
        all_steps = {}
        
        # Manager agent memory
        agent = _global_manager_agent
        if agent and hasattr(agent, "memory"):
            try:
                steps = agent.memory.get_full_steps()
                all_steps["manager_agent"] = steps
            except Exception:
                pass
        
        # Programming agent memory
        prog_agent = _global_programming_agent
        if prog_agent and hasattr(prog_agent, "memory"):
            try:
                steps = prog_agent.memory.get_full_steps()
                all_steps["programming_agent"] = steps
            except Exception:
                pass
        
        if all_steps:
            return json.dumps(all_steps, indent=2, default=str)
        return '{"message": "No memory data yet. Run a task first."}'
    
    memory_btn = gr.Button("📋 Get Memory", variant="secondary")
    
    # Connect buttons
    refresh_btn.click(fn=refresh_metrics, outputs=[metrics_display])
    code_btn.click(fn=get_full_code, outputs=[code_display])
    memory_btn.click(fn=get_memory_json, outputs=[memory_display])
    
    # Refresh all button
    def refresh_all():
        return refresh_metrics(), get_full_code(), get_memory_json()
    
    refresh_all_btn = gr.Button("🔄 Refresh Everything", variant="secondary")
    refresh_all_btn.click(fn=refresh_all, outputs=[metrics_display, code_display, memory_display])


# ============================================================================
# Agent Functions Tab
# ============================================================================

def create_agent_functions_tab():
    """Create tab content with all agent functions accessible."""
    gr.Markdown("## Direct Agent Function Access")
    
    # Memory Operations
    with gr.Accordion("Memory Operations", open=True):
        with gr.Row():
            with gr.Column():
                reset_memory_btn = gr.Button("🔄 Reset Memory", variant="secondary")
                reset_memory_status = gr.Markdown()
            with gr.Column():
                get_steps_btn = gr.Button("📋 Get Full Steps", variant="secondary")
                steps_download = gr.File(label="Download Steps JSON")
            with gr.Column():
                get_succinct_btn = gr.Button("📝 Get Succinct Steps", variant="secondary")
                succinct_download = gr.File(label="Download Succinct JSON")
        
        def reset_memory():
            agent = _global_manager_agent
            if agent and hasattr(agent, "memory"):
                agent.memory.reset()
                return "✅ Memory reset successfully"
            return "❌ No agent available"
        
        def get_full_steps():
            agent = _global_manager_agent
            if agent and hasattr(agent, "memory"):
                steps = agent.memory.get_full_steps()
                json_str = json.dumps(steps, indent=2, default=str)
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(json_str)
                    return f.name
            return None
        
        def get_succinct_steps():
            agent = _global_manager_agent
            if agent and hasattr(agent, "memory"):
                steps = agent.memory.get_succinct_steps()
                json_str = json.dumps(steps, indent=2, default=str)
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(json_str)
                    return f.name
            return None
        
        reset_memory_btn.click(fn=reset_memory, outputs=[reset_memory_status])
        get_steps_btn.click(fn=get_full_steps, outputs=[steps_download])
        get_succinct_btn.click(fn=get_succinct_steps, outputs=[succinct_download])
    
    # Agent Control
    with gr.Accordion("Agent Control", open=True):
        with gr.Row():
            with gr.Column():
                interrupt_btn = gr.Button("⏸️ Interrupt Agent", variant="secondary")
                interrupt_status = gr.Markdown()
            with gr.Column():
                cleanup_btn = gr.Button("🧹 Cleanup Agent", variant="secondary")
                cleanup_status = gr.Markdown()
        
        def interrupt_agent():
            agent = _global_manager_agent
            if agent and hasattr(agent, "interrupt"):
                agent.interrupt()
                return "✅ Interrupt signal sent"
            return "❌ No agent available or interrupt not supported"
        
        def cleanup_agent():
            agent = _global_manager_agent
            if agent and hasattr(agent, "cleanup"):
                agent.cleanup()
                return "✅ Agent cleaned up"
            return "❌ No agent available"
        
        interrupt_btn.click(fn=interrupt_agent, outputs=[interrupt_status])
        cleanup_btn.click(fn=cleanup_agent, outputs=[cleanup_status])
    
    # Serialization
    with gr.Accordion("Serialization", open=False):
        with gr.Row():
            with gr.Column():
                export_dict_btn = gr.Button("📤 Export to Dict", variant="secondary")
                export_dict_display = gr.Code(language="json")
            with gr.Column():
                visualize_btn = gr.Button("🌳 Visualize Agent Tree", variant="secondary")
                visualize_status = gr.Markdown()
        
        def export_to_dict():
            agent = _global_manager_agent
            if agent and hasattr(agent, "to_dict"):
                return json.dumps(agent.to_dict(), indent=2, default=str)
            return "{}"
        
        def visualize_tree():
            agent = _global_manager_agent
            if agent and hasattr(agent, "visualize"):
                agent.visualize()
                return "✅ Visualization generated (check console/terminal)"
            return "❌ No agent available or visualize not supported"
        
        export_dict_btn.click(fn=export_to_dict, outputs=[export_dict_display])
        visualize_btn.click(fn=visualize_tree, outputs=[visualize_status])


# ============================================================================
# Agent Initialization Function
# ============================================================================

def initialize_agents_with_models(programming_model_name: str, manager_model_name: str, startup_result: StartupResult, config: StartupConfig):
    """Initialize agents with selected models."""
    global _global_manager_agent, _global_programming_agent
    global _global_current_programming_model, _global_current_manager_model
    
    try:
        # Setup models
        if startup_result.ollama["available"]:
            programming_model, manager_model = setup_ollama_models(
                startup_result.ollama, 
                config,
                programming_model_name=programming_model_name,
                manager_model_name=manager_model_name
            )
            _global_current_programming_model = programming_model.model_id.replace("ollama_chat/", "")
            _global_current_manager_model = manager_model.model_id.replace("ollama_chat/", "")
        else:
            programming_model, manager_model = setup_api_models()
            _global_current_programming_model = "API Model"
            _global_current_manager_model = "API Model"
        
        # Create agents
        programming_agent = create_programming_agent(
            model=programming_model,
            memory_backend=_global_memory_backend,
            db_path=_global_db_path,
            qdrant_collection_name="microsampling_publications"
        )
        
        manager_agent = create_manager_agent_gradio(
            model=manager_model,
            programming_agent=programming_agent,
            memory_backend=_global_memory_backend,
            startup_config=config
        )
        
        # Store in global
        _global_manager_agent = manager_agent
        _global_programming_agent = programming_agent
        
        return True, f"✅ Agents initialized successfully!\n- Programming: {_global_current_programming_model}\n- Manager: {_global_current_manager_model}"
    except Exception as e:
        import traceback
        error_msg = f"❌ Error initializing agents: {str(e)}\n\n{traceback.format_exc()}"
        return False, error_msg


# ============================================================================
# Main Function
# ============================================================================

# Global state for agents (cannot use gr.State as agents are not pickleable)
_global_manager_agent = None
_global_programming_agent = None
_global_startup_result = None
_global_startup_config = None
_global_available_models = []
_global_vision_models = []
_global_current_programming_model = None
_global_current_manager_model = None
_global_memory_backend = None
_global_db_path = None
_global_document_reader = SimpleDocumentReaderTool()

def main():
    """Main entry point for Gradio UI."""
    global _global_manager_agent, _global_programming_agent, _global_startup_result, _global_startup_config
    global _global_available_models, _global_vision_models, _global_current_programming_model
    global _global_current_manager_model, _global_memory_backend, _global_db_path
    
    # Run startup checks
    print("=" * 80)
    print("[STARTUP] Starting Gradio UI application...")
    print("=" * 80)
    print("[STARTUP] Running startup checks...")
    import sys
    sys.stdout.flush()
    config = StartupConfig()
    startup_result = run_startup_checks(config)
    
    # Store in global
    _global_startup_result = startup_result
    _global_startup_config = config
    
    # Get available models
    if startup_result.ollama.get("available", False):
        _global_available_models = get_available_ollama_models(startup_result.ollama.get("url", "http://localhost:11434"))
        _global_vision_models = get_vision_capable_models(_global_available_models)
        print(f"[OK] Found {len(_global_available_models)} Ollama models, {len(_global_vision_models)} vision-capable")
    
    # Initialize agents if ready
    manager_agent = None
    programming_agent = None
    
    # Check if we have saved model preferences
    saved_preferences = load_model_preferences() if startup_result.ollama.get("available", False) else {}
    has_saved_models = bool(saved_preferences.get("programming_model") and saved_preferences.get("manager_model"))
    
    # Check if saved models are still available
    models_available = False
    if has_saved_models and startup_result.ollama.get("available", False):
        available_models = startup_result.ollama.get("models", [])
        prog_model = saved_preferences.get("programming_model")
        mgr_model = saved_preferences.get("manager_model")
        models_available = prog_model in available_models and mgr_model in available_models
    
    # Initialize agents if ready and we have valid saved models
    if startup_result.all_critical_services_ready and models_available:
        print("[STARTUP] Initializing services with saved model preferences...")
        import sys
        sys.stdout.flush()
        memory_backend = initialize_memory_backend(startup_result.qdrant)
        qdrant_client = initialize_qdrant_client(startup_result.qdrant)
        db_path = initialize_sqlite_db(startup_result.sqlite)
        
        _global_memory_backend = memory_backend
        _global_db_path = db_path
        
        # Setup models using saved preferences
        if startup_result.ollama["available"]:
            print("[STARTUP] Setting up Ollama models from saved preferences...")
            import sys
            sys.stdout.flush()
            programming_model, manager_model = setup_ollama_models(
                startup_result.ollama, 
                config,
                programming_model_name=saved_preferences.get("programming_model"),
                manager_model_name=saved_preferences.get("manager_model")
            )
            print(f"[STARTUP] Models ready: {programming_model.model_id}, {manager_model.model_id}")
            sys.stdout.flush()
            # Store current model names
            _global_current_programming_model = programming_model.model_id.replace("ollama_chat/", "")
            _global_current_manager_model = manager_model.model_id.replace("ollama_chat/", "")
        else:
            print("[STARTUP] Setting up API models (fallback)...")
            import sys
            sys.stdout.flush()
            programming_model, manager_model = setup_api_models()
            _global_current_programming_model = "API Model"
            _global_current_manager_model = "API Model"
            print("[STARTUP] API models ready")
            sys.stdout.flush()
        
        # Create agents
        print("[STARTUP] Creating programming agent...")
        import sys
        sys.stdout.flush()
        programming_agent = create_programming_agent(
            model=programming_model,
            memory_backend=memory_backend,
            db_path=db_path,
            qdrant_collection_name="microsampling_publications"
        )
        print("[STARTUP] Programming agent created successfully")
        sys.stdout.flush()
        
        # Create manager agent using Gradio-specific function
        print("[STARTUP] Creating manager agent...")
        import sys
        sys.stdout.flush()
        manager_agent = create_manager_agent_gradio(
            model=manager_model,
            programming_agent=programming_agent,
            memory_backend=memory_backend,
            startup_config=config
        )
        print("[STARTUP] Manager agent created successfully")
        sys.stdout.flush()
        
        # Store in global
        _global_manager_agent = manager_agent
        _global_programming_agent = programming_agent
    elif startup_result.all_critical_services_ready and startup_result.ollama.get("available", False):
        # Services ready but no valid saved models - will need model selection
        print("[STARTUP] Services ready but model selection required...")
        import sys
        sys.stdout.flush()
        # Initialize non-model services
        memory_backend = initialize_memory_backend(startup_result.qdrant)
        qdrant_client = initialize_qdrant_client(startup_result.qdrant)
        db_path = initialize_sqlite_db(startup_result.sqlite)
        
        _global_memory_backend = memory_backend
        _global_db_path = db_path
    
    # Create main interface with tabs
    print("[STARTUP] Building Gradio interface...")
    import sys
    sys.stdout.flush()
    with gr.Blocks(title="Daniel's Army of Agents") as main_ui:
        gr.Markdown("# 🚀 Daniel's Army of Agents")
        
        # State for startup result and config (these are pickleable)
        startup_result_state = gr.State(value=startup_result)
        startup_config_state = gr.State(value=config)
        
        with gr.Tabs() as tabs:
            # Startup Checks Tab
            with gr.Tab("Startup Checks"):
                # Create startup interface content directly in tab
                startup_result_state = gr.State(value=startup_result)
                startup_config_state = gr.State(value=config)
                
                def retry_checks():
                    """Retry startup checks."""
                    config = StartupConfig()
                    result = run_startup_checks(config)
                    return (
                        result,
                        config,
                        format_status_markdown(result),
                        format_ollama_status(result.ollama),
                        format_qdrant_status(result.qdrant),
                        format_sqlite_status(result.sqlite),
                        "\n".join([f"- {w}" for w in result.warnings]) if result.warnings else "No warnings",
                        "\n".join([f"- {e}" for e in result.errors]) if result.errors else "No errors",
                        gr.update(interactive=result.all_critical_services_ready),
                    )
                
                gr.Markdown("# 🚀 System Startup Checks")
                gr.Markdown("Checking system dependencies...")
                
                # Overall status
                status_display = gr.Markdown()
                
                # Service status accordions
                with gr.Accordion("🦙 Ollama Status", open=True):
                    ollama_display = gr.Markdown()
                
                with gr.Accordion("🗄️ Qdrant Status", open=True):
                    qdrant_display = gr.Markdown()
                
                with gr.Accordion("💾 SQLite Status", open=True):
                    sqlite_display = gr.Markdown()
                
                # Warnings and errors
                with gr.Accordion("⚠️ Warnings", open=False):
                    warnings_display = gr.Markdown()
                
                with gr.Accordion("❌ Errors", open=False):
                    errors_display = gr.Markdown()
                
                # Action buttons
                with gr.Row():
                    retry_btn = gr.Button("🔄 Retry Checks", variant="secondary")
                
                # Initial load
                main_ui.load(
                    fn=retry_checks,
                    outputs=[
                        startup_result_state,
                        startup_config_state,
                        status_display,
                        ollama_display,
                        qdrant_display,
                        sqlite_display,
                        warnings_display,
                        errors_display,
                    ],
                )
                
                # Retry button
                retry_btn.click(
                    fn=retry_checks,
                    outputs=[
                        startup_result_state,
                        startup_config_state,
                        status_display,
                        ollama_display,
                        qdrant_display,
                        sqlite_display,
                        warnings_display,
                        errors_display,
                    ],
                )
            
            # Chat Tab
            with gr.Tab("Chat"):
                chat_container = gr.Column(visible=manager_agent is not None)
                init_container = gr.Column(visible=manager_agent is None)
                
                with init_container:
                    gr.Markdown("## ⚠️ Agents not initialized")
                    gr.Markdown("Please check the Startup Checks tab and ensure all critical services are ready, then click 'Initialize Agents' below.")
                    
                    def initialize_agents():
                        """Initialize agents on demand."""
                        global _global_manager_agent, _global_programming_agent
                        
                        if not startup_result_state.value or not startup_result_state.value.all_critical_services_ready:
                            return (
                                "Please fix errors in Startup Checks tab first.",
                                gr.update(visible=False),
                                gr.update(visible=True)
                            )
                        
                        result = startup_result_state.value
                        cfg = startup_config_state.value
                        
                        print("Initializing services...")
                        memory_backend = initialize_memory_backend(result.qdrant)
                        qdrant_client = initialize_qdrant_client(result.qdrant)
                        db_path = initialize_sqlite_db(result.sqlite)
                        
                        # Setup models
                        if result.ollama["available"]:
                            programming_model, manager_model = setup_ollama_models(result.ollama, cfg)
                        else:
                            programming_model, manager_model = setup_api_models()
                        
                        # Create agents
                        prog_agent = create_programming_agent(
                            model=programming_model,
                            memory_backend=memory_backend,
                            db_path=db_path,
                            qdrant_collection_name="microsampling_publications"
                        )
                        
                        # Create manager agent using Gradio-specific function
                        mgr_agent = create_manager_agent_gradio(
                            model=manager_model,
                            programming_agent=prog_agent,
                            memory_backend=memory_backend,
                            startup_config=cfg
                        )
                        
                        # Store in global
                        _global_manager_agent = mgr_agent
                        _global_programming_agent = prog_agent
                        
                        return (
                            "✅ Agents initialized successfully! Chat interface is now available.",
                            gr.update(visible=True),
                            gr.update(visible=False)
                        )
                    
                    init_btn = gr.Button("Initialize Agents", variant="primary")
                    init_status = gr.Markdown()
                    init_btn.click(
                        fn=initialize_agents,
                        outputs=[
                            init_status,
                            chat_container,
                            init_container
                        ]
                    )
                
                with chat_container:
                    if manager_agent and programming_agent:
                        # Create chat interface inline (ChatInterface works better when created in context)
                        gr.Markdown("### Chat with the agents")
                        
                        # Chat function for ChatInterface
                        def chat_fn(message, history, files, images, selected_model=None, use_open_deep_research=False):
                            """Handle chat messages with PDF reading and vision model auto-selection."""
                            global _global_manager_agent
                            agent = _global_manager_agent if _global_manager_agent else manager_agent
                            if not agent:
                                yield "❌ Agent not initialized. Please check Startup Checks tab."
                                return
                            
                            # Maximum input length (configurable via AGENT_MAX_INPUT_LENGTH env var)
                            MAX_TOTAL_INPUT = int(os.getenv("AGENT_MAX_INPUT_LENGTH", "50000"))
                            MAX_DOC_CONTENT = MAX_TOTAL_INPUT - len(message) - 2000  # Reserve 2000 chars for metadata/paths
                            
                            # If Open Deep Research is requested, add it to agent
                            if use_open_deep_research and OPEN_DEEP_RESEARCH_AVAILABLE:
                                has_odr = False
                                if hasattr(agent, 'managed_agents'):
                                    for ma in agent.managed_agents:
                                        if hasattr(ma, 'name') and ma.name == 'search_agent':
                                            has_odr = True
                                            break
                                
                                if not has_odr:
                                    try:
                                        odr_agent = create_open_deep_research_agent(agent.model)
                                        if odr_agent:
                                            agent.managed_agents.append(odr_agent)
                                            _global_manager_agent = agent
                                    except Exception as e:
                                        yield f"⚠️ Could not enable Open Deep Research: {str(e)}\n\n"
                            
                            # Handle model change
                            # Convert empty string to None for selected_model
                            if selected_model == "":
                                selected_model = None
                            if selected_model and _global_startup_result and _global_startup_result.ollama["available"]:
                                try:
                                    new_model = create_model_from_name(selected_model, _global_startup_result.ollama.get("url", "http://localhost:11434"), _global_startup_config)
                                    agent.model = new_model
                                    _global_manager_agent = agent
                                except Exception as e:
                                    yield f"Error switching model: {str(e)}\n\n"
                            
                            # Prepare task with file content reading
                            task = message
                            file_contents = []
                            file_summaries = []
                            
                            # Process uploaded files - read PDF/document content
                            if files:
                                saved_paths = []
                                total_content_chars = 0
                                
                                for file in files:
                                    if file:
                                        import tempfile
                                        import shutil
                                        temp_path = tempfile.mktemp(suffix=os.path.splitext(file.name)[1])
                                        shutil.copy(file.name, temp_path)
                                        saved_paths.append(temp_path)
                                        
                                        # Read document content
                                        ext = os.path.splitext(file.name)[1].lower()
                                        if ext in [".pdf", ".docx", ".txt"]:
                                            try:
                                                content = _global_document_reader.forward(temp_path)
                                                if content and not content.startswith("Error"):
                                                    original_len = len(content)
                                                    remaining_space = MAX_DOC_CONTENT - total_content_chars
                                                    
                                                    if remaining_space <= 0:
                                                        file_summaries.append(f"- {os.path.basename(file.name)}: [Skipped - input limit reached, {original_len} chars]")
                                                        continue
                                                    
                                                    # Smart truncation
                                                    if len(content) > remaining_space:
                                                        first_part_len = int(remaining_space * 0.7)
                                                        last_part_len = int(remaining_space * 0.2)
                                                        first_part = content[:first_part_len]
                                                        last_part = content[-last_part_len:] if last_part_len > 0 else ""
                                                        content = (
                                                            first_part + 
                                                            f"\n\n... [MIDDLE SECTION OMITTED - {original_len - first_part_len - last_part_len} chars] ...\n\n" +
                                                            last_part
                                                        )
                                                        file_summaries.append(f"- {os.path.basename(file.name)}: {original_len} chars (truncated to fit)")
                                                    else:
                                                        file_summaries.append(f"- {os.path.basename(file.name)}: {original_len} chars (full content)")
                                                    
                                                    file_contents.append(f"\n\n--- Content of {os.path.basename(file.name)} ---\n{content}\n--- End of {os.path.basename(file.name)} ---")
                                                    total_content_chars += len(content)
                                            except Exception as e:
                                                file_summaries.append(f"- {os.path.basename(file.name)}: [Error: {str(e)}]")
                                
                                if saved_paths:
                                    task += f"\n\nFiles provided: {', '.join(saved_paths)}"
                                if file_summaries:
                                    task += f"\n\n📄 **Document Status:**\n" + "\n".join(file_summaries)
                                if file_contents:
                                    task += "\n".join(file_contents)
                            
                            # Prepare images and auto-select vision model if needed
                            task_images = None
                            vision_agent = None
                            
                            if images:
                                from PIL import Image
                                task_images = []
                                for img in images:
                                    if isinstance(img, Image.Image):
                                        task_images.append(img)
                            
                                # Use vision model when images are provided
                                if task_images and _global_vision_models:
                                    vision_model_name = _global_vision_models[0]
                                    yield f"📸 Image detected! Creating vision agent with **{vision_model_name}**...\n\n"
                                    
                                    try:
                                        base_url = _global_startup_result.ollama.get("url", "http://localhost:11434") if _global_startup_result else "http://localhost:11434"
                                        vision_model = create_model_from_name(vision_model_name, base_url, _global_startup_config)
                                        
                                        vision_agent = CodeAgent(
                                            tools=[],
                                            model=vision_model,
                                            max_steps=5,
                                            verbosity_level=1,
                                            import_risk_tolerance="medium",
                                            instructions="You are a vision assistant. Analyze images and describe what you see in detail. Extract any text, data, or relevant information from images.",
                                        )
                                        yield f"✅ Vision agent ready with **{vision_model_name}**\n\nAnalyzing image...\n\n"
                                    except Exception as e:
                                        yield f"⚠️ Could not create vision agent: {e}\n\nFalling back to text-only processing...\n\n"
                                        vision_agent = None
                            
                            # Stream response - use vision agent for images if available
                            response = ""
                            try:
                                # If we have images and a vision agent, use it first
                                if task_images and vision_agent:
                                    yield "🔍 **Vision Analysis (Qwen VL):**\n\n"
                                    try:
                                        vision_result = vision_agent.run(task, images=task_images)
                                        vision_response = str(vision_result) if vision_result else "No analysis available"
                                        response = f"**Image Analysis:**\n{vision_response}\n\n---\n\n"
                                        yield response
                                        
                                        # Pass vision analysis to main agent
                                        task = f"{message}\n\n**Image Analysis from Vision Model:**\n{vision_response}"
                                        task_images = None  # Don't pass images again
                                    except Exception as e:
                                        response = f"⚠️ Vision analysis error: {e}\n\n"
                                        yield response
                                
                                # Now run the main agent
                                if agent:
                                    yield response + "🤖 **Processing with DeepSeek (via Manager):**\n\n"
                                    max_steps = getattr(agent, "max_steps", None)
                                    for chunk in stream_to_gradio(
                                        agent=agent,
                                        task=task,
                                        task_images=task_images,
                                        reset_agent_memory=False,
                                        max_steps=max_steps,
                                    ):
                                        response, chunk_output = process_stream_chunk(chunk, response)
                                        yield chunk_output
                            except Exception as e:
                                error_msg = f"❌ **Error:** {str(e)}\n\n```python\n{traceback.format_exc()}\n```"
                                yield error_msg
                        
                        # Additional inputs - ORDER MUST MATCH chat_fn parameters!
                        # chat_fn signature: (message, history, files, images, selected_model, use_open_deep_research)
                        additional_inputs = []
                        
                        file_upload = gr.File(
                            label="Upload files",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".xlsx", ".pptx"],
                        )
                        additional_inputs.append(file_upload)
                        
                        image_upload = gr.Image(
                            label="Upload images",
                            type="pil",
                            sources=["upload"],
                            height=200,
                        )
                        additional_inputs.append(image_upload)
                        
                        # Model selection if Ollama available (must come before use_open_deep_research)
                        if _global_startup_result and _global_startup_result.ollama["available"] and _global_startup_result.ollama["models"]:
                            model_dropdown = gr.Dropdown(
                                label="Select Ollama Model",
                                choices=_global_startup_result.ollama["models"],
                                value=_global_startup_result.ollama["models"][0] if _global_startup_result.ollama["models"] else None,
                                interactive=True,
                            )
                            additional_inputs.append(model_dropdown)
                        else:
                            # Add hidden dummy component if no model dropdown (Gradio doesn't accept None)
                            model_dropdown = gr.Textbox(value="", visible=False)
                            additional_inputs.append(model_dropdown)
                        
                        # Open Deep Research checkbox (must be last to match function signature)
                        if OPEN_DEEP_RESEARCH_AVAILABLE:
                            use_odr = gr.Checkbox(
                                label="Use Open Deep Research",
                                value=False,
                                info="Enable web browsing and research capabilities",
                            )
                            additional_inputs.append(use_odr)
                        else:
                            # Add hidden dummy component if ODR not available (Gradio doesn't accept None)
                            use_odr = gr.Checkbox(value=False, visible=False)
                            additional_inputs.append(use_odr)
                        
                        # Create ChatInterface
                        # When using additional_inputs, examples must be lists of lists
                        # Each example is [message, file1, file2, ..., image1, image2, ..., model]
                        examples_list = []
                        if additional_inputs:
                            # Simple examples without additional inputs
                            examples_list = [
                                ["Search PubMed for microsampling publications"] + [None] * len(additional_inputs),
                                ["Find researchers working on microsampling"] + [None] * len(additional_inputs),
                                ["Generate a researcher register"] + [None] * len(additional_inputs),
                            ]
                        else:
                            examples_list = [
                                "Search PubMed for microsampling publications",
                                "Find researchers working on microsampling",
                                "Generate a researcher register",
                            ]
                        
                        chat_interface = gr.ChatInterface(
                            fn=chat_fn,
                            title="",
                            description="Chat with the multi-agent system",
                            examples=examples_list if additional_inputs else examples_list,
                            additional_inputs=additional_inputs if additional_inputs else None,
                        )
                        # Note: ChatInterface renders automatically, no need to call .render()
                    else:
                        gr.Markdown("Agents will be available after initialization.")
            
            # Monitoring Tab
            with gr.Tab("📊 Monitoring"):
                create_monitoring_tab()
            
            # Agent Functions Tab
            with gr.Tab("🎯 Agent Functions"):
                create_agent_functions_tab()
            
            # Settings Tab
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("## Model Selection")
                gr.Markdown("Choose which Ollama models to use for each agent role.")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Programming Agent Model")
                        programming_model_dropdown = gr.Dropdown(
                            choices=_global_available_models if _global_available_models else ["No models available"],
                            value=_global_current_programming_model if _global_current_programming_model else None,
                            label="Programming Model",
                            info="Model for code generation and execution"
                        )
                        current_prog_model = gr.Markdown(f"**Current:** {_global_current_programming_model or 'Not set'}")
                    
                    with gr.Column():
                        gr.Markdown("### Manager Agent Model")
                        manager_model_dropdown = gr.Dropdown(
                            choices=_global_available_models if _global_available_models else ["No models available"],
                            value=_global_current_manager_model if _global_current_manager_model else None,
                            label="Manager Model",
                            info="Model for task delegation and coordination"
                        )
                        current_mgr_model = gr.Markdown(f"**Current:** {_global_current_manager_model or 'Not set'}")
                
                gr.Markdown("### Vision Models (Auto-selected for images)")
                vision_info = gr.Markdown(
                    f"**Available vision models:** {', '.join(_global_vision_models) if _global_vision_models else 'None detected'}\n\n"
                    "Vision models are automatically used when you upload images."
                )
                
                apply_models_btn = gr.Button("🔄 Apply Model Changes", variant="primary")
                model_status = gr.Markdown()
                
                def apply_model_changes(prog_model, mgr_model):
                    """Apply new model selections."""
                    global _global_manager_agent, _global_programming_agent
                    global _global_current_programming_model, _global_current_manager_model
                    
                    if not _global_startup_result or not _global_startup_result.ollama.get("available", False):
                        return "❌ Ollama not available"
                    
                    try:
                        base_url = _global_startup_result.ollama.get("url", "http://localhost:11434")
                        
                        # Create new models
                        new_prog_model = create_model_from_name(prog_model, base_url, _global_startup_config)
                        new_mgr_model = create_model_from_name(mgr_model, base_url, _global_startup_config)
                        
                        # Recreate agents with new models
                        _global_programming_agent = create_programming_agent(
                            model=new_prog_model,
                            memory_backend=_global_memory_backend,
                            db_path=_global_db_path,
                            qdrant_collection_name="microsampling_publications"
                        )
                        
                        _global_manager_agent = create_manager_agent_gradio(
                            model=new_mgr_model,
                            programming_agent=_global_programming_agent,
                            memory_backend=_global_memory_backend,
                            startup_config=_global_startup_config,
                            include_open_deep_research=False,
                        )
                        
                        _global_current_programming_model = prog_model
                        _global_current_manager_model = mgr_model
                        
                        return f"✅ Models updated!\n- Programming: **{prog_model}**\n- Manager: **{mgr_model}**"
                    except Exception as e:
                        return f"❌ Error applying models: {str(e)}"
                
                apply_models_btn.click(
                    fn=apply_model_changes,
                    inputs=[programming_model_dropdown, manager_model_dropdown],
                    outputs=[model_status]
                )
                
                # Refresh models button
                refresh_models_btn = gr.Button("🔄 Refresh Available Models", variant="secondary")
                
                def refresh_models():
                    """Refresh the list of available models."""
                    global _global_available_models, _global_vision_models
                    if _global_startup_result and _global_startup_result.ollama.get("available", False):
                        base_url = _global_startup_result.ollama.get("url", "http://localhost:11434")
                        _global_available_models = get_available_ollama_models(base_url)
                        _global_vision_models = get_vision_capable_models(_global_available_models)
                        return (
                            gr.update(choices=_global_available_models),
                            gr.update(choices=_global_available_models),
                            f"**Available vision models:** {', '.join(_global_vision_models) if _global_vision_models else 'None detected'}\n\nVision models are automatically used when you upload images."
                        )
                    return (gr.update(), gr.update(), "❌ Ollama not available")
                
                refresh_models_btn.click(
                    fn=refresh_models,
                    outputs=[programming_model_dropdown, manager_model_dropdown, vision_info]
                )
            
            # Health Check Tab
            with gr.Tab("Health Check"):
                try:
                    from smolagents.health_check import HealthChecker
                    
                    health_output = gr.Markdown()
                    
                    def run_health_check():
                        """Run health check and display results."""
                        checker = HealthChecker()
                        report = checker.format_report()
                        return f"```\n{report}\n```"
                    
                    health_btn = gr.Button("Run Health Check", variant="primary")
                    health_btn.click(fn=run_health_check, outputs=[health_output])
                    
                    # Run on load
                    main_ui.load(fn=run_health_check, outputs=[health_output])
                except ImportError:
                    gr.Markdown("Health check system not available.")
    
    # Launch with theme
    # Try to find an available port if 7860 is busy
    import socket
    port = 7860
    for attempt in range(10):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        if result != 0:  # Port is available
            break
        port += 1
    else:
        port = 7860  # Fallback to default if no port found
    
    print("=" * 80)
    print(f"[INFO] Starting Gradio server on port {port}")
    print(f"[INFO] Access the UI at: http://localhost:{port}")
    print("=" * 80)
    print("[DEBUG] About to call launch()...")
    import sys
    sys.stdout.flush()
    try:
        main_ui.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            theme=gr.themes.Soft(),
            prevent_thread_lock=False,  # Ensure the server blocks
        )
        print("[DEBUG] launch() returned - this means server stopped")
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"[ERROR] launch() raised exception: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    # Keep the server running
    print("[INFO] Server should be running. Press Ctrl+C to stop.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Shutting down...")


if __name__ == "__main__":
    main()

