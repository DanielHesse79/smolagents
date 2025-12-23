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
import warnings
from typing import Optional, Generator, Dict, Any

# Suppress RuntimeWarning about tracemalloc
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    import io
    # Set UTF-8 encoding for stdout/stderr
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    # Set environment variables for subprocesses and Rich
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Prevent Rich from detecting Windows terminal
    os.environ['NO_COLOR'] = '1'
    os.environ['TERM'] = 'dumb'
    # Disable Rich's Windows terminal detection
    os.environ['FORCE_COLOR'] = '0'
    
    # Globally patch Rich's Windows rendering to prevent Unicode errors
    # This must be done before any Rich Console objects are created
    try:
        from rich import _windows_renderer
        original_legacy_windows_render = _windows_renderer.legacy_windows_render
        
        def safe_legacy_windows_render(buffer, term):
            # Just skip Windows rendering completely - we don't need it for Gradio
            # This prevents UnicodeEncodeError when Rich tries to write special characters
            pass
        
        _windows_renderer.legacy_windows_render = safe_legacy_windows_render
    except (ImportError, AttributeError):
        # If we can't patch it, that's okay - we'll handle it in the logger
        pass

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
    create_unicode_safe_logger,
    setup_api_models,
    create_programming_agent,
    create_unicode_safe_logger,
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
    from examples.shared_agent_utils import create_unicode_safe_logger

    # Ensure the agent uses our Unicode-safe logger on Windows to avoid Rich cp1252 errors
    try:
        if agent and getattr(agent, "logger", None):
            agent.logger = create_unicode_safe_logger(getattr(agent.logger, "level", 1))
    except Exception:
        pass
    
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
        
        # Only show required models section if there are any required models
        if ollama_status.get("required_models"):
            lines.append("\n**Required Models:**")
            for model, found in ollama_status["required_models"].items():
                status = "✅" if found else "❌"
                lines.append(f"{status} {model}")
                if not found:
                    lines.append(f"  Install with: `ollama pull {model}`")
        else:
            lines.append("\n💡 **Auto-selection enabled:** System will automatically choose the best models from available models.")
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
    """Extract metrics from agent and its managed agents."""
    metrics = {
        "total_steps": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_duration": 0.0,
        "steps": [],
        "agent_breakdown": {},
    }
    
    def collect_from_agent(agent, agent_name="main"):
        """Collect metrics from a single agent."""
        agent_metrics = {
            "steps": 0,
            "tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration": 0.0,
        }
        
        if hasattr(agent, "memory") and agent.memory:
            steps = agent.memory.steps
            agent_metrics["steps"] = len(steps)
            
            for step in steps:
                if hasattr(step, "token_usage") and step.token_usage:
                    tokens = step.token_usage.total_tokens or 0
                    input_tokens = step.token_usage.input_tokens or 0
                    output_tokens = step.token_usage.output_tokens or 0
                    
                    agent_metrics["tokens"] += tokens
                    agent_metrics["input_tokens"] += input_tokens
                    agent_metrics["output_tokens"] += output_tokens
                    
                    metrics["input_tokens"] += input_tokens
                    metrics["output_tokens"] += output_tokens
                    metrics["total_tokens"] += tokens
                
                if hasattr(step, "timing") and step.timing:
                    duration = step.timing.duration or 0.0
                    agent_metrics["duration"] += duration
                    metrics["total_duration"] += duration
                    metrics["steps"].append({
                        "agent": agent_name,
                        "step_number": getattr(step, "step_number", 0),
                        "duration": duration,
                        "tokens": step.token_usage.total_tokens if hasattr(step, "token_usage") and step.token_usage else 0,
                    })
        
        metrics["agent_breakdown"][agent_name] = agent_metrics
        metrics["total_steps"] += agent_metrics["steps"]
    
    # Collect from main agent
    if agent:
        collect_from_agent(agent, "manager")
        
        # Also collect from managed agents (e.g., programming agent)
        if hasattr(agent, "managed_agents") and agent.managed_agents:
            for i, managed_agent in enumerate(agent.managed_agents):
                agent_name = getattr(managed_agent, "name", f"managed_{i}")
                if not agent_name or agent_name == "managed_0":
                    agent_name = "programming"
                collect_from_agent(managed_agent, agent_name)
    
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
    
    # Show breakdown by agent if available
    if metrics.get("agent_breakdown"):
        lines.append("")
        lines.append("### 📈 Per Agent Breakdown:")
        for agent_name, agent_metrics in metrics["agent_breakdown"].items():
            lines.append(f"")
            lines.append(f"**{agent_name.capitalize()} Agent:**")
            lines.append(f"  - Steps: {agent_metrics['steps']}")
            lines.append(f"  - Tokens: {agent_metrics['tokens']:,}")
            lines.append(f"  - Duration: {agent_metrics['duration']:.2f}s")
            if agent_metrics['duration'] > 0:
                agent_tps = agent_metrics['tokens'] / agent_metrics['duration']
                lines.append(f"  - Tokens/sec: {agent_tps:.1f}")
    
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
    
    # Create Unicode-safe loggers for Windows compatibility
    search_logger = create_unicode_safe_logger(verbosity_level=2)
    odr_manager_logger = create_unicode_safe_logger(verbosity_level=2)
    
    search_agent = ToolCallingAgent(
        model=model,
        tools=web_tools,
        max_steps=20,
        verbosity_level=2,
        logger=search_logger,  # Use custom logger with Unicode-safe console
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
        logger=odr_manager_logger,  # Use custom logger with Unicode-safe console
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
    
    # Create Unicode-safe logger for Windows compatibility
    logger = create_unicode_safe_logger(verbosity_level=1)
    
    manager_agent = CodeAgent(
        tools=manager_tools,
        model=model,
        managed_agents=managed_agents,
        verbosity_level=1,
        logger=logger,  # Use custom logger with Unicode-safe console
        planning_interval=3,
        stream_outputs=True,
        max_steps=20,
        memory_backend=memory_backend,
        enable_experience_retrieval=True,
        instructions="""You are an evidence-first OSINT and commercial intelligence agent. You build reproducible, source-backed dossiers and relationship maps around target companies. You delegate programming tasks to the 'programmer' agent.

DELEGATION:
- Provide clear task descriptions with expected inputs/outputs
- Delegate ALL programming tasks (code, data processing, file operations, database queries, report generation, fact extraction)
- Handle non-programming tasks yourself (web search, information gathering, webpage visits)

PRIMARY INPUTS (extract from user query or use defaults):
- TargetCompany: Extract company name from user query
- TargetCompanyWebsite: Extract from search results or use "unknown"
- Mode: COMPETITOR or PARTNER (extract from query, default to COMPETITOR if unclear)
- Timeframe: Extract from query or use "2019–present" as default
- Geography Focus: Extract from query or use "Global" as default
- Industry/Domain: Extract from query or infer from company context
- OutputDepth: light, standard, or deep (default to "deep")
- Hard Constraints: Only public sources, English language, no paywall bypass

NON-NEGOTIABLE RULES:
1) Evidence-first: Every non-trivial claim must include at least one direct source URL and a short snippet (10–30 words) that supports it. Use store_atomic_fact() for each fact.
2) Prefer primary sources: official site pages, press releases, annual reports, regulatory filings, investor decks, conference programs, publications, job ads, procurement portals.
3) No paywall bypass, no logins, no scraping private gated data, no doxxing. Do not infer private emails/phone numbers. Use only publicly posted contact channels.
4) Track freshness: record publish date (or "unknown") and retrieval date for each source.
5) Deduplicate: If the same press release is reposted, treat it as one source and note syndication.
6) If evidence conflicts, report the conflict explicitly, do not resolve it by guessing.

HIGH-LEVEL MISSION:
Build a source-backed dossier on TargetCompany, then build a relationship graph including:
- Customers (who buys from them, who they sell to)
- Partners (collaboration, distribution, alliances)
- Competitors (direct and adjacent)
- Suppliers/technology dependencies (platforms, key tooling, OEM links when evidenced)
- People (leadership, key technical/commercial figures, hiring signals)

TWO-TRACK INTENT (Mode):
A) If Mode = COMPETITOR
   Focus more on:
   - Who they already worked with (customers, partners, distributors, OEMs, consortiums)
   - Proof of traction: contracts, case studies, "trusted by", reference logos, deployments
   - Where they sell: geographies, verticals, segments
   - Differentiators: claims that show how they position against others ("vs", "alternative to", "replaces")
   - Weak points: complaints, negative reviews, compliance incidents (only if sourced)

B) If Mode = PARTNER
   Focus more on:
   - Who their competitors are (so we understand conflicts and positioning)
   - Who their potential customers are (segments they target, current gaps, buyer personas)
   - Partnership fit: where a collaboration makes sense (distribution gaps, complementary tech, channel needs)
   - Go-to-market signals: hiring for BD, partnerships, channel managers, distributors, regional expansion
   - Integration surface: APIs, workflows, sample logistics, manufacturing, regulatory readiness, QA systems

COMPANY & PRODUCT OSINT RESEARCH:
When users ask about companies or products, conduct COMPREHENSIVE deep research covering ALL of these areas:

1. COMPANY OVERVIEW:
   - Company name, legal name, founding date, headquarters location
   - Company size (employees, revenue if available)
   - Company history and milestones
   - Ownership structure, parent companies, subsidiaries
   - Key executives and leadership team (names, backgrounds, roles)
   - Board members and advisors

2. PRODUCTS & SERVICES:
   - Complete product portfolio with detailed descriptions
   - Product specifications, features, capabilities
   - Product versions, release dates, update history
   - Pricing information (if publicly available):
     * Base prices, subscription tiers, enterprise pricing
     * Pricing models (one-time, subscription, usage-based)
     * Discounts, promotions, special offers
   - Product use cases and applications
   - Target markets and customer segments

3. COMPETITIVE ANALYSIS:
   - Identify main competitors (direct and indirect)
   - For EACH competitor, research:
     * Their products/services
     * Pricing compared to target company
     * Market position and market share
     * Strengths and weaknesses
   - Competitive landscape overview
   - Market positioning of target company vs competitors

4. UNIQUE SELLING POINTS (USP):
   - What makes the company/products unique?
   - Key differentiators vs competitors
   - Competitive advantages
   - Innovation and technology advantages
   - Patents, proprietary technology, trade secrets (if mentioned)
   - Why customers choose them over competitors

5. CUSTOMER & MARKET INTELLIGENCE:
   - Who uses their products? (customer types, industries, use cases)
   - Customer testimonials and case studies
   - Customer reviews (positive and negative)
   - Market adoption and growth trends
   - Customer retention and satisfaction indicators
   - Why people buy their products (value propositions, pain points solved)

6. BUSINESS INTELLIGENCE:
   - Funding rounds, investors, valuation (if available)
   - Partnerships and collaborations
   - Strategic alliances
   - Distribution channels
   - Sales and marketing strategies
   - Recent news, press releases, announcements
   - Regulatory compliance and certifications

7. ACADEMIC & SCIENTIFIC CONTEXT (if relevant):
   - Scientific publications mentioning the company/products
   - Clinical studies, trials, validation studies
   - Research collaborations with academic institutions
   - Scientific validation and peer-reviewed evidence

SOURCE BASKETS (use as checklists):
- Official: website pages (solutions, products, customers), blog, news/press, documentation, careers
- Corporate: annual report, filings, investor decks, company registry (if public), certifications (ISO, CLIA, etc.)
- Market presence: conference programs, webinars, posters, slides, public talks
- Scientific: publications, posters, patents (if relevant)
- Hiring: job postings (own site + aggregators) with keywords that reveal capabilities/stack
- Ecosystem: distributors/resellers, marketplaces, partner pages, OEM references
- Independent: reputable media, analyst notes (only if publicly accessible), customer reviews (handle carefully)

SEARCH STRATEGY (run in passes):
Pass 1: Broad discovery
- "<TargetCompany> customers"
- "<TargetCompany> partner" OR "distribution" OR "reseller" OR "OEM"
- "<TargetCompany> case study" OR "success story"
- "<TargetCompany> vs" OR "alternative" OR "compared to"
- "<TargetCompany> conference" OR "webinar" OR "poster" OR "abstract"
- "<TargetCompany> site:<their domain> pdf" OR "filetype:pdf"

Pass 2: Relationship mining
- For each discovered partner/customer/competitor, run:
  - "<TargetCompany> AND <EntityName> partnership"
  - "<TargetCompany> AND <EntityName> customer"
  - "<EntityName> AND <TargetCompany> distributor"
  - "<TargetCompany> AND <EntityName> agreement" OR "contract" OR "pilot"

Pass 3: Competitive landscape
- Identify 10–30 competitors across direct, adjacent, and substitute categories.
- For each competitor: note differentiation claims with sources.

Pass 4: Hiring + signals
- Extract capabilities, tech stack, and GTM intent from job ads:
  - keywords: "partnerships", "channel", "KAM", "regulatory", "validation", "GxP", "ISO", "automation", "API", etc.

DATA EXTRACTION REQUIREMENTS:
For every extracted fact, create an "AtomicFact" using store_atomic_fact():
- subject (entity)
- predicate (relationship/attribute)
- object (value or other entity)
- qualifiers (date, region, product line, certainty notes) as JSON string
- evidence: {url, snippet (10-30 words), page/section locator if available, published_date}
- retrieved_at (today's date)
- confidence: 'high', 'medium', or 'low'
- target_company and mode

RELATIONSHIP GRAPH (required):
Create a network list with edges using store_relationship_edge():
- from_entity
- to_entity
- edge_type: {customer_of, partner_of, competitor_of, distributor_of, supplier_of, member_of, collaborated_with}
- strength: {weak, medium, strong} based on evidence quality
- evidence_urls: JSON array of 1–3 source URLs
- evidence_snippets: JSON array matching URLs
- target_company and mode

SCORING (lightweight but useful):
Compute a Mode-specific score 0–100 with a brief rationale:
- EvidenceStrength (0–25): # and quality of sources
- MarketTraction (0–25): customers, deployments, revenue signals (only if sourced)
- StrategicRelevance (0–25):
   - COMPETITOR: overlap with our ICP + traction in our target segments
   - PARTNER: complementarity + channel fit + low conflict risk
- ExecutionSignals (0–25): hiring, expansion, compliance readiness, delivery capability
Delegate to programmer to calculate using calculate_osint_score() function.

QUALITY BAR:
- If you cannot find evidence for a claim, write: "No public evidence found" and move on.
- Avoid vague language. Prefer specific nouns, numbers, and quotes (short snippets).
- Keep it reproducible: another analyst should be able to reach the same conclusions using your sources.

DELIVERABLES (final output):
Delegate to programmer to generate using generate_osint_report_dossier():
1) Executive Summary (10–15 bullet points, each with a source link)
2) Company Snapshot:
   - What they do, who they sell to, geographies, key offerings
3) Proof & Traction:
   - Customers/partners/distributors with evidence table
4) Competitive Landscape:
   - Top competitors and differentiation claims
5) Relationship Map:
   - Edge list + short narrative: "their ecosystem in 2 paragraphs"
6) Risks & Watchouts:
   - conflicts, compliance red flags, dependency risks, outdated sources
7) Appendix:
   - Source list (deduped), with publish dates and retrieval dates

OUTPUT FORMATS:
- A clean Markdown report (use generate_osint_report_dossier())
- Two CSV-style tables embedded in Markdown:
  A) Evidence Table: claim | confidence | source_url | snippet | published_date (use generate_evidence_table_markdown())
  B) Relationship Edges: from | to | edge_type | strength | source_url (use generate_relationship_edges_table())
- PDF version (use write_pdf_file() with the markdown content)
- JSON backup files (atomic_facts.json, relationship_edges.json) automatically created

OUTPUT FILES:
- When users ask for information about companies or products, ALWAYS:
  1. Extract facts using store_atomic_fact() for each piece of evidence
  2. Extract relationships using store_relationship_edge() for each connection
  3. Generate markdown report using generate_osint_report_dossier()
  4. Create PDF using write_pdf_file()
  5. Create markdown file using write_markdown_file()
- Mention the file paths in your final answer so users know where to find the reports

UNDERSTANDING USER INTENT:
- When users ask about companies or products, conduct FULL OSINT research covering all areas above
- Academic publications are SUPPLEMENTARY, not primary - focus on business intelligence first
- When users ask about database contents (e.g., "how many publications/researchers are stored"), understand they likely want you to:
  1. FIRST search online if the database might be empty
  2. Store the results in the database
  3. THEN query and report the counts
- Don't just query an empty database - be proactive about searching and populating data when needed

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

PUBLICATION MINING (SUPPLEMENTARY RESEARCH):
- Use pubmed_search() ONLY for scientific/clinical context when researching companies/products
- For company research, prioritize business intelligence, competitive analysis, and market research
- Academic publications are valuable for: clinical validation, scientific studies, research collaborations
- Use web_search() and visit_webpage() as PRIMARY research methods for company intelligence
- CRITICAL: pubmed_search() returns a STRING, not a list! You MUST parse it to extract the JSON data.
- The output contains [STRUCTURED_DATA]...[/STRUCTURED_DATA] tags with JSON inside.
- CORRECT USAGE PATTERN (for scientific context only):
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
  
  # Step 4: Create comprehensive reports
  # Format publications as markdown
  md_content = "# Swedish Microsampling Publications\\n\\n"
  for pub in swedish_pubs:
      md_content += f"## {pub.get('title', 'N/A')}\\n"
      md_content += f"**Authors:** {', '.join(pub.get('authors', []))}\\n"
      md_content += f"**Year:** {pub.get('year', 'N/A')}\\n"
      md_content += f"**DOI:** {pub.get('doi', 'N/A')}\\n\\n"
  
  # Write markdown and PDF files
  write_markdown_file("swedish_microsampling_pubs.md", md_content)
  write_pdf_file("swedish_microsampling_pubs.pdf", md_content, title="Swedish Microsampling Publications")
  
  final_answer(f"Found {len(swedish_pubs)} publications. Reports saved to: swedish_microsampling_pubs.md and swedish_microsampling_pubs.pdf")
  ```
- WRONG: Don't iterate over pubmed_search() result directly - it's a string, not a list!
- Delegate to programmer for: parsing JSON, filtering by affiliations, deduplication (use DOI/PMID/hash), storing (qdrant_upsert_publication, sql_upsert_publication), writing markdown and PDF files, researcher tracking (upsert_researcher, link_researcher_publication)
- ALWAYS create both markdown and PDF files when generating publication reports

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
    
    # Initialize with a helpful message
    initial_message = "📊 **Click 'Refresh All' to load metrics.**\n\n" + \
                     "💡 *Run a task in the Chat tab first, then refresh to see metrics.*"
    metrics_display = gr.Markdown(value=initial_message)
    refresh_btn = gr.Button("🔄 Refresh All", variant="primary")
    
    def refresh_metrics():
        agent = _global_manager_agent
        if not agent:
            return "❌ **No agent available.**\n\nPlease run a task in the Chat tab first to initialize the agents."
        
        try:
            metrics = get_agent_metrics(agent)
            
            # If no steps, show helpful message
            if metrics["total_steps"] == 0:
                return "📊 **No metrics yet.**\n\n" + \
                       "**Total Steps:** 0\n" + \
                       "**Total Tokens:** 0\n" + \
                       "**Total Duration:** 0.00s\n\n" + \
                       "💡 *Run a task in the Chat tab, then click Refresh to see metrics.*"
            
            return format_metrics_markdown(metrics)
        except Exception as e:
            import traceback
            return f"❌ **Error retrieving metrics:**\n\n```python\n{str(e)}\n\n{traceback.format_exc()}\n```"
    
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
            # Chat Tab (moved to first position to be default)
            with gr.Tab("Chat"):
                chat_container = gr.Column(visible=_global_manager_agent is not None)
                init_container = gr.Column(visible=_global_manager_agent is None)
                
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
                            # Try to use saved preferences
                            saved_prefs = load_model_preferences()
                            prog_model = saved_prefs.get("programming_model")
                            mgr_model = saved_prefs.get("manager_model")
                            
                            # Validate saved models are still available
                            available_models = result.ollama.get("models", [])
                            if prog_model not in available_models:
                                prog_model = None
                            if mgr_model not in available_models:
                                mgr_model = None
                            
                            programming_model, manager_model = setup_ollama_models(
                                result.ollama, 
                                cfg,
                                programming_model_name=prog_model,
                                manager_model_name=mgr_model
                            )
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
                    if _global_manager_agent and _global_programming_agent:
                        # Create chat interface inline (ChatInterface works better when created in context)
                        gr.Markdown("### Chat with the agents")
                        
                        # Chat function for ChatInterface
                        def chat_fn(message, history, files, images, selected_model=None, osint_mode="COMPETITOR", osint_timeframe="2019–present", osint_geography="Global", osint_industry="", osint_depth="deep", osint_constraints="Only public sources, English language", use_open_deep_research=False):
                            """Handle chat messages with PDF reading and vision model auto-selection."""
                            try:
                                global _global_manager_agent
                                agent = _global_manager_agent
                                if not agent:
                                    yield "❌ Agent not initialized. Please check Startup Checks tab or Model Selection tab to initialize agents."
                                    return
                                
                                # Validate message
                                if not message or not message.strip():
                                    yield "❌ Please provide a message."
                                    return
                                
                                # Enhance message with OSINT parameters if it's a company research query
                                # Detect if this is a company research query (contains company name, "research", "OSINT", etc.)
                                company_research_keywords = ["company", "research", "osint", "dossier", "competitor", "partner", "analyze", "intelligence"]
                                is_company_research = any(keyword in message.lower() for keyword in company_research_keywords)
                                
                                if is_company_research:
                                    osint_context = f"\n\n[OSINT Research Parameters]\nMode: {osint_mode}\nTimeframe: {osint_timeframe}\nGeography: {osint_geography}\nIndustry: {osint_industry or 'Not specified'}\nDepth: {osint_depth}\nConstraints: {osint_constraints}\n\nPlease conduct evidence-first OSINT research following the specified parameters."
                                    message = message + osint_context
                                
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
                                    # Handle both single Image object and list of images
                                    if isinstance(images, Image.Image):
                                        # Single image - wrap in list
                                        task_images = [images]
                                    elif isinstance(images, (list, tuple)):
                                        # List of images - process each one
                                        for img in images:
                                            if isinstance(img, Image.Image):
                                                task_images.append(img)
                                    # If task_images is still empty, images might be in a different format
                                    if not task_images and images:
                                        # Try to convert if it's a file path or other format
                                        try:
                                            if isinstance(images, str):
                                                task_images = [Image.open(images)]
                                            elif hasattr(images, '__iter__') and not isinstance(images, (str, bytes)):
                                                # It's iterable but not an Image - try to process
                                                for item in images:
                                                    if isinstance(item, Image.Image):
                                                        task_images.append(item)
                                                    elif isinstance(item, str):
                                                        task_images.append(Image.open(item))
                                        except Exception:
                                            # If conversion fails, just skip images
                                            task_images = None
                                
                                    # Use vision model when images are provided
                                    if task_images and _global_vision_models:
                                        vision_model_name = _global_vision_models[0]
                                        yield f"📸 Image detected! Creating vision agent with **{vision_model_name}**...\n\n"
                                        
                                        try:
                                            base_url = _global_startup_result.ollama.get("url", "http://localhost:11434") if _global_startup_result else "http://localhost:11434"
                                            vision_model = create_model_from_name(vision_model_name, base_url, _global_startup_config)
                                            
                                            # Create Unicode-safe logger for Windows compatibility
                                            vision_logger = create_unicode_safe_logger(verbosity_level=1)
                                            
                                            vision_agent = CodeAgent(
                                                tools=[],
                                                model=vision_model,
                                                max_steps=5,
                                                verbosity_level=1,
                                                logger=vision_logger,  # Use custom logger with Unicode-safe console
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
                                            # Ensure vision agent uses Unicode-safe logger
                                            if getattr(vision_agent, "logger", None):
                                                vision_agent.logger = create_unicode_safe_logger(getattr(vision_agent.logger, "level", 1))

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
                                        # Get the actual model name being used
                                        model_display_name = selected_model if selected_model else _global_current_manager_model or "Manager"
                                        yield response + f"🤖 **Processing with {model_display_name}:**\n\n"
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
                                    import traceback as tb
                                    error_msg = f"❌ **Error:** {str(e)}\n\n```python\n{tb.format_exc()}\n```"
                                    yield error_msg
                                    import sys
                                    print(f"[ERROR] Chat function error: {e}", file=sys.stderr)
                                    tb.print_exc()
                            except Exception as e:
                                # Catch any errors at the top level
                                import traceback as tb
                                import sys
                                error_msg = f"❌ **Unexpected Error:** {str(e)}\n\nPlease check the terminal for more details."
                                yield error_msg
                                print(f"[ERROR] Chat function top-level error: {e}", file=sys.stderr)
                                tb.print_exc()
                        
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
                        
                        # OSINT Research Mode Selection
                        osint_mode = gr.Dropdown(
                            label="OSINT Research Mode",
                            choices=["COMPETITOR", "PARTNER"],
                            value="COMPETITOR",
                            interactive=True,
                            info="COMPETITOR: Focus on customers, traction, differentiators. PARTNER: Focus on partnership fit, integration, GTM signals."
                        )
                        additional_inputs.append(osint_mode)
                        
                        # Optional OSINT parameters (collapsible)
                        with gr.Accordion("Advanced OSINT Parameters (Optional)", open=False):
                            osint_timeframe = gr.Textbox(
                                label="Timeframe",
                                value="2019–present",
                                placeholder="e.g., 2019–present, 2020–2024",
                                info="Time period for research focus"
                            )
                            osint_geography = gr.Textbox(
                                label="Geography Focus",
                                value="Global",
                                placeholder="e.g., EU + US, North America, Europe",
                                info="Geographic regions to focus on"
                            )
                            osint_industry = gr.Textbox(
                                label="Industry/Domain",
                                value="",
                                placeholder="e.g., life science tools, diagnostics, CRO, biotech, SaaS",
                                info="Industry or domain context"
                            )
                            osint_depth = gr.Dropdown(
                                label="Output Depth",
                                choices=["light", "standard", "deep"],
                                value="deep",
                                info="Research depth: light (quick), standard (balanced), deep (comprehensive)"
                            )
                            osint_constraints = gr.Textbox(
                                label="Hard Constraints",
                                value="Only public sources, English language",
                                placeholder="e.g., only public sources, English + Swedish, exclude certain markets",
                                info="Constraints for research"
                            )
                        
                        # Model selection if Ollama available (must come before use_open_deep_research)
                        if _global_startup_result and _global_startup_result.ollama["available"] and _global_startup_result.ollama["models"]:
                            # Default to current manager model if set, otherwise first available
                            default_model = None
                            if _global_current_manager_model and _global_current_manager_model in _global_startup_result.ollama["models"]:
                                default_model = _global_current_manager_model
                            elif _global_startup_result.ollama["models"]:
                                default_model = _global_startup_result.ollama["models"][0]
                            model_dropdown = gr.Dropdown(
                                label="Select Ollama Model",
                                choices=_global_startup_result.ollama["models"],
                                value=default_model,
                                interactive=True,
                                info="Changes model for this chat session"
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
                        return "❌ Ollama not available", f"**Current:** {_global_current_programming_model or 'Not set'}", f"**Current:** {_global_current_manager_model or 'Not set'}"
                    
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
                        
                        # Save model preferences
                        save_model_preferences(prog_model, mgr_model)
                        
                        return (
                            f"✅ Models updated!\n- Programming: **{prog_model}**\n- Manager: **{mgr_model}**",
                            f"**Current:** {prog_model}",
                            f"**Current:** {mgr_model}"
                        )
                    except Exception as e:
                        import traceback
                        return (
                            f"❌ Error applying models: {str(e)}\n\n```\n{traceback.format_exc()}\n```",
                            f"**Current:** {_global_current_programming_model or 'Not set'}",
                            f"**Current:** {_global_current_manager_model or 'Not set'}"
                        )
                
                apply_models_btn.click(
                    fn=apply_model_changes,
                    inputs=[programming_model_dropdown, manager_model_dropdown],
                    outputs=[model_status, current_prog_model, current_mgr_model]
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
                health_output = gr.Markdown()
                
                def run_health_check():
                    """Run health check and display results."""
                    config = StartupConfig()
                    result = run_startup_checks(config)
                    
                    # Format comprehensive health report
                    report_lines = []
                    report_lines.append("# 🏥 System Health Check Report\n")
                    report_lines.append(f"**Overall Status:** {'✅ All systems operational' if result.all_critical_services_ready else '⚠️ Some issues detected'}\n")
                    
                    # Ollama Status
                    report_lines.append("\n## 🦙 Ollama")
                    if result.ollama["available"]:
                        report_lines.append("✅ **Status:** Running")
                        if result.ollama.get("version"):
                            report_lines.append(f"**Version:** {result.ollama['version']}")
                        report_lines.append(f"**Models Available:** {len(result.ollama['models'])}")
                        if result.ollama.get("gpu_available"):
                            report_lines.append(f"**GPU:** ✅ {result.ollama.get('gpu_info', 'Available')}")
                        else:
                            report_lines.append("**GPU:** ⚠️ Not detected")
                    else:
                        report_lines.append("❌ **Status:** Not available")
                        if result.ollama.get("error"):
                            report_lines.append(f"**Error:** {result.ollama['error']}")
                    
                    # Qdrant Status
                    report_lines.append("\n## 🗄️ Qdrant")
                    if result.qdrant["available"]:
                        report_lines.append("✅ **Status:** Running")
                        report_lines.append(f"**URL:** {result.qdrant.get('url', 'N/A')}:{result.qdrant.get('port', 'N/A')}")
                        report_lines.append(f"**Collections:** {len([c for c, found in result.qdrant.get('required_collections', {}).items() if found])} available")
                    else:
                        report_lines.append("❌ **Status:** Not available")
                        if result.qdrant.get("error"):
                            report_lines.append(f"**Error:** {result.qdrant['error']}")
                    
                    # SQLite Status
                    report_lines.append("\n## 💾 SQLite")
                    if result.sqlite["available"]:
                        report_lines.append("✅ **Status:** Running")
                        report_lines.append(f"**Database:** {result.sqlite.get('db_path', 'N/A')}")
                        report_lines.append(f"**Tables:** {len([t for t, found in result.sqlite.get('required_tables', {}).items() if found])} available")
                    else:
                        report_lines.append("❌ **Status:** Not available")
                        if result.sqlite.get("error"):
                            report_lines.append(f"**Error:** {result.sqlite['error']}")
                    
                    # Warnings and Errors
                    if result.warnings:
                        report_lines.append("\n## ⚠️ Warnings")
                        for warning in result.warnings:
                            report_lines.append(f"- {warning}")
                    
                    if result.errors:
                        report_lines.append("\n## ❌ Errors")
                        for error in result.errors:
                            report_lines.append(f"- {error}")
                    
                    if not result.warnings and not result.errors:
                        report_lines.append("\n## ✅ No Issues")
                        report_lines.append("All systems are operating normally.")
                    
                    return "\n".join(report_lines)
                
                health_btn = gr.Button("🔄 Run Health Check", variant="primary")
                health_btn.click(fn=run_health_check, outputs=[health_output])
                
                # Run on load
                main_ui.load(fn=run_health_check, outputs=[health_output])
    
    # Launch with theme
    # Use environment variable for port if set, otherwise try to find an available port
    import socket
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    # Only try to find available port if not explicitly set via environment variable
    if "GRADIO_SERVER_PORT" not in os.environ:
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

