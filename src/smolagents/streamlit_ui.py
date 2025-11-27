#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import re
import shutil
from pathlib import Path
from typing import Generator

from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import CodeAgent, MultiStepAgent, PlanningStep, RunResult
from smolagents.memory import ActionStep, FinalAnswerStep
from smolagents.models import ChatMessageStreamDelta, MessageRole, agglomerate_stream_deltas
from smolagents.utils import _is_package_available


def get_step_footnote_content(step_log: ActionStep | PlanningStep, step_name: str) -> str:
    """Get a footnote string for a step log with duration and token information"""
    step_footnote = f"**{step_name}**"
    if step_log.token_usage is not None:
        step_footnote += f" | Input tokens: {step_log.token_usage.input_tokens:,} | Output tokens: {step_log.token_usage.output_tokens:,}"
    step_footnote += f" | Duration: {round(float(step_log.timing.duration), 2)}s" if step_log.timing.duration else ""
    return step_footnote


def _clean_model_output(model_output: str) -> str:
    """
    Clean up model output by removing trailing tags and extra backticks.

    Args:
        model_output (`str`): Raw model output.

    Returns:
        `str`: Cleaned model output.
    """
    if not model_output:
        return ""
    model_output = model_output.strip()
    # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
    model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
    model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
    model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
    return model_output.strip()


def _format_code_content(content: str) -> str:
    """
    Format code content as Python code block if it's not already formatted.

    Args:
        content (`str`): Code content to format.

    Returns:
        `str`: Code content formatted as a Python code block.
    """
    content = content.strip()
    # Remove existing code blocks and end_code tags
    content = re.sub(r"```.*?\n", "", content)
    content = re.sub(r"\s*<end_code>\s*", "", content)
    content = content.strip()
    # Add Python code block formatting if not already present
    if not content.startswith("```python"):
        content = f"```python\n{content}\n```"
    return content


def _process_action_step(step_log: ActionStep, skip_model_outputs: bool = False) -> Generator:
    """
    Process an [`ActionStep`] and yield appropriate content for Streamlit display.

    Args:
        step_log ([`ActionStep`]): ActionStep to process.
        skip_model_outputs (`bool`): Whether to skip model outputs.

    Yields:
        `dict`: Dictionary with type and content for Streamlit rendering.
    """
    # Output the step number
    step_number = f"Step {step_log.step_number}"
    if not skip_model_outputs:
        yield {"type": "markdown", "content": f"**{step_number}**"}

    # First yield the thought/reasoning from the LLM
    if not skip_model_outputs and getattr(step_log, "model_output", ""):
        model_output = _clean_model_output(step_log.model_output)
        yield {"type": "markdown", "content": model_output}

    # For tool calls, create a tool call message
    if getattr(step_log, "tool_calls", []):
        first_tool_call = step_log.tool_calls[0]
        used_code = first_tool_call.name == "python_interpreter"

        # Process arguments based on type
        args = first_tool_call.arguments
        if isinstance(args, dict):
            content = str(args.get("answer", str(args)))
        else:
            content = str(args).strip()

        # Format code content if needed
        if used_code:
            content = _format_code_content(content)

        # Create the tool call message
        yield {
            "type": "tool_call",
            "content": content,
            "tool_name": first_tool_call.name,
        }

    # Display execution logs if they exist
    if getattr(step_log, "observations", "") and step_log.observations.strip():
        log_content = step_log.observations.strip()
        if log_content:
            log_content = re.sub(r"^Execution logs:\s*", "", log_content)
            yield {"type": "code", "content": log_content, "language": "bash"}

    # Display any images in observations
    if getattr(step_log, "observations_images", []):
        for image in step_log.observations_images:
            yield {"type": "image", "content": image}

    # Handle errors
    if getattr(step_log, "error", None):
        yield {"type": "error", "content": str(step_log.error)}

    # Add step footnote and separator
    yield {"type": "markdown", "content": get_step_footnote_content(step_log, step_number)}
    yield {"type": "markdown", "content": "-----"}


def _process_planning_step(step_log: PlanningStep, skip_model_outputs: bool = False) -> Generator:
    """
    Process a [`PlanningStep`] and yield appropriate content for Streamlit display.

    Args:
        step_log ([`PlanningStep`]): PlanningStep to process.

    Yields:
        `dict`: Dictionary with type and content for Streamlit rendering.
    """
    if not skip_model_outputs:
        yield {"type": "markdown", "content": "**Planning step**"}
        yield {"type": "markdown", "content": step_log.plan}
    yield {"type": "markdown", "content": get_step_footnote_content(step_log, "Planning step")}
    yield {"type": "markdown", "content": "-----"}


def _process_final_answer_step(step_log: FinalAnswerStep) -> Generator:
    """
    Process a [`FinalAnswerStep`] and yield appropriate content for Streamlit display.

    Args:
        step_log ([`FinalAnswerStep`]): FinalAnswerStep to process.

    Yields:
        `dict`: Dictionary with type and content for Streamlit rendering.
    """
    final_answer = step_log.output
    if isinstance(final_answer, AgentText):
        yield {"type": "markdown", "content": f"**Final answer:**\n{final_answer.to_string()}\n"}
    elif isinstance(final_answer, AgentImage):
        yield {"type": "image", "content": final_answer.to_string()}
    elif isinstance(final_answer, AgentAudio):
        yield {"type": "audio", "content": final_answer.to_string()}
    else:
        yield {"type": "markdown", "content": f"**Final answer:** {str(final_answer)}"}


def pull_messages_from_step(step_log: ActionStep | PlanningStep | FinalAnswerStep, skip_model_outputs: bool = False):
    """Extract content dictionaries from agent steps for Streamlit rendering.

    Args:
        step_log: The step log to display.
        skip_model_outputs: If True, skip the model outputs when creating the content:
            This is used for instance when streaming model outputs have already been displayed.
    """
    if not _is_package_available("streamlit"):
        raise ModuleNotFoundError(
            "Please install 'streamlit' extra to use the StreamlitUI: `pip install 'smolagents[streamlit]'`"
        )
    if isinstance(step_log, ActionStep):
        yield from _process_action_step(step_log, skip_model_outputs)
    elif isinstance(step_log, PlanningStep):
        yield from _process_planning_step(step_log, skip_model_outputs)
    elif isinstance(step_log, FinalAnswerStep):
        yield from _process_final_answer_step(step_log)
    else:
        raise ValueError(f"Unsupported step type: {type(step_log)}")


def stream_to_streamlit(
    agent,
    task: str,
    task_images: list | None = None,
    reset_agent_memory: bool = False,
    additional_args: dict | None = None,
    max_steps: int | None = None,
) -> Generator:
    """Runs an agent with the given task and streams the messages from the agent for Streamlit display."""

    if not _is_package_available("streamlit"):
        raise ModuleNotFoundError(
            "Please install 'streamlit' extra to use the StreamlitUI: `pip install 'smolagents[streamlit]'`"
        )
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
            for message in pull_messages_from_step(
                event,
                # If we're streaming model outputs, no need to display them twice
                skip_model_outputs=getattr(agent, "stream_outputs", False),
            ):
                yield message
            accumulated_events = []
        elif isinstance(event, ChatMessageStreamDelta):
            accumulated_events.append(event)
            text = agglomerate_stream_deltas(accumulated_events).render_as_markdown()
            yield {"type": "stream", "content": text}


class StreamlitUI:
    """
    Streamlit interface for interacting with a [`MultiStepAgent`].

    This class provides a web interface to interact with the agent in real-time, allowing users to submit prompts, upload files, and receive responses in a chat-like format.
    It can reset the agent's memory at the start of each interaction if desired.
    It supports file uploads, which are saved to a specified folder.
    It uses Streamlit's chat components to display the conversation history.
    This class requires the `streamlit` extra to be installed: `pip install 'smolagents[streamlit]'`.

    Args:
        agent ([`MultiStepAgent`]): The agent to interact with.
        file_upload_folder (`str`, *optional*): The folder where uploaded files will be saved.
            If not provided, file uploads are disabled.
        reset_agent_memory (`bool`, *optional*, defaults to `False`): Whether to reset the agent's memory at the start of each interaction.
            If `True`, the agent will not remember previous interactions.

    Raises:
        ModuleNotFoundError: If the `streamlit` extra is not installed.

    Example:
        ```python
        from smolagents import CodeAgent, StreamlitUI, InferenceClientModel

        model = InferenceClientModel(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
        agent = CodeAgent(tools=[], model=model)
        streamlit_ui = StreamlitUI(agent, file_upload_folder="uploads", reset_agent_memory=True)
        streamlit_ui.run()
        ```
    """

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None, reset_agent_memory: bool = False):
        if not _is_package_available("streamlit"):
            raise ModuleNotFoundError(
                "Please install 'streamlit' extra to use the StreamlitUI: `pip install 'smolagents[streamlit]'`"
            )
        self.agent = agent
        self.file_upload_folder = Path(file_upload_folder) if file_upload_folder is not None else None
        self.reset_agent_memory = reset_agent_memory
        self.name = getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        if self.file_upload_folder is not None:
            if not self.file_upload_folder.exists():
                self.file_upload_folder.mkdir(parents=True, exist_ok=True)

        # Initialize model registry with default model
        self.model_registry = {
            "Primary": agent.model,
        }
        # Add per-step models to registry if they differ
        if hasattr(agent, "planning_model") and agent.planning_model != agent.model:
            self.model_registry["Planning"] = agent.planning_model
        if hasattr(agent, "action_model") and agent.action_model != agent.model:
            self.model_registry["Action"] = agent.action_model
        if hasattr(agent, "final_answer_model") and agent.final_answer_model != agent.model:
            self.model_registry["Final Answer"] = agent.final_answer_model

    def upload_file(self, file, allowed_file_types=None):
        """
        Upload a file and add it to the list of uploaded files in the session state.

        The file is saved to the `self.file_upload_folder` folder.
        If the file type is not allowed, it returns a message indicating the disallowed file type.

        Args:
            file: The uploaded file from Streamlit.
            allowed_file_types (`list`, *optional*): List of allowed file extensions. Defaults to [".pdf", ".docx", ".txt"].
        """
        if file is None:
            return None

        if allowed_file_types is None:
            allowed_file_types = [".pdf", ".docx", ".txt"]

        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in allowed_file_types:
            return None

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        return file_path

    def run(self):
        """
        Run the Streamlit app with the agent interface.
        This should be called from a Streamlit script.
        """
        import streamlit as st
        from PIL import Image

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "file_uploads_log" not in st.session_state:
            st.session_state.file_uploads_log = []
        if "image_uploads" not in st.session_state:
            st.session_state.image_uploads = []
        if "agent_instance" not in st.session_state:
            st.session_state.agent_instance = self.agent
        if "max_steps_override" not in st.session_state:
            st.session_state.max_steps_override = None
        if "return_full_result" not in st.session_state:
            st.session_state.return_full_result = False
        if "additional_args" not in st.session_state:
            st.session_state.additional_args = {}
        if "show_config" not in st.session_state:
            st.session_state.show_config = False
        if "show_memory" not in st.session_state:
            st.session_state.show_memory = False
        if "model_registry" not in st.session_state:
            st.session_state.model_registry = self.model_registry.copy()
        if "agent_settings" not in st.session_state:
            st.session_state.agent_settings = {}
        if "tool_registry" not in st.session_state:
            st.session_state.tool_registry = {}
        if "executor_config" not in st.session_state:
            st.session_state.executor_config = {}
        if "prompt_templates_edited" not in st.session_state:
            st.session_state.prompt_templates_edited = False
        if "settings_saved" not in st.session_state:
            st.session_state.settings_saved = False

        # Sidebar
        with st.sidebar:
            # 1. Agent Info Section
            with st.expander("ℹ️ Agent Info", expanded=False):
                self._display_agent_config(st)
                st.markdown("> This web ui allows you to interact with a `smolagents` agent that can use tools and execute steps to complete tasks.")

            st.markdown("---")

            # 2. Settings Section (Main)
            with st.expander("⚙️ Settings", expanded=False):
                self._render_settings_section(st)

            st.markdown("---")

            # 3. Input Options Section
            st.markdown("**Input Options**")

            # Image upload support
            uploaded_images = st.file_uploader(
                "Upload images", type=["png", "jpg", "jpeg", "gif", "bmp", "webp"], accept_multiple_files=True
            )
            if uploaded_images:
                st.session_state.image_uploads = []
                for img_file in uploaded_images:
                    img = Image.open(img_file)
                    st.session_state.image_uploads.append(img)
                    st.image(img, caption=img_file.name, use_container_width=True)

            # File upload support
            if self.file_upload_folder is not None:
                uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
                if uploaded_file is not None:
                    file_path = self.upload_file(uploaded_file)
                    if file_path and file_path not in st.session_state.file_uploads_log:
                        st.session_state.file_uploads_log.append(file_path)
                        st.success(f"File uploaded: {os.path.basename(file_path)}")

            # Additional args input
            st.markdown("**Additional Arguments (JSON)**")
            additional_args_text = st.text_area(
                "Enter additional arguments as JSON",
                value=json.dumps(st.session_state.additional_args, indent=2) if st.session_state.additional_args else "{}",
                height=100,
                help="Pass custom variables to the agent (e.g., dataframes, images, etc.)",
            )
            try:
                if additional_args_text.strip():
                    st.session_state.additional_args = json.loads(additional_args_text)
                else:
                    st.session_state.additional_args = {}
            except json.JSONDecodeError as e:
                st.warning(f"Invalid JSON: {str(e)}. Using empty dict.")
                st.session_state.additional_args = {}

            st.markdown("---")

            # 4. Control Panel Section
            st.markdown("**Control Panel**")
            # Memory Management
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.agent_instance.memory.reset()
                    st.session_state.image_uploads = []
                    st.session_state.file_uploads_log = []
                    st.rerun()

            with col2:
                if st.button("Reset Memory", use_container_width=True):
                    st.session_state.agent_instance.memory.reset()
                    st.success("Memory reset!")

            # Agent Control
            col3, col4 = st.columns(2)
            with col3:
                if st.button("Interrupt Agent", use_container_width=True, type="secondary"):
                    if hasattr(st.session_state.agent_instance, "interrupt"):
                        st.session_state.agent_instance.interrupt()
                        st.warning("Agent interrupt signal sent!")
            with col4:
                if st.button("Cleanup", use_container_width=True, type="secondary"):
                    if hasattr(st.session_state.agent_instance, "cleanup"):
                        st.session_state.agent_instance.cleanup()
                        st.success("Agent cleaned up!")

            st.markdown("---")

            # 5. Monitoring Section
            st.markdown("**Monitoring**")
            with st.expander("📊 Token Usage & Metrics"):
                self._display_token_usage_summary(st)

            with st.expander("📝 Memory Inspector"):
                self._display_memory(st)
                
                # Replay option
                if st.button("Replay Memory", use_container_width=True):
                    from smolagents.monitoring import AgentLogger, LogLevel
                    logger = AgentLogger(level=LogLevel.INFO)
                    st.session_state.agent_instance.replay(detailed=False)
                    st.info("Check console/terminal for replay output")

            st.markdown(
                "<br><br><h4><center>Powered by <a target='_blank' href='https://github.com/huggingface/smolagents'><b>smolagents</b></a></center></h4>",
                unsafe_allow_html=True,
            )

        # Main chat interface
        st.title("Agent Chat")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    # Display user message content
                    if isinstance(message["content"], dict):
                        # Handle structured user message with images
                        if "text" in message["content"]:
                            st.write(message["content"]["text"])
                        if "images" in message["content"]:
                            for img in message["content"]["images"]:
                                st.image(img, use_container_width=True)
                    else:
                        st.write(message["content"])
                else:
                    # Assistant messages can have multiple content items
                    for content_item in message.get("content", []):
                        self._render_content(content_item)

        # Chat input at the bottom
        if prompt := st.chat_input("Enter your prompt here"):
            # Prepare user message
            user_message_content = prompt
            user_message_data = {"text": prompt}

            # Add file references
            if len(st.session_state.file_uploads_log) > 0:
                file_refs = "\n\nYou have been provided with these files, which might be helpful or not: " + str(
                    st.session_state.file_uploads_log
                )
                user_message_content += file_refs
                user_message_data["text"] += file_refs

            # Add images
            if st.session_state.image_uploads:
                user_message_data["images"] = st.session_state.image_uploads.copy()

            st.session_state.messages.append({"role": "user", "content": user_message_data})

            # Display user message
            with st.chat_message("user"):
                st.write(user_message_data["text"])
                if "images" in user_message_data:
                    for img in user_message_data["images"]:
                        st.image(img, use_container_width=True)

            # Process with agent
        with st.chat_message("assistant"):
                message_placeholder = st.empty()
                assistant_content = []
                current_stream_text = ""

                try:
                    # Prepare images for agent
                    task_images = st.session_state.image_uploads if st.session_state.image_uploads else None

                    # Prepare additional args
                    additional_args = st.session_state.additional_args if st.session_state.additional_args else None

                    # Get max steps
                    max_steps = (
                        st.session_state.max_steps_override
                        if st.session_state.max_steps_override != st.session_state.agent_instance.max_steps
                        else None
                    )

                    # Check if we should use non-streaming mode for return_full_result
                    if st.session_state.return_full_result:
                        # Use non-streaming mode to get RunResult
                        result = st.session_state.agent_instance.run(
                            task=prompt,
                            stream=False,
                            reset=self.reset_agent_memory,
                            images=task_images,
                            additional_args=additional_args,
                            max_steps=max_steps,
                            return_full_result=True,
                        )
                        # Display RunResult
                        self._display_run_result(st, result)
                        # Also display the output
                        if result.output:
                            if isinstance(result.output, (AgentText, AgentImage, AgentAudio)):
                                final_content = self._format_final_answer(result.output)
                                assistant_content.append(final_content)
                                self._render_content(final_content)
                            else:
                                assistant_content.append({"type": "markdown", "content": f"**Final answer:** {str(result.output)}"})
                                st.markdown(f"**Final answer:** {str(result.output)}")
                    else:
                        # Use streaming mode
                        for msg in stream_to_streamlit(
                            st.session_state.agent_instance,
                            task=prompt,
                            task_images=task_images,
                            reset_agent_memory=self.reset_agent_memory,
                            additional_args=additional_args,
                            max_steps=max_steps,
                        ):
                            if msg["type"] == "stream":
                                # Streaming text update
                                current_stream_text = msg["content"]
                                message_placeholder.markdown(current_stream_text)
                            else:
                                # Other content types
                                assistant_content.append(msg)
                                self._render_content(msg)

                    # Add final stream text if any
                    if current_stream_text:
                        assistant_content.append({"type": "markdown", "content": current_stream_text})

                    # Store complete assistant message
                    st.session_state.messages.append({"role": "assistant", "content": assistant_content})

                    # Clear image uploads after use
                    st.session_state.image_uploads = []

            except Exception as e:
                st.error(f"Error in interaction: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

    def _render_content(self, content_item: dict):
        """Render a content item in Streamlit."""
        import streamlit as st

        content_type = content_item.get("type", "markdown")
        content = content_item.get("content", "")

        if content_type == "markdown":
            st.markdown(content)
        elif content_type == "code":
            language = content_item.get("language", "python")
            st.code(content, language=language)
        elif content_type == "image":
            if isinstance(content, str):
                st.image(content)
            else:
                st.image(content)
        elif content_type == "audio":
            st.audio(content)
        elif content_type == "error":
            st.error(content)
        elif content_type == "tool_call":
            tool_name = content_item.get("tool_name", "tool")
            st.markdown(f"🛠️ **Used tool {tool_name}**")
            st.code(content, language="python" if tool_name == "python_interpreter" else None)
        elif content_type == "stream":
                st.markdown(content)

    def _display_agent_config(self, st):
        """Display agent configuration information."""
        agent = st.session_state.agent_instance

        # Model information
        st.markdown("**Primary Model**")
        model_info = f"Type: `{type(agent.model).__name__}`"
        if hasattr(agent.model, "model_id"):
            model_info += f"\nID: `{agent.model.model_id}`"
        if hasattr(agent.model, "provider") and agent.model.provider:
            model_info += f"\nProvider: `{agent.model.provider}`"
        st.markdown(model_info)
        
        # Per-step models if different
        if hasattr(agent, "planning_model") and agent.planning_model != agent.model:
            st.markdown(f"**Planning Model:** `{type(agent.planning_model).__name__}`")
        if hasattr(agent, "action_model") and agent.action_model != agent.model:
            st.markdown(f"**Action Model:** `{type(agent.action_model).__name__}`")
        if hasattr(agent, "final_answer_model") and agent.final_answer_model != agent.model:
            st.markdown(f"**Final Answer Model:** `{type(agent.final_answer_model).__name__}`")

        # Agent type
        st.markdown("**Agent Type**")
        st.markdown(f"`{type(agent).__name__}`")

        # Tools
        st.markdown("**Tools**")
        tool_names = list(agent.tools.keys())
        if tool_names:
            tools_text = ", ".join([f"`{name}`" for name in tool_names if name != "final_answer"])
            st.markdown(tools_text if tools_text else "None")
        else:
            st.markdown("None")

        # Managed agents
        if hasattr(agent, "managed_agents") and agent.managed_agents:
            st.markdown("**Managed Agents**")
            managed_agent_names = list(agent.managed_agents.keys())
            st.markdown(", ".join([f"`{name}`" for name in managed_agent_names]))

        # Configuration
        st.markdown("**Configuration**")
        config_info = f"Max Steps: `{agent.max_steps}`"
        if hasattr(agent, "planning_interval") and agent.planning_interval:
            config_info += f"\nPlanning Interval: `{agent.planning_interval}`"
        if hasattr(agent, "executor_type"):
            config_info += f"\nExecutor: `{agent.executor_type}`"
        if hasattr(agent, "stream_outputs"):
            config_info += f"\nStream Outputs: `{agent.stream_outputs}`"
        st.markdown(config_info)

    def _display_memory(self, st):
        """Display enhanced memory information with filtering and search."""
        agent = st.session_state.agent_instance
        memory = agent.memory

        st.markdown(f"**Total Steps:** {len(memory.steps)}")

        if memory.steps:
            # Filter options
            filter_type = st.selectbox(
                "Filter by Type",
                options=["All", "ActionStep", "PlanningStep", "TaskStep", "FinalAnswerStep"],
                key="memory_filter_type",
            )
            
            # Search
            search_query = st.text_input("Search Memory", key="memory_search", placeholder="Search in step content...")
            
            # Display step summary with enhanced info
            filtered_steps = memory.steps
            if filter_type != "All":
                filtered_steps = [s for s in filtered_steps if type(s).__name__ == filter_type]
            
            if search_query:
                filtered_steps = [
                    s for s in filtered_steps
                    if search_query.lower() in str(s).lower() or
                    (hasattr(s, "model_output") and search_query.lower() in str(s.model_output).lower()) or
                    (hasattr(s, "observations") and search_query.lower() in str(s.observations).lower())
                ]
            
            for i, step in enumerate(filtered_steps):
                with st.expander(f"{type(step).__name__} {getattr(step, 'step_number', i+1)}", expanded=False):
                    step_type = type(step).__name__
                    
                    if isinstance(step, ActionStep):
                        st.markdown(f"**Step Number:** {step.step_number}")
                        if step.tool_calls:
                            tool_names = [tc.name for tc in step.tool_calls]
                            st.markdown(f"**Tools Used:** {', '.join(tool_names)}")
                        if step.token_usage:
                            st.markdown(
                                f"**Tokens:** Input: {step.token_usage.input_tokens:,} | "
                                f"Output: {step.token_usage.output_tokens:,} | "
                                f"Total: {step.token_usage.total_tokens:,}"
                            )
                        if step.timing and step.timing.duration:
                            st.markdown(f"**Duration:** {round(step.timing.duration, 2)}s")
                        if step.model_output:
                            with st.expander("Model Output"):
                                st.text(step.model_output[:1000] + ("..." if len(step.model_output) > 1000 else ""))
                        if step.observations:
                            with st.expander("Observations"):
                                st.text(step.observations[:1000] + ("..." if len(step.observations) > 1000 else ""))
                        if step.error:
                            st.error(f"Error: {str(step.error)}")
                    
                    elif isinstance(step, PlanningStep):
                        st.markdown(f"**Plan:** {step.plan[:200]}...")
                        if step.token_usage:
                            st.markdown(
                                f"**Tokens:** Input: {step.token_usage.input_tokens:,} | "
                                f"Output: {step.token_usage.output_tokens:,}"
                            )
                        if step.timing and step.timing.duration:
                            st.markdown(f"**Duration:** {round(step.timing.duration, 2)}s")
                    
                    elif isinstance(step, FinalAnswerStep):
                        st.markdown(f"**Output:** {str(step.output)[:200]}...")
                    
                    else:
                        st.json(step.dict() if hasattr(step, "dict") else str(step))

            # Option to view full memory
            st.markdown("---")
            if st.button("View Full Memory (JSON)", key="view_full_memory_json"):
                memory_dict = memory.get_full_steps()
                st.json(memory_dict)
                st.download_button(
                    "Download Memory JSON",
                    data=json.dumps(memory_dict, indent=2, default=str),
                    file_name="agent_memory.json",
                    mime="application/json",
                    key="download_memory_json",
                )
        else:
            st.info("No steps in memory yet.")

    def _display_token_usage_summary(self, st):
        """Display enhanced token usage summary with per-step breakdown."""
        agent = st.session_state.agent_instance
        if hasattr(agent, "monitor"):
            monitor = agent.monitor
            token_usage = monitor.get_total_token_counts()
            if token_usage.total_tokens > 0:
                st.markdown("**Token Usage Summary**")
                st.markdown(
                    f"Input: `{token_usage.input_tokens:,}` | "
                    f"Output: `{token_usage.output_tokens:,}` | "
                    f"Total: `{token_usage.total_tokens:,}`"
                )
                
                # Per-step breakdown
                if st.checkbox("Show Per-Step Breakdown", key="show_token_breakdown"):
                    memory = agent.memory
                    step_data = []
                    for step in memory.steps:
                        if isinstance(step, (ActionStep, PlanningStep)) and step.token_usage:
                            step_type = type(step).__name__
                            step_num = getattr(step, "step_number", "N/A")
                            step_data.append({
                                "Step": f"{step_type} {step_num}",
                                "Input Tokens": step.token_usage.input_tokens,
                                "Output Tokens": step.token_usage.output_tokens,
                                "Total Tokens": step.token_usage.total_tokens,
                                "Duration (s)": round(step.timing.duration, 2) if step.timing and step.timing.duration else "N/A",
                            })
                    
                    if step_data:
                        try:
                            import pandas as pd
                            df = pd.DataFrame(step_data)
                            st.dataframe(df, use_container_width=True)
                        except ImportError:
                            # Fallback to table display if pandas not available
                            st.table(step_data)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_tokens = df["Total Tokens"].mean()
                            st.metric("Avg Tokens/Step", f"{avg_tokens:.0f}")
                        with col2:
                            total_duration = sum([d["Duration (s)"] for d in step_data if isinstance(d["Duration (s)"], (int, float))])
                            st.metric("Total Duration", f"{total_duration:.2f}s")
                        with col3:
                            if isinstance(total_duration, (int, float)) and total_duration > 0:
                                tokens_per_sec = token_usage.total_tokens / total_duration
                                st.metric("Tokens/sec", f"{tokens_per_sec:.0f}")
                
                # Timing information
                if hasattr(monitor, "step_durations") and monitor.step_durations:
                    st.markdown("**Timing Information**")
                    avg_duration = sum(monitor.step_durations) / len(monitor.step_durations)
                    max_duration = max(monitor.step_durations)
                    min_duration = min(monitor.step_durations)
                    st.markdown(
                        f"Avg: `{avg_duration:.2f}s` | "
                        f"Max: `{max_duration:.2f}s` | "
                        f"Min: `{min_duration:.2f}s`"
                    )

    def _format_final_answer(self, output):
        """Format final answer for display."""
        if isinstance(output, AgentText):
            return {"type": "markdown", "content": f"**Final answer:**\n{output.to_string()}\n"}
        elif isinstance(output, AgentImage):
            return {"type": "image", "content": output.to_string()}
        elif isinstance(output, AgentAudio):
            return {"type": "audio", "content": output.to_string()}
        else:
            return {"type": "markdown", "content": f"**Final answer:** {str(output)}"}

    def _display_run_result(self, st, run_result):
        """Display RunResult information."""
        if not isinstance(run_result, RunResult):
            return

        with st.expander("📊 Run Result Details", expanded=True):
            # State
            state_emoji = "✅" if run_result.state == "success" else "⚠️"
            st.markdown(f"**State:** {state_emoji} `{run_result.state}`")

            # Timing
            if run_result.timing:
                duration = run_result.timing.duration
                if duration:
                    st.markdown(f"**Duration:** `{round(duration, 2)}s`")

            # Token usage
            if run_result.token_usage:
                st.markdown(
                    f"**Token Usage:** "
                    f"Input: `{run_result.token_usage.input_tokens:,}` | "
                    f"Output: `{run_result.token_usage.output_tokens:,}` | "
                    f"Total: `{run_result.token_usage.total_tokens:,}`"
                )

            # Steps summary
            if run_result.steps:
                st.markdown(f"**Total Steps:** `{len(run_result.steps)}`")
                if st.checkbox("View all steps (JSON)"):
                    st.json(run_result.steps)

    def _get_available_model_types(self) -> list[str]:
        """Return list of available model class names."""
        return [
            "InferenceClientModel",
            "TransformersModel",
            "LiteLLMModel",
            "LiteLLMRouterModel",
            "OpenAIModel",
            "AzureOpenAIModel",
            "AmazonBedrockModel",
            "VLLMModel",
            "MLXModel",
        ]

    def _create_model_from_config(
        self,
        model_type: str,
        model_id: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ):
        """Create a model instance from configuration."""
        from smolagents.models import (
            AmazonBedrockModel,
            AzureOpenAIModel,
            InferenceClientModel,
            LiteLLMModel,
            LiteLLMRouterModel,
            MLXModel,
            OpenAIModel,
            TransformersModel,
            VLLMModel,
        )

        model_classes = {
            "InferenceClientModel": InferenceClientModel,
            "TransformersModel": TransformersModel,
            "LiteLLMModel": LiteLLMModel,
            "LiteLLMRouterModel": LiteLLMRouterModel,
            "OpenAIModel": OpenAIModel,
            "AzureOpenAIModel": AzureOpenAIModel,
            "AmazonBedrockModel": AmazonBedrockModel,
            "VLLMModel": VLLMModel,
            "MLXModel": MLXModel,
        }

        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = model_classes[model_type]

        # Prepare model arguments based on type
        model_kwargs = kwargs.copy()
        if model_id:
            model_kwargs["model_id"] = model_id
        if provider:
            model_kwargs["provider"] = provider
        if api_key:
            model_kwargs["api_key"] = api_key
        if api_base:
            model_kwargs["api_base"] = api_base

        # Special handling for InferenceClientModel
        if model_type == "InferenceClientModel" and api_key:
            model_kwargs["token"] = api_key
            model_kwargs.pop("api_key", None)

        return model_class(**model_kwargs)

    def _validate_model_config(self, model_type: str, model_id: str | None = None) -> tuple[bool, str | None]:
        """Validate model configuration before creation."""
        if model_type not in self._get_available_model_types():
            return False, f"Unknown model type: {model_type}"

        # Some models require model_id
        if model_type in ["InferenceClientModel", "TransformersModel"] and not model_id:
            return False, f"{model_type} requires a model_id"

        return True, None

    def _render_settings_section(self, st):
        """Main settings section renderer with all sub-sections."""
        # Model Configuration
        with st.expander("🤖 Model Configuration", expanded=False):
            self._render_model_config(st)

        # Agent Parameters
        with st.expander("⚙️ Agent Parameters", expanded=False):
            self._render_agent_params(st)

        # Tool Management
        with st.expander("🛠️ Tool Management", expanded=False):
            self._render_tool_management(st)

        # Executor Configuration (CodeAgent only)
        if isinstance(st.session_state.agent_instance, CodeAgent):
            with st.expander("💻 Executor Configuration", expanded=False):
                self._render_executor_config(st)

        # Prompt Templates
        with st.expander("📝 Prompt Templates", expanded=False):
            self._render_prompt_templates(st)

        # Advanced Settings
        with st.expander("🔬 Advanced Settings", expanded=False):
            self._render_advanced_settings(st)

        # Function Access Panel
        with st.expander("🎯 Function Access", expanded=False):
            self._render_function_panel(st)
        
        # Settings Persistence
        st.markdown("---")
        st.markdown("**Settings Persistence**")
        persist_col1, persist_col2 = st.columns(2)
        with persist_col1:
            if st.button("Save Settings", use_container_width=True, key="save_settings_btn"):
                self._save_settings_to_file(st)
        with persist_col2:
            uploaded_settings = st.file_uploader("Load Settings", type=["json"], key="upload_settings_file")
            if uploaded_settings:
                if st.button("Load Settings", use_container_width=True, key="load_settings_btn"):
                    self._load_settings_from_file(st, uploaded_settings)

    def _render_model_config(self, st):
        """Render model configuration UI with registry management and per-step assignment."""
        agent = st.session_state.agent_instance
        
        # Model Registry Management
        st.markdown("**Model Registry**")
        registry_keys = list(st.session_state.model_registry.keys())
        
        if registry_keys:
            selected_model_key = st.selectbox("Select Model", options=registry_keys, key="selected_model_key")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Remove", key="remove_model"):
                    if len(registry_keys) > 1:
                        del st.session_state.model_registry[selected_model_key]
                        st.rerun()
                    else:
                        st.warning("Cannot remove the last model!")
            with col2:
                if st.button("Edit", key="edit_model"):
                    st.session_state[f"editing_model_{selected_model_key}"] = True
            with col3:
                if st.button("Test", key="test_model"):
                    model = st.session_state.model_registry[selected_model_key]
                    test_prompt = st.text_input("Test Prompt", value="Hello, how are you?", key="test_prompt_input")
                    if st.button("Run Test", key="run_test"):
                        try:
                            from smolagents.models import ChatMessage, MessageRole
                            test_msg = model.generate([ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": test_prompt}])])
                            st.success(f"Response: {test_msg.content}")
                        except Exception as e:
                            st.error(f"Test failed: {str(e)}")
        
        # Add New Model
        st.markdown("---")
        st.markdown("**Add New Model**")
        with st.expander("➕ Create Model", expanded=False):
            model_type = st.selectbox("Model Type", options=self._get_available_model_types(), key="new_model_type")
            model_id = st.text_input("Model ID", key="new_model_id", help="Required for InferenceClientModel and TransformersModel")
            provider = st.text_input("Provider (optional)", key="new_provider", help="For InferenceClientModel")
            api_key = st.text_input("API Key (optional)", type="password", key="new_api_key")
            api_base = st.text_input("API Base URL (optional)", key="new_api_base")
            
            # Additional kwargs
            st.markdown("**Additional Parameters (JSON)**")
            model_kwargs_text = st.text_area(
                "Model-specific parameters as JSON",
                value="{}",
                height=100,
                key="new_model_kwargs",
                help="e.g., {'temperature': 0.7, 'max_tokens': 1000}",
            )
            
            model_name = st.text_input("Model Name (for registry)", value=f"Model_{len(registry_keys) + 1}", key="new_model_name")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create Model", key="create_model_btn"):
                    try:
                        model_kwargs = json.loads(model_kwargs_text) if model_kwargs_text.strip() else {}
                        is_valid, error_msg = self._validate_model_config(model_type, model_id if model_id else None)
                        if not is_valid:
                            st.error(error_msg)
                        else:
                            new_model = self._create_model_from_config(
                                model_type=model_type,
                                model_id=model_id if model_id else None,
                                provider=provider if provider else None,
                                api_key=api_key if api_key else None,
                                api_base=api_base if api_base else None,
                                **model_kwargs,
                            )
                            st.session_state.model_registry[model_name] = new_model
                            st.success(f"Model '{model_name}' added to registry!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create model: {str(e)}")
            with col2:
                if st.button("Test Connection", key="test_new_model"):
                    try:
                        model_kwargs = json.loads(model_kwargs_text) if model_kwargs_text.strip() else {}
                        is_valid, error_msg = self._validate_model_config(model_type, model_id if model_id else None)
                        if not is_valid:
                            st.error(error_msg)
                        else:
                            test_model = self._create_model_from_config(
                                model_type=model_type,
                                model_id=model_id if model_id else None,
                                provider=provider if provider else None,
                                api_key=api_key if api_key else None,
                                api_base=api_base if api_base else None,
                                **model_kwargs,
                            )
                            from smolagents.models import ChatMessage, MessageRole
                            test_msg = test_model.generate([ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])])
                            st.success("Connection successful!")
                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")
        
        # Per-Step Model Assignment
        st.markdown("---")
        st.markdown("**Per-Step Model Assignment**")
        planning_model_key = st.selectbox(
            "Planning Steps Model",
            options=["Primary"] + [k for k in registry_keys if k != "Primary"],
            index=0 if "Planning" not in registry_keys else registry_keys.index("Planning") if "Planning" in registry_keys else 0,
            key="planning_model_select",
            help="Model to use for planning steps",
        )
        action_model_key = st.selectbox(
            "Action Steps Model",
            options=["Primary"] + [k for k in registry_keys if k != "Primary"],
            index=0 if "Action" not in registry_keys else registry_keys.index("Action") if "Action" in registry_keys else 0,
            key="action_model_select",
            help="Model to use for action steps",
        )
        final_answer_model_key = st.selectbox(
            "Final Answer Model",
            options=["Primary"] + [k for k in registry_keys if k != "Primary"],
            index=0 if "Final Answer" not in registry_keys else registry_keys.index("Final Answer") if "Final Answer" in registry_keys else 0,
            key="final_answer_model_select",
            help="Model to use for final answer generation",
        )
        
        if st.button("Apply Model Configuration", key="apply_model_config"):
            # Get models from registry
            planning_model = st.session_state.model_registry.get(planning_model_key, st.session_state.model_registry["Primary"])
            action_model = st.session_state.model_registry.get(action_model_key, st.session_state.model_registry["Primary"])
            final_answer_model = st.session_state.model_registry.get(final_answer_model_key, st.session_state.model_registry["Primary"])
            
            # Update agent models
            agent.planning_model = planning_model
            agent.action_model = action_model
            agent.final_answer_model = final_answer_model
            
            # Update registry if needed
            if planning_model_key != "Primary" and planning_model_key not in st.session_state.model_registry:
                st.session_state.model_registry["Planning"] = planning_model
            if action_model_key != "Primary" and action_model_key not in st.session_state.model_registry:
                st.session_state.model_registry["Action"] = action_model
            if final_answer_model_key != "Primary" and final_answer_model_key not in st.session_state.model_registry:
                st.session_state.model_registry["Final Answer"] = final_answer_model
            
            st.success("Model configuration applied!")
        
        # Predefined Presets
        st.markdown("---")
        st.markdown("**Model Presets**")
        preset_col1, preset_col2 = st.columns(2)
        with preset_col1:
            if st.button("Save Current Config as Preset", key="save_preset"):
                preset_name = st.text_input("Preset Name", key="preset_name_input", value="my_preset")
                if preset_name:
                    # Save to session state (could be extended to save to file)
                    if "model_presets" not in st.session_state:
                        st.session_state.model_presets = {}
                    st.session_state.model_presets[preset_name] = {
                        "planning": planning_model_key,
                        "action": action_model_key,
                        "final_answer": final_answer_model_key,
                    }
                    st.success(f"Preset '{preset_name}' saved!")
        with preset_col2:
            if "model_presets" in st.session_state and st.session_state.model_presets:
                preset_options = list(st.session_state.model_presets.keys())
                selected_preset = st.selectbox("Load Preset", options=preset_options, key="load_preset_select")
                if st.button("Load Preset", key="load_preset_btn"):
                    preset = st.session_state.model_presets[selected_preset]
                    st.session_state.planning_model_select = preset["planning"]
                    st.session_state.action_model_select = preset["action"]
                    st.session_state.final_answer_model_select = preset["final_answer"]
                    st.success(f"Preset '{selected_preset}' loaded!")
                    st.rerun()

    def _render_agent_params(self, st):
        """Render agent parameters UI."""
        agent = st.session_state.agent_instance
        
        # Max Steps
        st.session_state.max_steps_override = st.number_input(
            "Max Steps",
            min_value=1,
            max_value=100,
            value=agent.max_steps if st.session_state.max_steps_override is None else st.session_state.max_steps_override,
            help="Maximum number of steps the agent can take",
            key="agent_max_steps",
        )
        
        # Planning Interval
        planning_interval_value = agent.planning_interval if agent.planning_interval is not None else 0
        planning_interval_input = st.number_input(
            "Planning Interval",
            min_value=0,
            max_value=20,
            value=planning_interval_value,
            help="Interval at which the agent runs a planning step (0 = disabled)",
            key="agent_planning_interval",
        )
        if st.button("Apply Planning Interval", key="apply_planning_interval"):
            agent.planning_interval = planning_interval_input if planning_interval_input > 0 else None
            st.success("Planning interval updated!")
        
        # Verbosity Level
        from smolagents.monitoring import LogLevel
        verbosity_options = {
            "DEBUG (0)": LogLevel.DEBUG,
            "INFO (1)": LogLevel.INFO,
            "ERROR (2)": LogLevel.ERROR,
        }
        current_verbosity = agent.logger.level
        verbosity_key = [k for k, v in verbosity_options.items() if v == current_verbosity][0] if current_verbosity in verbosity_options.values() else "INFO (1)"
        selected_verbosity = st.selectbox(
            "Verbosity Level",
            options=list(verbosity_options.keys()),
            index=list(verbosity_options.keys()).index(verbosity_key) if verbosity_key in verbosity_options else 1,
            key="agent_verbosity",
        )
        if st.button("Apply Verbosity", key="apply_verbosity"):
            agent.logger.level = verbosity_options[selected_verbosity]
            st.success("Verbosity level updated!")
        
        # Stream Outputs
        stream_outputs = getattr(agent, "stream_outputs", False)
        new_stream_outputs = st.checkbox(
            "Stream Outputs",
            value=stream_outputs,
            help="Whether to stream outputs during execution",
            key="agent_stream_outputs",
        )
        if new_stream_outputs != stream_outputs:
            if st.button("Apply Stream Outputs", key="apply_stream_outputs"):
                agent.stream_outputs = new_stream_outputs
                st.success("Stream outputs setting updated!")
        
        # Return Full Result
        st.session_state.return_full_result = st.checkbox(
            "Return Full Result",
            value=st.session_state.return_full_result,
            help="If enabled, returns full RunResult with token usage, timing, and all steps",
            key="agent_return_full_result",
        )
        
        # Reset Memory on Run
        reset_memory_on_run = st.checkbox(
            "Reset Memory on Run",
            value=self.reset_agent_memory,
            help="Whether to reset agent memory at the start of each interaction",
            key="agent_reset_memory",
        )
        if reset_memory_on_run != self.reset_agent_memory:
            if st.button("Apply Reset Memory Setting", key="apply_reset_memory"):
                self.reset_agent_memory = reset_memory_on_run
                st.success("Reset memory setting updated!")
        
        # Custom Instructions
        current_instructions = getattr(agent, "instructions", "") or ""
        new_instructions = st.text_area(
            "Custom Instructions",
            value=current_instructions,
            height=100,
            help="Custom instructions for the agent, inserted in the system prompt",
            key="agent_instructions",
        )
        if new_instructions != current_instructions:
            if st.button("Apply Instructions", key="apply_instructions"):
                agent.instructions = new_instructions if new_instructions.strip() else None
                # Update system prompt
                agent.memory.system_prompt = agent.system_prompt
                st.success("Instructions updated!")
        
        # Agent Name
        current_name = agent.name or ""
        new_name = st.text_input(
            "Agent Name",
            value=current_name,
            help="Name of the agent (for managed agents)",
            key="agent_name",
        )
        if new_name != current_name:
            if st.button("Apply Name", key="apply_name"):
                try:
                    agent.name = new_name if new_name.strip() else None
                    st.success("Agent name updated!")
                except ValueError as e:
                    st.error(f"Invalid name: {str(e)}")
        
        # Agent Description
        current_description = agent.description or ""
        new_description = st.text_area(
            "Agent Description",
            value=current_description,
            height=80,
            help="Description of the agent (for managed agents)",
            key="agent_description",
        )
        if new_description != current_description:
            if st.button("Apply Description", key="apply_description"):
                agent.description = new_description if new_description.strip() else None
                st.success("Agent description updated!")

    def _render_tool_management(self, st):
        """Render tool management UI."""
        agent = st.session_state.agent_instance
        
        # Display current tools
        st.markdown("**Current Tools**")
        tool_names = [name for name in agent.tools.keys() if name != "final_answer"]
        if tool_names:
            for tool_name in tool_names:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"- `{tool_name}`")
                with col2:
                    if st.button("Remove", key=f"remove_tool_{tool_name}"):
                        if tool_name in agent.tools:
                            del agent.tools[tool_name]
                            st.success(f"Tool '{tool_name}' removed!")
                            st.rerun()
        else:
            st.info("No tools currently configured")
        
        st.markdown("---")
        st.markdown("**Add Tool**")
        
        # Tool import options
        tool_import_method = st.radio(
            "Import Method",
            options=["From Hub", "From MCP", "From LangChain", "From Code", "Built-in Tool"],
            key="tool_import_method",
        )
        
        if tool_import_method == "From Hub":
            hub_repo_id = st.text_input("Hugging Face Repo ID", key="tool_hub_repo", help="e.g., username/repo-name")
            if st.button("Load from Hub", key="load_tool_hub"):
                try:
                    from smolagents.tools import Tool
                    tool = Tool.from_hub(hub_repo_id)
                    agent.tools[tool.name] = tool
                    st.success(f"Tool '{tool.name}' loaded from Hub!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load tool: {str(e)}")
        
        elif tool_import_method == "From MCP":
            mcp_server_url = st.text_input("MCP Server URL", key="tool_mcp_url")
            if st.button("Connect to MCP", key="connect_mcp"):
                try:
                    from smolagents.tools import ToolCollection
                    tool_collection = ToolCollection.from_mcp(mcp_server_url)
                    st.success(f"Found {len(tool_collection.tools)} tools")
                    selected_tool = st.selectbox("Select Tool", options=[t.name for t in tool_collection.tools], key="mcp_tool_select")
                    if st.button("Add Tool", key="add_mcp_tool"):
                        tool = next(t for t in tool_collection.tools if t.name == selected_tool)
                        agent.tools[tool.name] = tool
                        st.success(f"Tool '{tool.name}' added!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to connect to MCP: {str(e)}")
        
        elif tool_import_method == "From LangChain":
            st.info("LangChain tool import requires the tool object. This feature can be extended.")
            langchain_tool_code = st.text_area("LangChain Tool Code/JSON", height=150, key="langchain_tool_code")
            if st.button("Import from LangChain", key="import_langchain_tool"):
                st.warning("LangChain import not yet implemented in UI")
        
        elif tool_import_method == "From Code":
            tool_code = st.text_area(
                "Tool Code",
                height=200,
                key="tool_code_input",
                help="Python code defining a tool function with @tool decorator",
            )
            if st.button("Create Tool from Code", key="create_tool_code"):
                try:
                    from smolagents.tools import Tool
                    tool = Tool.from_code(tool_code)
                    agent.tools[tool.name] = tool
                    st.success(f"Tool '{tool.name}' created from code!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create tool: {str(e)}")
        
        elif tool_import_method == "Built-in Tool":
            from smolagents.default_tools import TOOL_MAPPING
            builtin_tools = [name for name in TOOL_MAPPING.keys() if name != "python_interpreter" or isinstance(agent, CodeAgent)]
            selected_builtin = st.selectbox("Select Built-in Tool", options=builtin_tools, key="builtin_tool_select")
            if st.button("Add Built-in Tool", key="add_builtin_tool"):
                try:
                    tool_class = TOOL_MAPPING[selected_builtin]
                    tool = tool_class()
                    agent.tools[tool.name] = tool
                    st.success(f"Built-in tool '{tool.name}' added!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add tool: {str(e)}")
        
        # Tool execution test
        if tool_names:
            st.markdown("---")
            st.markdown("**Test Tool**")
            test_tool_name = st.selectbox("Select Tool to Test", options=tool_names, key="test_tool_select")
            test_tool_args = st.text_area("Tool Arguments (JSON)", value="{}", height=100, key="test_tool_args")
            if st.button("Execute Tool", key="execute_test_tool"):
                try:
                    tool = agent.tools[test_tool_name]
                    args = json.loads(test_tool_args)
                    result = tool(**args)
                    st.success(f"Tool Result: {str(result)[:500]}")
                except Exception as e:
                    st.error(f"Tool execution failed: {str(e)}")

    def _render_executor_config(self, st):
        """Render executor configuration UI for CodeAgent."""
        agent = st.session_state.agent_instance
        if not isinstance(agent, CodeAgent):
            st.info("Executor configuration is only available for CodeAgent")
            return
        
        # Executor Type
        executor_types = ["local", "blaxel", "e2b", "modal", "docker", "wasm"]
        current_executor = agent.executor_type
        executor_type_index = executor_types.index(current_executor) if current_executor in executor_types else 0
        new_executor_type = st.selectbox(
            "Executor Type",
            options=executor_types,
            index=executor_type_index,
            key="executor_type_select",
            help="Type of code executor to use",
        )
        
        # Executor kwargs
        current_kwargs = agent.executor_kwargs or {}
        executor_kwargs_text = st.text_area(
            "Executor kwargs (JSON)",
            value=json.dumps(current_kwargs, indent=2),
            height=100,
            key="executor_kwargs_input",
            help="Additional arguments for the executor",
        )
        
        if st.button("Apply Executor Config", key="apply_executor_config"):
            try:
                new_kwargs = json.loads(executor_kwargs_text) if executor_kwargs_text.strip() else {}
                agent.executor_type = new_executor_type
                agent.executor_kwargs = new_kwargs
                # Recreate executor
                agent.python_executor = agent.create_python_executor()
                st.success("Executor configuration updated!")
            except Exception as e:
                st.error(f"Failed to update executor: {str(e)}")
        
        # Authorized Imports
        current_imports = agent.additional_authorized_imports or []
        imports_text = st.text_area(
            "Authorized Imports",
            value="\n".join(current_imports),
            height=100,
            key="authorized_imports_input",
            help="One import per line (e.g., numpy, pandas)",
        )
        if st.button("Apply Authorized Imports", key="apply_imports"):
            new_imports = [imp.strip() for imp in imports_text.split("\n") if imp.strip()]
            agent.additional_authorized_imports = new_imports
            agent.authorized_imports = sorted(set(agent.authorized_imports) | set(new_imports))
            st.success("Authorized imports updated!")
        
        # Max Print Outputs Length
        current_max_print = agent.max_print_outputs_length
        new_max_print = st.number_input(
            "Max Print Outputs Length",
            min_value=0,
            value=current_max_print if current_max_print else 10000,
            key="max_print_outputs",
            help="Maximum length of print outputs",
        )
        if new_max_print != current_max_print:
            if st.button("Apply Max Print Length", key="apply_max_print"):
                agent.max_print_outputs_length = new_max_print
                st.success("Max print outputs length updated!")
        
        # Code Block Tags
        current_tags = agent.code_block_tags
        tag_col1, tag_col2 = st.columns(2)
        with tag_col1:
            opening_tag = st.text_input("Opening Tag", value=current_tags[0], key="code_opening_tag")
        with tag_col2:
            closing_tag = st.text_input("Closing Tag", value=current_tags[1], key="code_closing_tag")
        if (opening_tag, closing_tag) != current_tags:
            if st.button("Apply Code Block Tags", key="apply_code_tags"):
                agent.code_block_tags = (opening_tag, closing_tag)
                st.success("Code block tags updated!")

    def _render_prompt_templates(self, st):
        """Render prompt templates UI."""
        agent = st.session_state.agent_instance
        
        # View current templates
        st.markdown("**Current Prompt Templates**")
        import yaml
        if st.checkbox("Show Templates", key="show_templates"):
            templates_yaml = yaml.dump(agent.prompt_templates, default_flow_style=False, allow_unicode=True)
            st.code(templates_yaml, language="yaml")
        
        # Edit templates
        st.markdown("---")
        st.markdown("**Edit Templates**")
        template_editor = st.text_area(
            "YAML Editor",
            value=yaml.dump(agent.prompt_templates, default_flow_style=False, allow_unicode=True),
            height=300,
            key="template_editor",
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Apply Templates", key="apply_templates"):
                try:
                    new_templates = yaml.safe_load(template_editor)
                    agent.prompt_templates = new_templates
                    agent.memory.system_prompt = agent.system_prompt
                    st.success("Prompt templates updated!")
                except Exception as e:
                    st.error(f"Failed to parse templates: {str(e)}")
        with col2:
            if st.button("Reset to Defaults", key="reset_templates"):
                from smolagents.agents import EMPTY_PROMPT_TEMPLATES
                agent.prompt_templates = EMPTY_PROMPT_TEMPLATES
                st.success("Templates reset to defaults!")
        with col3:
            uploaded_file = st.file_uploader("Upload YAML file", type=["yaml", "yml"], key="template_file_upload")
            if uploaded_file:
                try:
                    templates = yaml.safe_load(uploaded_file)
                    agent.prompt_templates = templates
                    st.success("Templates loaded from file!")
                except Exception as e:
                    st.error(f"Failed to load templates: {str(e)}")

    def _render_advanced_settings(self, st):
        """Render advanced settings UI."""
        agent = st.session_state.agent_instance
        
        # Step Callbacks
        st.markdown("**Step Callbacks**")
        st.info("Step callbacks configuration requires code. This is a read-only view.")
        if agent.step_callbacks and hasattr(agent.step_callbacks, "_callbacks"):
            callback_info = {}
            for step_cls, callbacks in agent.step_callbacks._callbacks.items():
                callback_info[step_cls.__name__] = len(callbacks)
            st.json(callback_info)
        
        # Final Answer Checks
        st.markdown("**Final Answer Checks**")
        if agent.final_answer_checks:
            st.info(f"{len(agent.final_answer_checks)} validation function(s) configured")
            for i, check in enumerate(agent.final_answer_checks):
                st.code(f"Check {i+1}: {check.__name__ if hasattr(check, '__name__') else str(check)}", language="python")
        else:
            st.info("No final answer checks configured")
        
        # Managed Agents
        st.markdown("**Managed Agents**")
        if agent.managed_agents:
            for name, managed_agent in agent.managed_agents.items():
                st.markdown(f"- `{name}`: {managed_agent.description or 'No description'}")
        else:
            st.info("No managed agents configured")
        
        # Provide Run Summary
        provide_summary = getattr(agent, "provide_run_summary", False)
        new_provide_summary = st.checkbox(
            "Provide Run Summary",
            value=provide_summary,
            help="Whether to provide a run summary when called as a managed agent",
            key="provide_run_summary",
        )
        if new_provide_summary != provide_summary:
            if st.button("Apply Run Summary Setting", key="apply_run_summary"):
                agent.provide_run_summary = new_provide_summary
                st.success("Run summary setting updated!")
        
        # Add Base Tools
        st.markdown("**Base Tools**")
        st.info("Base tools are automatically added. This setting is read-only.")
        
        # System Prompt Override (Advanced)
        st.markdown("---")
        st.markdown("**System Prompt Override (Advanced)**")
        current_system_prompt = agent.system_prompt
        st.text_area(
            "System Prompt",
            value=current_system_prompt,
            height=200,
            key="system_prompt_override",
            help="⚠️ Advanced: Modifying system prompt directly may break agent functionality",
            disabled=True,  # Read-only for safety
        )
        st.warning("System prompt is read-only. Modify prompt templates instead.")

    def _render_function_panel(self, st):
        """Render function access panel."""
        agent = st.session_state.agent_instance
        
        # Execution Functions
        st.markdown("**Execution**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Interrupt Agent", use_container_width=True, key="func_interrupt"):
                agent.interrupt()
                st.warning("Agent interrupt signal sent!")
        with col2:
            if st.button("Cleanup", use_container_width=True, key="func_cleanup"):
                if hasattr(agent, "cleanup"):
                    agent.cleanup()
                    st.success("Agent cleaned up!")
        
        # Memory Operations
        st.markdown("**Memory Operations**")
        mem_col1, mem_col2, mem_col3 = st.columns(3)
        with mem_col1:
            if st.button("Reset Memory", use_container_width=True, key="func_reset_memory"):
                agent.memory.reset()
                st.success("Memory reset!")
        with mem_col2:
            if st.button("Get Full Steps", use_container_width=True, key="func_full_steps"):
                steps = agent.memory.get_full_steps()
                st.download_button(
                    "Download JSON",
                    data=json.dumps(steps, indent=2),
                    file_name="memory_steps.json",
                    mime="application/json",
                    key="download_full_steps",
                )
        with mem_col3:
            if st.button("Get Succinct Steps", use_container_width=True, key="func_succinct_steps"):
                steps = agent.memory.get_succinct_steps()
                st.download_button(
                    "Download JSON",
                    data=json.dumps(steps, indent=2),
                    file_name="memory_succinct.json",
                    mime="application/json",
                    key="download_succinct_steps",
                )
        
        if isinstance(agent, CodeAgent):
            if st.button("Return Full Code", use_container_width=True, key="func_full_code"):
                full_code = agent.memory.return_full_code()
                st.code(full_code, language="python")
                st.download_button(
                    "Download Code",
                    data=full_code,
                    file_name="agent_code.py",
                    mime="text/plain",
                    key="download_code",
                )
        
        # Visualization
        st.markdown("**Visualization**")
        if st.button("Visualize Agent Tree", use_container_width=True, key="func_visualize"):
            agent.visualize()
            st.info("Check console/terminal for visualization output")
        
        # Serialization
        st.markdown("**Serialization**")
        ser_col1, ser_col2 = st.columns(2)
        with ser_col1:
            if st.button("Export to Dict", use_container_width=True, key="func_export_dict"):
                agent_dict = agent.to_dict()
                st.download_button(
                    "Download JSON",
                    data=json.dumps(agent_dict, indent=2, default=str),
                    file_name="agent_config.json",
                    mime="application/json",
                    key="download_agent_dict",
                )
        with ser_col2:
            uploaded_agent = st.file_uploader("Load Agent Config", type=["json"], key="upload_agent_config")
            if uploaded_agent:
                if st.button("Load from Dict", use_container_width=True, key="func_load_dict"):
                    try:
                        agent_dict = json.load(uploaded_agent)
                        from smolagents.agents import MultiStepAgent
                        new_agent = MultiStepAgent.from_dict(agent_dict)
                        st.session_state.agent_instance = new_agent
                        st.success("Agent loaded from config!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load agent: {str(e)}")
        
        # Hub Operations
        st.markdown("**Hub Operations**")
        hub_col1, hub_col2 = st.columns(2)
        with hub_col1:
            hub_repo_id = st.text_input("Hub Repo ID", key="hub_repo_id", placeholder="username/repo-name")
            if st.button("Push to Hub", use_container_width=True, key="func_push_hub"):
                try:
                    agent.push_to_hub(hub_repo_id, space_sdk="streamlit")
                    st.success(f"Agent pushed to Hub: {hub_repo_id}")
                except Exception as e:
                    st.error(f"Failed to push to Hub: {str(e)}")
        with hub_col2:
            load_hub_repo = st.text_input("Load from Hub Repo ID", key="load_hub_repo_id", placeholder="username/repo-name")
            if st.button("Load from Hub", use_container_width=True, key="func_load_hub"):
                try:
                    from smolagents.agents import MultiStepAgent
                    new_agent = MultiStepAgent.from_hub(load_hub_repo, trust_remote_code=True)
                    st.session_state.agent_instance = new_agent
                    st.success("Agent loaded from Hub!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load from Hub: {str(e)}")
        
        # Save Agent
        st.markdown("**Save Agent**")
        save_dir = st.text_input("Output Directory", value="./saved_agent", key="save_agent_dir")
        if st.button("Save Agent", use_container_width=True, key="func_save_agent"):
            try:
                from pathlib import Path
                agent.save(Path(save_dir))
                st.success(f"Agent saved to {save_dir}")
            except Exception as e:
                    st.error(f"Failed to save agent: {str(e)}")

    def _save_settings_to_file(self, st):
        """Save current settings to a JSON file."""
        agent = st.session_state.agent_instance
        settings = {
            "agent_settings": {
                "max_steps": agent.max_steps,
                "planning_interval": agent.planning_interval,
                "verbosity_level": int(agent.logger.level),
                "stream_outputs": getattr(agent, "stream_outputs", False),
                "return_full_result": st.session_state.return_full_result,
                "reset_agent_memory": self.reset_agent_memory,
                "instructions": getattr(agent, "instructions", None),
                "name": agent.name,
                "description": agent.description,
            },
            "model_registry": {
                name: {
                    "type": type(model).__name__,
                    "config": model.to_dict(),
                }
                for name, model in st.session_state.model_registry.items()
            },
            "model_assignments": {
                "planning": getattr(agent, "planning_model", None) and type(agent.planning_model).__name__,
                "action": getattr(agent, "action_model", None) and type(agent.action_model).__name__,
                "final_answer": getattr(agent, "final_answer_model", None) and type(agent.final_answer_model).__name__,
            },
            "tools": [tool.to_dict() for tool in agent.tools.values() if tool.name != "final_answer"],
        }
        
        if isinstance(agent, CodeAgent):
            settings["executor_config"] = {
                "executor_type": agent.executor_type,
                "executor_kwargs": agent.executor_kwargs,
                "authorized_imports": agent.additional_authorized_imports,
                "max_print_outputs_length": agent.max_print_outputs_length,
                "code_block_tags": agent.code_block_tags,
            }
        
        settings_json = json.dumps(settings, indent=2, default=str)
        st.download_button(
            "Download Settings",
            data=settings_json,
            file_name="agent_settings.json",
            mime="application/json",
            key="download_settings",
        )
        st.success("Settings prepared for download!")

    def _load_settings_from_file(self, st, uploaded_file):
        """Load settings from a JSON file."""
        try:
            settings = json.load(uploaded_file)
            agent = st.session_state.agent_instance
            
            # Load agent settings
            if "agent_settings" in settings:
                agent_settings = settings["agent_settings"]
                if "max_steps" in agent_settings:
                    agent.max_steps = agent_settings["max_steps"]
                if "planning_interval" in agent_settings:
                    agent.planning_interval = agent_settings["planning_interval"]
                if "verbosity_level" in agent_settings:
                    from smolagents.monitoring import LogLevel
                    agent.logger.level = LogLevel(agent_settings["verbosity_level"])
                if "stream_outputs" in agent_settings:
                    agent.stream_outputs = agent_settings["stream_outputs"]
                if "instructions" in agent_settings:
                    agent.instructions = agent_settings["instructions"]
                if "name" in agent_settings:
                    agent.name = agent_settings["name"]
                if "description" in agent_settings:
                    agent.description = agent_settings["description"]
            
            # Load model registry
            if "model_registry" in settings:
                for name, model_info in settings["model_registry"].items():
                    try:
                        from smolagents.models import Model
                        model_class = getattr(__import__("smolagents.models", fromlist=[model_info["type"]]), model_info["type"])
                        model = model_class.from_dict(model_info["config"])
                        st.session_state.model_registry[name] = model
                    except Exception as e:
                        st.warning(f"Failed to load model '{name}': {str(e)}")
            
            # Load model assignments
            if "model_assignments" in settings:
                assignments = settings["model_assignments"]
                # This would require matching models from registry
            
            # Load tools
            if "tools" in settings:
                for tool_dict in settings["tools"]:
                    try:
                        from smolagents.tools import Tool
                        tool = Tool.from_code(tool_dict.get("code", ""))
                        agent.tools[tool.name] = tool
                    except Exception as e:
                        st.warning(f"Failed to load tool: {str(e)}")
            
            # Load executor config
            if "executor_config" in settings and isinstance(agent, CodeAgent):
                exec_config = settings["executor_config"]
                agent.executor_type = exec_config.get("executor_type", "local")
                agent.executor_kwargs = exec_config.get("executor_kwargs", {})
                agent.additional_authorized_imports = exec_config.get("authorized_imports", [])
                agent.max_print_outputs_length = exec_config.get("max_print_outputs_length")
                agent.code_block_tags = tuple(exec_config.get("code_block_tags", ("<code>", "</code>")))
            
            st.success("Settings loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load settings: {str(e)}")

    def _update_agent_from_settings(self, st):
        """Apply all settings from session state to the agent."""
        agent = st.session_state.agent_instance
        
        # Update max_steps if overridden
        if st.session_state.max_steps_override:
            agent.max_steps = st.session_state.max_steps_override
        
        # Update other settings that might have been changed
        # This method can be called to apply all pending changes at once
        st.success("Agent settings updated!")

    def _execute_agent_function(self, st, function_name: str, **kwargs):
        """Execute an agent function with error handling."""
        agent = st.session_state.agent_instance
        try:
            if hasattr(agent, function_name):
                func = getattr(agent, function_name)
                if callable(func):
                    result = func(**kwargs)
                    return result
                else:
                    st.error(f"{function_name} is not callable")
            else:
                st.error(f"Function {function_name} not found on agent")
        except Exception as e:
            st.error(f"Error executing {function_name}: {str(e)}")
            return None


__all__ = ["stream_to_streamlit", "StreamlitUI"]
