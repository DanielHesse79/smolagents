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
    Minimal Streamlit interface for interacting with a [`MultiStepAgent`].

    This class provides a simple web interface to interact with the agent in real-time, allowing users to submit prompts, upload files, and receive responses in a chat-like format.
    It can reset the agent's memory at the start of each interaction if desired.
    It supports file uploads, which are saved to a specified folder.
    This class requires the `streamlit` extra to be installed: `pip install 'smolagents[streamlit]'`.

    For advanced features (model registry, settings, tool management), use `AdvancedStreamlitUI` from `smolagents.streamlit_ui_advanced`.

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
        self.name = getattr(agent, "name") or "Agent Chat"
        self.description = getattr(agent, "description", None)
        if self.file_upload_folder is not None:
            if not self.file_upload_folder.exists():
                self.file_upload_folder.mkdir(parents=True, exist_ok=True)

    def upload_file(self, file, allowed_file_types=None):
        """
        Upload a file and add it to the list of uploaded files in the session state.

        The file is saved to the `self.file_upload_folder` folder.
        If the file type is not allowed, it returns None.

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

    def _get_ollama_models(self):
        """Get list of available Ollama models."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model.get("name", "") for model in models if model.get("name")]
        except ImportError:
            # requests not installed
            pass
        except Exception:
            # Ollama not running or connection failed
            pass
        return []

    def _is_ollama_model(self):
        """Check if the current model is an Ollama model."""
        from smolagents.models import LiteLLMModel
        if isinstance(self.agent.model, LiteLLMModel):
            model_id = getattr(self.agent.model, "model_id", "")
            return model_id.startswith("ollama_chat/")
        return False

    def run(self):
        """
        Run the Streamlit app with the agent interface.
        This should be called from a Streamlit script.
        """
        import streamlit as st
        from PIL import Image
        from smolagents import LiteLLMModel, CodeAgent, ToolCallingAgent

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "file_uploads_log" not in st.session_state:
            st.session_state.file_uploads_log = []
        if "image_uploads" not in st.session_state:
            st.session_state.image_uploads = []
        if "selected_ollama_model" not in st.session_state:
            # Extract current model name if using Ollama
            if self._is_ollama_model():
                model_id = self.agent.model.model_id
                st.session_state.selected_ollama_model = model_id.replace("ollama_chat/", "")
            else:
                st.session_state.selected_ollama_model = None

        # Sidebar - minimal controls
        with st.sidebar:
            st.markdown("---")
            
            # Model selection for Ollama
            if self._is_ollama_model():
                available_models = self._get_ollama_models()
                if available_models:
                    current_model = st.session_state.selected_ollama_model or available_models[0]
                    selected_model = st.selectbox(
                        "Select Ollama Model",
                        options=available_models,
                        index=available_models.index(current_model) if current_model in available_models else 0,
                        help="Choose which local Ollama model to use"
                    )
                    
                    # Rebuild agent if model changed
                    if selected_model != st.session_state.selected_ollama_model:
                        st.session_state.selected_ollama_model = selected_model
                        # Create new model
                        new_model = LiteLLMModel(
                            model_id=f"ollama_chat/{selected_model}",
                            api_base="http://localhost:11434",
                            api_key="ollama",
                            num_ctx=8192,
                        )
                        # Rebuild agent with new model
                        agent_kwargs = {
                            "model": new_model,
                            "tools": self.agent.tools,
                            "verbosity_level": getattr(
                                self.agent, "verbosity_level", getattr(getattr(self.agent, "logger", None), "level", 1)
                            ),
                            "name": self.agent.name,
                            "description": getattr(self.agent, "description", None),
                            "step_callbacks": getattr(self.agent, "step_callbacks", []),
                            "stream_outputs": getattr(self.agent, "stream_outputs", True),
                        }
                        if isinstance(self.agent, CodeAgent):
                            agent_kwargs["planning_interval"] = getattr(self.agent, "planning_interval", 3)
                            agent_kwargs["max_steps"] = getattr(self.agent, "max_steps", 10)
                            self.agent = CodeAgent(**agent_kwargs)
                        elif isinstance(self.agent, ToolCallingAgent):
                            self.agent = ToolCallingAgent(**agent_kwargs)
                        else:
                            # Fallback: just update the model
                            self.agent.model = new_model
                        st.success(f"Switched to model: {selected_model}")
                        st.rerun()
            
            st.markdown("---")
            
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                self.agent.memory.reset()
                st.session_state.image_uploads = []
                st.session_state.file_uploads_log = []
                st.rerun()

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

        # Main chat interface
        st.title(self.name)

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

                    # Use streaming mode - use agent's max_steps if set, otherwise None (uses agent default)
                    max_steps = getattr(self.agent, "max_steps", None)
                    for msg in stream_to_streamlit(
                        self.agent,
                        task=prompt,
                        task_images=task_images,
                        reset_agent_memory=self.reset_agent_memory,
                        additional_args=None,
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


__all__ = ["stream_to_streamlit", "StreamlitUI"]
