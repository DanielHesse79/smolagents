import streamlit as st
from smolagents import CodeAgent, StreamlitUI, InferenceClientModel, WebSearchTool


agent = CodeAgent(
    tools=[WebSearchTool()],
    model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct", provider="fireworks-ai"),
    verbosity_level=1,
    planning_interval=3,
    name="example_agent",
    description="This is an example agent.",
    step_callbacks=[],
    stream_outputs=True,
    # use_structured_outputs_internally=True,
)

streamlit_ui = StreamlitUI(agent, file_upload_folder="./data")
streamlit_ui.run()
