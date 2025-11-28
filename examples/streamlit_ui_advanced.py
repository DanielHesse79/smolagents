import sys
import os

# Add parent directory to path to use local source (for development)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents import CodeAgent, AdvancedStreamlitUI, InferenceClientModel, WebSearchTool


agent = CodeAgent(
    tools=[WebSearchTool()],
    model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct", provider="fireworks-ai"),
    verbosity_level=1,
    planning_interval=3,
    name="example_agent",
    description="This is an example agent with advanced UI features.",
    step_callbacks=[],
    stream_outputs=True,
)

advanced_ui = AdvancedStreamlitUI(agent, file_upload_folder="./data")
advanced_ui.run()

