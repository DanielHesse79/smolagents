from run import create_agent

from smolagents.streamlit_ui import StreamlitUI


agent = create_agent()

streamlit_ui = StreamlitUI(agent)
streamlit_ui.run()
