from run import create_agent

from intelcore.streamlit_ui import StreamlitUI


agent = create_agent()

streamlit_ui = StreamlitUI(agent)
streamlit_ui.run()
