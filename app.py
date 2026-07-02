import streamlit as st
import warnings
from langchain_core.messages import HumanMessage
from agent.graph import agent_app
import os

warnings.filterwarnings("ignore", category=ResourceWarning)

# Configure the web page
st.set_page_config(page_title="AI Financial Analyst", page_icon="📈", layout="wide")
st.title("🤖 Autonomous Institutional Financial Analyst")
st.markdown("Powered by Llama 3.1, LangGraph, and a 10-Tool Data Engine.")

# Create the user input box
user_query = st.text_area("Enter your financial research objective:", height=100, 
                          placeholder="e.g., Run a DCF valuation on MSFT and summarize recent news...")

if st.button("Execute Research Protocol"):
    if not user_query.strip():
        st.warning("Please enter a query first.")
    else:
        # Create UI containers for live updates
        status_text = st.empty()
        report_container = st.container()
        
        inputs = {"messages": [HumanMessage(content=user_query)]}
        thread_config = {"configurable": {"thread_id": "web_session_1"}}
        
        final_report = ""
        current_inputs = inputs  # Track inputs for the loop
        
        with st.spinner("🧠 Initializing agent reasoning loop..."):
            try:
                # ---------------------------------------------------------
                # THE FIX: Wrap in a while loop to handle the pause state!
                # ---------------------------------------------------------
                while True:
                    for event in agent_app.stream(current_inputs, thread_config, stream_mode="values"):
                        last_message = event["messages"][-1]
                        
                        # Show tools being executed in the UI
                        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            for tool in last_message.tool_calls:
                                status_text.info(f"🔧 **Tool Triggered:** Executing `{tool['name']}`...")
                        
                        # Capture the text
                        if last_message.content:
                            final_report = last_message.content
                    
                    # Check if the graph paused because of our Phase 6 checkpointer
                    graph_state = agent_app.get_state(thread_config)
                    if graph_state.next:
                        # Automatically pass 'None' to authorize the tools in the background
                        current_inputs = None 
                    else:
                        # No more steps left, execution is fully complete
                        break
                
                # Clear the status messages and display the final markdown report
                status_text.empty()
                with report_container:
                    st.success("✅ Research Protocol Completed")
                    st.markdown("---")
                    st.markdown(final_report)
                    
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")