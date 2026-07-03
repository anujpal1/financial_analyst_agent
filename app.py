import streamlit as st
import warnings
import os
from langchain_core.messages import HumanMessage
from agent.graph import agent_app
from utils.pdf_generator import export_markdown_to_pdf
from utils.ticker_resolver import resolve_query_tickers
from config import SYSTEM_LOGGER as logger

warnings.filterwarnings("ignore", category=ResourceWarning)

# 1. Page Configuration & Premium Theme Override
st.set_page_config(
    page_title="Institutional Intelligence Terminal", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State tracking to keep assets persistent across clicks
if "final_report" not in st.session_state:
    st.session_state.final_report = ""
if "extracted_tickers" not in st.session_state:
    st.session_state.extracted_tickers = []
if "running" not in st.session_state:
    st.session_state.running = False

# 2. Sidebar Terminal Controls
with st.sidebar:
    st.markdown("### 🏛️ System Control Panel")
    st.markdown("---")
    st.markdown("**Core Engine:** `Llama 3.1 (8B)` via Ollama")
    st.markdown("**Framework:** `LangGraph ReAct Loop`")
    
    st.markdown("### 🔌 API Integration Status")
    # Quick live feedback checking if keys are populated
    tavily_ok = "✅ Active" if os.getenv("TAVILY_API_KEY") else "❌ Missing"
    news_ok = "✅ Active" if os.getenv("NEWSAPI_KEY") else "❌ Missing"
    sec_ok = "✅ Active" if os.getenv("SEC_USER_AGENT") else "❌ Missing"
    
    st.write(f"- **Tavily AI Search Engine:** {tavily_ok}")
    st.write(f"- **NewsAPI Market Feed:** {news_ok}")
    st.write(f"- **SEC EDGAR Retrieval:** {sec_ok}")
    st.markdown("---")
    st.caption("Institutional Research Framework v2.0 • 2026")

# 3. Hero Header Section
st.markdown("# ⚡ Institutional Equity Research Terminal")
st.markdown("##### High-performance decentralized data pipelines powered by Pandas, NumPy, Loguru, and Plotly.")
st.markdown("---")

# 4. Input Architecture Component
col_input, col_action = st.columns([5, 1])

with col_input:
    user_query = st.text_input(
        "Search Objective Routing Entry:",
        placeholder="e.g., Run a multi-source fundamental analysis on NVDA and verify risk factors...",
        label_visibility="collapsed"
    )

with col_action:
    execute_click = st.button("🚀 Run Analysis", use_container_width=True)

# 5. Core Operational Execution Block
if execute_click and user_query.strip():
    st.session_state.running = True
    st.session_state.final_report = ""
    st.session_state.extracted_tickers = []
    
    status_box = st.empty()
    
    # Run Smart Named Entity Resolution (NER)
    with st.spinner("🔍 Scanning market directory for corporate entities..."):
        st.session_state.extracted_tickers = resolve_query_tickers(user_query)
        
    augmented_query = user_query
    if st.session_state.extracted_tickers:
        augmented_query = f"{user_query} (Target Tickers: {', '.join(st.session_state.extracted_tickers)})"
        
    inputs = {"messages": [HumanMessage(content=augmented_query)]}
    thread_config = {"configurable": {"thread_id": "production_session_1"}}
    
    with st.spinner("🧠 Initializing execution graph pathways..."):
        try:
            while True:
                for event in agent_app.stream(inputs, thread_config, stream_mode="values"):
                    last_message = event["messages"][-1]
                    
                    # Stream current execution steps directly to user
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        for tool in last_message.tool_calls:
                            status_box.info(f"🔧 **Pipeline Invocation:** Running module `{tool['name']}`...")
                    
                    if last_message.content:
                        st.session_state.final_report = last_message.content
                
                graph_state = agent_app.get_state(thread_config)
                if graph_state.next:
                    inputs = None
                else:
                    break
                    
            status_box.empty()
            st.toast("Analysis Suite Execution Complete!", icon="✅")
            
            # Auto-compile PDF asset on completion
            pdf_path = "reports/Executive_Briefing.pdf"
            export_markdown_to_pdf(st.session_state.final_report, pdf_path)
            
        except Exception as e:
            logger.error(f"Execution Error: {str(e)}")
            st.error(f"Operational pipeline failed: {str(e)}")
            
    st.session_state.running = False

# 6. Persistent Institutional Output Layout
if st.session_state.final_report:
    # Quick Metric Ticker Display Header if tickers exist
    if st.session_state.extracted_tickers:
        metric_cols = st.columns(len(st.session_state.extracted_tickers))
        for idx, tkr in enumerate(st.session_state.extracted_tickers):
            with metric_cols[idx]:
                st.metric(label="Target System Locked", value=tkr.upper(), delta="Pipeline Connected")
    
    # Layout Split: Main Report Tabs on left, Persistent PDF Action Card on right
    report_layout, download_layout = st.columns([4, 1])
    
    with report_layout:
        # Organizes details into standard clean tabs for better readability
        tab_brief, tab_charts = st.tabs(["📋 Executive Briefing", "📈 Technical Visualizations"])
        
        with tab_brief:
            st.markdown(st.session_state.final_report)
            
        with tab_charts:
            chart_folder = "reports/charts"
            chart_rendered = False
            if os.path.exists(chart_folder):
                charts = [f for f in os.listdir(chart_folder) if f.endswith(".html")]
                for chart in charts:
                    # Filter charts to match currently analyzed tickers to avoid screen spam
                    if any(tkr.upper() in chart.upper() for tkr in st.session_state.extracted_tickers) or not st.session_state.extracted_tickers:
                        with open(f"{chart_folder}/{chart}", "r", encoding="utf-8") as f:
                            st.html(f.read())
                            chart_rendered = True
            if not chart_rendered:
                st.info("No interactive charts compiled for this specific search query path.")
                
    with download_layout:
        st.markdown("### 📥 Document Hub")
        st.markdown("Download and print production-ready materials for physical handouts or presentation decks.")
        
        pdf_target_path = "reports/Executive_Briefing.pdf"
        if os.path.exists(pdf_target_path):
            with open(pdf_target_path, "rb") as pdf_file:
                # This button is now safely separated and will not vanish upon click interaction!
                st.download_button(
                    label="📄 Download Institutional PDF",
                    data=pdf_file,
                    file_name="Executive_Briefing.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
        else:
            st.warning("Compiling briefing document...")
elif not st.session_state.running:
    st.info("💡 Terminal idle. Enter a target strategy query above to launch parallel processing agents.")