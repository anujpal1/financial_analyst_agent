from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from agent.state import AgentState
from tools import FINANCIAL_TOOLS


from memory.episodic import recall_past_episodes
from langchain_core.messages import SystemMessage


import config

# ==========================================
# 1. INITIALIZE THE MODEL (BOUND TO TOOLS)
# ==========================================
llm = ChatOllama(
    model=config.MODEL_NAME,
    temperature=config.MODEL_TEMPERATURE,
    num_ctx=config.MAX_CONTEXT_TOKENS,
    options={"num_thread": config.OLLAMA_NUM_THREADS}
)

# Bind our 10+ tool registry array directly to the model
llm_with_tools = llm.bind_tools(FINANCIAL_TOOLS)

# ==========================================
# 2. DEFINE THE NODES (The Actions)
# ==========================================

def agent_reasoner(state: AgentState):
    """The brain node that injects episodic context and handles strict conflicting data synthesis parameters."""
    messages = state["messages"]
    
    # 1. Dynamically read past experiences out of our JSON episode ledger
    past_experiences = recall_past_episodes()
    
    # 2. Inject strong conflict resolution guidelines alongside episodic behavior
    system_instruction = SystemMessage(content=(
        f"You are an elite, institutional financial analyst engine operating within a strict validation framework.\n\n"
        f"--- HISTORICAL OPERATIONAL EXPERIENCES ---\n"
        f"{past_experiences}\n"
        f"-----------------------------------------\n\n"
        f"CRITICAL EXECUTION MANDATES:\n"
        f"1. DATA ACCURACY: Always utilize your tools to fetch data. Never guess or approximate a current financial metric.\n"
        f"2. TOOL UTILIZATION: For math-heavy tasks (like DCF or ratios), invoke the corresponding tool. Do not perform complex multi-year discounting raw in text.\n"
        f"3. REPORT FORMATTING: Your final output MUST be a beautifully structured Markdown report using clear headers (##), bullet points, and explicit metrics.\n"
        f"4. CONFLICT RESOLUTION: If different tools present conflicting or missing numbers, explicitly write a 'Data Conflict Resolution' section explaining your synthesis logic.\n\n"
        f"RESPONSE OUTPUT STRUCTURE:\n"
        f"When you have completed your research and are ready to present your final report to the user, you MUST include the text '[FINAL REPORT]' on a line by itself immediately before your Markdown text begins."
    ))
    
    # Pack parameters and trigger local Llama execution thread
    response = llm_with_tools.invoke([system_instruction] + messages)
    return {"messages": [response]}

def route_decision(state: AgentState):
    """The router edge that inspects if a tool must be triggered or if we are finished."""
    last_message = state["messages"][-1]
    
    # If the LLM requested tool execution items, route to tools block
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "execute_tools"
        
    # Otherwise, complete the research run loop
    return END

# ==========================================
# 3. BUILD AND COMPILE THE STATE GRAPH
# ==========================================

# Instantiate the graph template using our short-term state schema
workflow = StateGraph(AgentState)

# Append operational processing nodes
workflow.add_node("agent", agent_reasoner)
workflow.add_node("execute_tools", ToolNode(FINANCIAL_TOOLS))

# Connect boundaries
workflow.add_edge(START, "agent")

# Add conditional routing paths out of the reasoning stage
workflow.add_conditional_edges(
    "agent",
    route_decision,
    {
        "execute_tools": "execute_tools",
        END: END
    }
)

# Route output collections back to reasoning node for reflection
workflow.add_edge("execute_tools", "agent")

# Initialize an in-memory saver checkpoint state tracking registry
memory_checkpoint = MemorySaver()

# Compile the agent graph. 
# INTERRUPT_BEFORE pauses execution right before entering the tool runner node!
agent_app = workflow.compile(
    checkpointer=memory_checkpoint,
    interrupt_before=["execute_tools"]
)