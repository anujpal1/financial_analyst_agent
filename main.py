import sys
import os
import warnings
from langchain_core.messages import HumanMessage
from agent.graph import agent_app
from memory.episodic import save_episode

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("\n" + "="*60)
print(" 🧠 RECORDING INITIAL EXPERIENCE EPISODE ")
print("="*60 + "\n")

# Simulate seeding a past historical victory into our local logs
seed_result = save_episode(
    ticker="AAPL",
    task="Analyze cash flows and check price levels.",
    successful_tools=["get_stock_price", "get_cash_flow_metrics"],
    execution_summary="Resolved a minor valuation gap by verifying operating metrics against current real-time prices. Prioritized yfinance data."
)
print(f"Log Status: {seed_result}\n")
print("-" * 60)

# Fire a query for an entirely different target company (AMZN)
user_query = "Gather a live stock price snapshot for AMZN and check for any data conflicts."
print(f"USER QUERY: {user_query}\n")

thread_config = {"configurable": {"thread_id": "episodic_session_1"}}
inputs = {"messages": [HumanMessage(content=user_query)]}

try:
    current_inputs = inputs
    while True:
        for event in agent_app.stream(current_inputs, thread_config, stream_mode="values"):
            last_message = event["messages"][-1]
            if last_message.content:
                print(f"\n🧠 AGENT SUMMARY:\n{last_message.content}")
                
        graph_state = agent_app.get_state(thread_config)
        if graph_state.next:
            # Bypass the human-in-the-loop interruption automatically for testing by inputting 'y' programmatically
            current_inputs = None
        else:
            break
            
    print("\n" + "="*60)
    print(" ✅ EPISODIC DISCOVERY RUN COMPLETED ")
    print("="*60 + "\n")

except KeyboardInterrupt:
    sys.exit(0)