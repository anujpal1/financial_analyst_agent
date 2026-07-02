import sys
import warnings
from langchain_core.messages import HumanMessage
from agent.graph import agent_app
from evaluation.metrics import EVALUATION_CHALLENGES, grade_report

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("\n" + "="*60)
print(" 🔬 INITIATING AUTOMATED 8-STAGE EVALUATION FRAMEWORK ")
print("="*60 + "\n")

total_score = 0
max_possible_score = 20 * len(EVALUATION_CHALLENGES)

for idx, challenge in enumerate(EVALUATION_CHALLENGES, 1):
    print(f"\n▶️ RUNNING CHALLENGE {idx}/8: {challenge['level']}")
    print(f"Task: {challenge['task']}")
    
    # 1. Run the Agent (Automated, bypassing human-in-the-loop)
    thread_config = {"configurable": {"thread_id": f"eval_session_{idx}"}}
    inputs = {"messages": [HumanMessage(content=challenge['task'])]}
    
    agent_final_output = ""
    current_inputs = inputs
    
    try:
        while True:
            for event in agent_app.stream(current_inputs, thread_config, stream_mode="values"):
                last_message = event["messages"][-1]
                if last_message.content:
                    agent_final_output = last_message.content
            
            graph_state = agent_app.get_state(thread_config)
            if graph_state.next:
                # Bypass the human interrupt automatically for testing
                current_inputs = None
            else:
                break
                
        # 2. Grade the Output
        print("🔍 Grading output...")
        evaluation = grade_report(challenge['task'], agent_final_output)
        
        print(f"📊 SCORE: {evaluation['score']}/20")
        print(f"📝 FEEDBACK: {evaluation['feedback']}")
        
        total_score += evaluation['score']
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Challenge {idx} failed during execution: {str(e)}")

print("\n" + "="*60)
print(f" 🏆 FINAL AGENT PERFORMANCE SCORE: {total_score}/{max_possible_score} ({(total_score/max_possible_score)*100:.1f}%)")
print("="*60 + "\n")