import os
import json
import config
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. THE 8 PROGRESSIVE CHALLENGES
# ==========================================
EVALUATION_CHALLENGES = [
    {"id": 1, "level": "Basic", "task": "Fetch the current live price and volume for AAPL."},
    {"id": 2, "level": "Basic Math", "task": "Calculate the trailing and forward P/E ratios for MSFT."},
    {"id": 3, "level": "Extraction", "task": "Extract the latest Net Income from the SEC EDGAR database for NVDA."},
    {"id": 4, "level": "Unstructured", "task": "Summarize the top 3 financial news headlines for TSLA."},
    {"id": 5, "level": "Time-Series", "task": "Run a 30-day historical momentum trend check on GOOGL."},
    {"id": 6, "level": "Logic/Mapping", "task": "Identify the primary industry competitors for AMD."},
    {"id": 7, "level": "Advanced Math", "task": "Calculate a 5-year Discounted Cash Flow (DCF) intrinsic valuation for META."},
    {"id": 8, "level": "Master Synthesis", "task": "Combine SEC data, live prices, DCF valuation, and recent news into a master briefing for AAPL."}
]

# ==========================================
# 2. LLM-AS-A-JUDGE CONFIGURATION
# ==========================================
# We use a separate instance of the LLM to grade the agent's work objectively
evaluator_llm = ChatOllama(
    model=config.MODEL_NAME,
    temperature=0.0,
    num_ctx=config.MAX_CONTEXT_TOKENS,
    options={"num_thread": config.OLLAMA_NUM_THREADS}
)

EVALUATION_RUBRIC = """
You are an elite QA Auditor. Grade the provided financial report on a scale of 0 to 20 based on these metrics:
1. Data Accuracy (0-5 pts): Are the numbers realistic and clearly sourced?
2. Tool Utilization (0-5 pts): Did the agent obviously use tools (like DCF or SEC) to get this data?
3. Formatting (0-5 pts): Is the report structured beautifully with bullet points and clear headers?
4. Conflict Resolution (0-5 pts): Did the agent avoid hallucinations and explicitly address any missing data?

Respond strictly in this format:
SCORE: [0-20]
FEEDBACK: [One sentence of critical feedback]
"""

def grade_report(challenge_task: str, agent_output: str) -> dict:
    """Uses Llama 3.1 to evaluate agent output with improved parsing resilience."""
    try:
        messages = [
            SystemMessage(content=EVALUATION_RUBRIC),
            HumanMessage(content=f"CHALLENGE: {challenge_task}\n\nAGENT REPORT:\n{agent_output}")
        ]
        
        response = evaluator_llm.invoke(messages)
        output_text = response.content
        
        score = 0
        feedback = "Could not parse descriptive feedback line cleanly."
        
        # Resilient line parsing
        for line in output_text.split('\n'):
            if "SCORE:" in line.upper():
                try:
                    # Extracts digits regardless of spaces or brackets
                    score = int(''.join(filter(str.isdigit, line)))
                except ValueError:
                    score = 0
            elif "FEEDBACK:" in line.upper():
                feedback = line.split(":", 1)[1].strip()
                
        # Guard rails for scores out of bounds
        score = min(max(score, 0), 20)
        
        return {"score": score, "feedback": feedback}
    except Exception as e:
        return {"score": 0, "feedback": f"Evaluation pipeline exception: {str(e)}"}