import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from config import SYSTEM_LOGGER as logger

# Quick local dictionary map to bypass the LLM completely for major names
COMMON_COMPANIES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "meta": "META",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC"
}

resolver_llm = ChatOllama(model="llama3.1:8b", temperature=0)

def resolve_query_tickers(user_query: str) -> list:
    """Parses user queries and extracts verified ticker symbols before graph execution."""
    logger.info(f"Normalizing query for entity resolution: '{user_query}'")
    
    found_tickers = set()
    query_lower = user_query.lower()

    # 1. Check local fast-dictionary map first
    for name, ticker in COMMON_COMPANIES.items():
        if name in query_lower or ticker.lower() in query_lower:
            found_tickers.add(ticker)

    # 2. If nothing matched or it's a complex multi-stock query, invoke a quick extraction pass
    if not found_tickers:
        logger.info("Local dictionary miss. Triggering pre-processing extraction pass...")
        prompt = (
            "You are a precise data extractor. Identify any corporate entities mentioned "
            "in the user query and convert them into standard stock exchange ticker symbols. "
            "Return the output strictly as a JSON list of strings. If no corporate entities are found, "
            "return an empty list []. Do not include markdown formatting or extra text.\n\n"
            f"Query: {user_query}"
        )
        
        try:
            response = resolver_llm.invoke([HumanMessage(content=prompt)])
            text_out = response.content.strip()
            
            # Clean up potential LLM markdown ticks if it didn't listen perfectly
            if "```json" in text_out:
                text_out = text_out.split("```json")[1].split("```")[0].strip()
            elif "```" in text_out:
                text_out = text_out.split("```")[1].strip()
                
            parsed = json.loads(text_out)
            if isinstance(parsed, list):
                for symbol in parsed:
                    found_tickers.add(symbol.upper().strip())
        except Exception as e:
            logger.error(f"Ticker resolution pass faulted: {str(e)}")

    final_list = list(found_tickers)
    logger.info(f"Resolution outcome: Extracted Tickers -> {final_list}")
    return final_list