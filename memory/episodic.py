import os
import json
import config

def save_episode(ticker: str, task: str, successful_tools: list, execution_summary: str):
    """Saves a summary of a successful research run into a local JSON file for historical reference."""
    try:
        log_path = config.EPISODIC_LOG_PATH if hasattr(config, 'EPISODIC_LOG_PATH') else "episodes.json"
        
        # Load existing episodes if file exists
        episodes = []
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                try:
                    episodes = json.load(f)
                except json.JSONDecodeError:
                    episodes = []
                    
        # Create the new episode object
        new_episode = {
            "ticker": ticker.upper(),
            "task": task,
            "tools_used": successful_tools,
            "summary": execution_summary
        }
        episodes.append(new_episode)
        
        # Write back to disk
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(episodes, f, indent=4)
            
        return f"Successfully saved episode for {ticker} into memory storage."
    except Exception as e:
        return f"Failed to record episodic memory: {str(e)}"

def recall_past_episodes() -> str:
    """Reads all historical episodes to serve as structural contextual patterns for the agent."""
    log_path = config.EPISODIC_LOG_PATH if hasattr(config, 'EPISODIC_LOG_PATH') else "episodes.json"
    if not os.path.exists(log_path):
        return "No past execution episodes found in the historical log."
        
    with open(log_path, "r", encoding="utf-8") as f:
        try:
            episodes = json.load(f)
            summary_strings = []
            for ep in episodes[-3:]:  # Limit to the last 3 historical runs to protect context limits
                summary_strings.append(
                    f"- Past Task for {ep['ticker']}: {ep['task']}\n"
                    f"  Successful Tools Called: {ep['tools_used']}\n"
                    f"  Resolution Summary: {ep['summary']}"
                )
            return "\n\n".join(summary_strings)
        except Exception:
            return "Historical execution records exist but are unreadable."