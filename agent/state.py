from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage

def append_messages(left: Sequence[BaseMessage], right: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """Reducer algorithm to seamlessly stitch incoming tracking streams."""
    return list(left) + list(right)

class AgentState(TypedDict):
    """The central short-term state interface of the autonomous analyst."""
    messages: Annotated[Sequence[BaseMessage], append_messages]
    target_ticker: str
    research_steps: list[str]
    collected_financials: dict
    compiled_markdown_report: str