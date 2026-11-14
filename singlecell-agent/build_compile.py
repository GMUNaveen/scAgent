from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from state import MessagesState
from model_node import llm_call
from tool_node import tool_node
from end_logic import should_continue

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Invoke (demo) — only when run directly, so importing `agent` is side-effect free
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content="Annotate the Heart tissue with ML using logistic regression and random forest.")]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()