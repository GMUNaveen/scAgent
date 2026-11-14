from langchain.messages import ToolMessage
from tool import tools_by_name


def tool_node(state: dict):
    """Performs the tool call.

    Errors are returned to the LLM as tool output (instead of crashing the graph) so it can
    correct itself, e.g. fix a wrong handle or tissue name.
    """

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        try:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
        except Exception as e:
            observation = f"Error running {tool_call['name']}: {type(e).__name__}: {e}"
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}
