from langchain.agents import create_agent
import config
import tools

agent_graph = create_agent(
    config.LLM,
    [tools.search_knowledge_base, tools.list_knowledge_base, tools.query_spreadsheet, tools.web_search],
    system_prompt=config.TOOL_PROMPT,
)


def run_agent(question: str) -> str:
    result = agent_graph.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


def run_agent_eval(question: str) -> dict:
    """Run agent and return answer + tool trace for eval scoring."""
    result = agent_graph.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]

    tools_called = []
    tool_outputs = []

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(tc["name"])
        # ToolMessage: has .name (tool name) and .content (tool output)
        if hasattr(msg, "name") and msg.name and hasattr(msg, "content"):
            tool_outputs.append(msg.content)

    return {
        "answer": messages[-1].content,
        "tools_called": tools_called,
        "tool_outputs": tool_outputs,
    }
