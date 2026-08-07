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
