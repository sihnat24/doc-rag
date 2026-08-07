from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import config


prompt = ChatPromptTemplate.from_messages([
    ("system", config.TOOL_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])