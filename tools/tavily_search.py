from typing import Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables.config import RunnableConfig
from core import tracer

class TavilySearchTool(TavilySearchResults):
    @tracer.tool(name="tavily_search")
    def invoke(self, input: dict, config: Optional[RunnableConfig] = None):
        print(f"[LOG] TavilySearch called with input: {input}")
        result = super().invoke(input, config)
        print(f"[LOG] TavilySearch result: {result}")
        return result