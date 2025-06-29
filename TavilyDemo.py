# 作者：顾涛
# 创建时间：2025/6/29
import os

from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'LangChainChatDemo'

# 创建模型
model = ChatOpenAI(model='gpt-4-turbo')

# LangChain内置了一个工具，可以轻松地实用Tavily搜索引擎作为工具
search = TavilySearch(max_results=2)

print(search.invoke('今天的天气怎么样？'))
