# 作者：顾涛
# 创建时间：2025/6/29
import os
from typing import List

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import chat_agent_executor
from pydantic import BaseModel

from DocumentDemo import response

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'LangChainChatDemo'

# 创建模型
model = ChatOpenAI(model='gpt-4-turbo')

# LangChain内置了一个工具，可以轻松地实用Tavily搜索引擎作为工具
search = TavilySearch(max_results=2)

# 让模型绑定工具
model_with_tools = model.bind_tools([search])
# 模型可以自动推理：是否需要调用工具去完成用户的答案
response1 = model_with_tools.invoke([HumanMessage(content="中国的首都是哪个城市？")])
print(f'Model_Result_Content:{response1.content}')
print(f'Model_Result_Content:{response1.tool_calls}')
response2 = model_with_tools.invoke([HumanMessage(content="北京天气怎么样？")])
print(f'Model_Result_Content:{response2.content}')
print(f'Model_Result_Content:{response2.tool_calls}')

# 创建代理
agent_executor = chat_agent_executor.create_tool_calling_executor(model, [search])
question1 = {'messages': [HumanMessage(content="中国的首都是哪个城市？")]}
resp1 = agent_executor.invoke(**question1)
print(resp1['messages'])
question2 = {'messages': [HumanMessage(content="北京天气怎么样？")]}
resp2 = agent_executor.invoke(**question2)
print(resp2['messages'])
