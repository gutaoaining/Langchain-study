# 作者：顾涛
# 创建时间：2025/6/22
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# 1、创建模型
model = ChatOpenAI(model='gpt-4-turbo')
# 2、准备提示prompt
msg = [
    SystemMessage(content='请将以下的内容翻译成意大利语'),
    HumanMessage(content='你好，请问你要去哪里？')
]

# result = model.invoke(msg)
# print(result)

paser = StrOutputParser()
# print(paser.invoke(result))
# 4、得到链
chain = model | paser

# 5、直接实用chain来调用
print(chain.invoke(msg))
