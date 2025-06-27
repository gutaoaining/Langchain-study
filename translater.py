# 作者：顾涛
# 创建时间：2025/6/26
import os

from fastapi import FastAPI
from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# 1、创建模型
model = ChatOpenAI(model='gpt-4-turbo')
# 2、准备提示prompt

my_prompt_template = ChatPromptTemplate.from_messages([
    ('system', '请将下面的内容翻译成{language}'),
    ('user', '{text}')
])
# msg = [
#     SystemMessage(content='请将以下的内容翻译成意大利语'),
#     HumanMessage(content='你好，请问你要去哪里？')
# ]

paser = StrOutputParser()
# 4、得到链
chain = my_prompt_template | model | paser

# 5、直接实用chain来调用
print(chain.invoke({'language': 'English', 'text': '我喜欢的那个女孩她的名字叫宁宁'}))

# 把我们的程序部署成服务
# 创建FastAPI的应用
app = FastAPI(title='我的langchain服务', version='v1.0', description='使用LangChain翻译任何语句的服务')

add_routes(
    app,
    chain,
    path='/chainDemo'
)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
