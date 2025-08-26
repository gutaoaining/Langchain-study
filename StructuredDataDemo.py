# 作者：顾涛
# 创建时间：2025/6/29

import os
from typing import Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langserve import add_routes
from pydantic import BaseModel, Field

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'LangChainChatDemo'

# 创建模型
model = ChatOpenAI(model='gpt-4-turbo')


# pydantic:处理数据，验证数据，定义数据的格式，虚拟化和反序列化，类型转换等等

# 定义个类
class Person(BaseModel):
    """
     关于一个人的数据模型
    """
    name: Optional[str] = Field(default=None, description="表示人的名字")

    hair_color: Optional[str] = Field(default=None, description="如果知道的话，这个人的头发颜色")

    height_in_meters: Optional[str] = Field(default=None, description="以米为单位测量的高度")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一个专业的提取算法。只从未结构化文本中提取相关信息。如果你不知道要提取的属性的值，返回该属性的值为null。"),
        ("human", "{text}"),
    ]
)

chain = {"text": RunnablePassthrough()} | prompt | model.with_structured_output(schema=Person)

text = "马路上走来一个女生，长长的黑头发披在肩上，大概1米7左右，"
resp = chain.invoke(text)
print(resp)
