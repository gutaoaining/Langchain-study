# 作者：顾涛
# 创建时间：2025/6/29

import os
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'LangChainDemo'

# 创建模型
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.8)

# 创建链
chain = create_data_generation_chain(model)

# 生成数据
result = chain.invoke(  # 给一些关键词，随机生成一句话
    {
        "fields": {"颜色": ['蓝色', '绿色']},
        "preferences": {"style": "让它像诗歌一样优美。"}
    }
)

print(result)
