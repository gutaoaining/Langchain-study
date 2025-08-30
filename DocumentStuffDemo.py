# 作者：顾涛
# 创建时间：2025/6/29

import os

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'LangChainDemo'

# 创建模型
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# 加载文档，我们将使用WebBaseLoader 来加载blog文章：
loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load()  # 得到整篇文章

# 第一种：Stuff
# chain = load_summarize_chain(model, chain_type='stuff')
# result = chain.invoke(docs)
# print(result['output_text'])

# 第二种：定义提示
prompt_template = """针对下面的内容，写一个简洁的总结摘要:
"{text}"
简洁的总结摘要:"""

prompt = PromptTemplate.from_template(prompt_template)

llm_chain = LLMChain(llm=model, prompt=prompt)

stuff_chain = StuffDocumentsChain(llm_chain=llm_chain,document_variable_name='text')

result = stuff_chain.invoke(docs)
print(result['output_text'])