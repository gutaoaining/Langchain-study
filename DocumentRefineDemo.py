# 作者：顾涛
# 创建时间：2025/6/29

import os

from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'LangChainDemo'

# 创建模型
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# 加载文档，我们将使用WebBaseLoader 来加载blog文章：
loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load()  # 得到整篇文章
'''
Refine: RefineDocumentsChain 类似于map-reduce：
文档链通过循环遍历输入文档并逐步更新其答案来构建响应。对于每个文档，它将当前文档和最新的中间答案传递给LLM链，以获得新的答案。
'''
# 第一步：切割阶段
# 每一个小docs为1000个token
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

#指定chain_type为refine
chain = load_summarize_chain(model,chain_type='refine')

result = chain.invoke(split_docs)
print(result['output_text'])