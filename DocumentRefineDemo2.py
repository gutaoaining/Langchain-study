# 作者：顾涛
# 创建时间：2025/6/29

import os
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

# 定义提示
prompt_template = """针对下面的内容，写一个简洁的总结摘要:
"{text}"
简洁的总结摘要:"""

prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "\n"
    "Given the new context, refine the original summary in Chinese"
    "If the context isn't useful, return the original summary."
)
# refine_template = (
#     "你的工作是做出一个最终的总结摘要。\n"
#     "我们提供了一个到某个点的现有摘要:{existing_answer}\n"
#     "我们有机会完善现有的摘要，基于下面更多的文本内容\n"
#     "------------\n"
#     "{text}\n"
#     "------------\n"
# )

refine_prompt = PromptTemplate.from_template(refine_template)

chain = load_summarize_chain(
    llm=model,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=False,
    input_key="input_documents",
    output_key="output_text",
)

result = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)
print(result['output_text'])
