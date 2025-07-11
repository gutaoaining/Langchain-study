# 作者：顾涛
# 创建时间：2025/7/6
import os
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'LangChainRagDemo'

model = ChatOpenAI(model='gpt-4-turbo')
# 1、加载数据，博客的内容数据
loader = WebBaseLoader(
    web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content'))
    )
)
docs = loader.load()
print(len(docs))
print(docs)

# 2、大文本的切割
# text = "hello world, how about you? thanks, I am fine.  the machine learning class. So what I wanna do today is just spend a little time going over the logistics of the class, and then we'll start to talk a bit about machine learning"

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)
# for s in res:
#     print(s, end='\n')

# 3、存储
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 4、检索器
retriever = vectorstore.as_retriever()

# 创建一个问题的模版
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
don't know. Use three sentences maximum and keep the answer concise.\n

{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        # MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
# 得到chain
chain1 = create_stuff_documents_chain(model, prompt)

# chain2 = create_retrieval_chain(retriever, chain1)
# resp = chain2.invoke({'input': 'What is Task Decomposition?'})
# print(resp['answer'])

# 创建一个子链
# 子链的提示模板
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, 
formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# 创建一个子链
history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

# 保持回答的历史记录
store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 创建父链，把两个链整合
chain = create_retrieval_chain(history_chain, chain1)
result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)


def chat(session_id: str, question: str):
    # 第一轮对话
    resp1 = result_chain.invoke(
        {'input': question},
        config={'configurable': {'session_id': session_id}}
    )
    print('问题回答：', resp1['answer'])


if __name__ == '__main__':
    session_id = input("请输入你的登录id：")
    while True:
        question = input("请输入你的问题：")
        chat(session_id, question)
