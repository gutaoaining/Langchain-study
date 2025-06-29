# 作者：顾涛
# 创建时间：2025/6/28
import os

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'LangChainChatDemo'
# 1、创建模型
model = ChatOpenAI(model='gpt-4-turbo')

# 2、定义模版和提示词
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手,用{language}尽你所能回答所有问题。'),
    MessagesPlaceholder(variable_name='my_msg')
])
# 3、定义链
chain = prompt_template | model

# 4、保存聊天的历史记录
store = {}  # 所有用户的聊天记录都保存到store，key：sessionid，value保存回话


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


config = {'configurable': {'session_id': 'test001'}}

do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg'
)


def chat(chat_message: str):
    print(f'历史输入信息：{store}')

    # 流式输出
    for response in do_message.stream({'my_msg': [HumanMessage(content=chat_message)], 'language': '中文'},
                                      config=config):
        print(response.content, end=' ')


if __name__ == '__main__':
    while True:
        chat_message = input("请输入你的问题：")
        chat(chat_message)
