# 作者：顾涛
# 创建时间：2025/8/9
from operator import itemgetter

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
HOST_NAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'mytest'
USER_NAME = os.getenv('USER_NAME')
PASSWORD = os.getenv('MYSQL_PASSWORD')

MYSQL_URL = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USER_NAME, PASSWORD, HOST_NAME, PORT, DATABASE)
print(MYSQL_URL)
db = SQLDatabase.from_uri(MYSQL_URL)

model = ChatOpenAI(model='gpt-3.5-turbo')
test_chain = create_sql_query_chain(model, db)
answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、SQL语句和SQL执行后的结果，回答用户问题。
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    回答: """
)

# 创建一个执行的sql语句的工具
execute_sql_tool = QuerySQLDatabaseTool(db=db)

# 1、生成SQL，
# 2、执行SQL
# 3、模板
chain = (RunnablePassthrough.assign(query=test_chain).assign(result=itemgetter('query') | execute_sql_tool)
         | answer_prompt
         | model
         | StrOutputParser()
         )
rep = chain.invoke(input={'question': '请问：分数表中有多少条数据？'})
print(rep)