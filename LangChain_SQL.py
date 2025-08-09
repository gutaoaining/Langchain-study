# 作者：顾涛
# 创建时间：2025/8/9
from langchain_community.utilities import SQLDatabase
import os

HOST_NAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'mytest'
USER_NAME = os.getenv('USER_NAME')
PASSWORD = os.getenv('MYSQL_PASSWORD')


MYSQL_URL = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USER_NAME, PASSWORD, HOST_NAME, PORT, DATABASE)
print(MYSQL_URL)
db = SQLDatabase.from_uri(MYSQL_URL)

print(db.get_usable_table_names())
print(db.run('select * from score limit 20;'))
