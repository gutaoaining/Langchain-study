# 作者：顾涛
# 创建时间：2025/6/27
from langserve import RemoteRunnable

if __name__ == '__main__':
    client = RemoteRunnable("http://localhost:8000/chainDemo/")
    print(client.invoke({"language": "English", "text": "你好！"}))
