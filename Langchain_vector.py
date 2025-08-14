# 作者：顾涛
# 创建时间：2025/8/15
import datetime
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "LangchainVector"
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

model = ChatOpenAI(model="gpt-4-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 存储向量数据库的目录
persist_dir = 'chroma_data_dir'

# 一些YouTube的视频连接
urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=rQdibOsL1ps",
    "https://www.youtube.com/watch?v=28lC4fqukoc",
    "https://www.youtube.com/watch?v=es-9MgxB-uc",
    "https://www.youtube.com/watch?v=wLRHwKuKvOE",
    "https://www.youtube.com/watch?v=ObIltMaRJvY",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
]
# document的数组
docs = []
for url in urls:
    # 一个YouTube的视频对应一个document
    docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())

print(len(docs))
print(docs[0])
# 给doc添加额外的元数据： 视频发布的年份
for doc in docs:
    doc.metadata['publish_year'] = int(
        datetime.datetime.strptime(doc.metadata['publish_date'], '%Y-%m-%d %H:%M:%S').strftime('%Y'))
print(docs[0].metadata)
# 第一个视频的字幕内容
print(docs[0].page_content[:500])
# 根据多个doc构建向量数据库
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30)
split_doc = text_splitter.split_documents(docs)
# 向量数据库的持久化
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
