import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
import gradio as gr
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.bm25 import BM25Retriever
# 加载 .env
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 设置 DeepSeek
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

# 1. 加载文档
if os.path.exists("RAG_Guide.pdf"):
    loader = PyPDFLoader("RAG_Guide.pdf")
    documents = loader.load()
else:
    loader = TextLoader("test.txt")
    documents = loader.load()

# 2. 切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3. 向量化存储
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]



chorma_db = './chroma_db'
if os.path.exists(chorma_db):
    #已存在
    vectorstore = Chroma(persist_directory=chorma_db, embedding_function=embeddings)
    print('加载已有向量库')
    #不存在
else:
    vectorstore = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory=chorma_db)
    print('已保存向量库')

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

bm25_retriever = BM25Retriever.from_texts(texts)
bm25_retriever.k = 3
retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
)
# Reranker 重排序
reranker = HuggingFaceCrossEncoder(model_name="./models/bge-reranker")

def retrieve_and_rerank(query):
    docs = retriever.invoke(query)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.score(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:3]]

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# 4. 定义 Prompt 模板
# ========== Phase 2: Agentic RAG ==========

# --- Agent 1: Query Expansion（多查询扩写）---
query_expansion_template = """你是一个问答助手。
用户的问题是：{question}
请生成3个不同的搜索查询，从不同角度来搜索相关信息。
每个查询需要严格联系问题，不要重复。

输出3个查询，每行一个："""

query_prompt = ChatPromptTemplate.from_template(query_expansion_template)

def expand_queries(question):
    messages = query_prompt.format_messages(question=question)
    response = llm.invoke(messages)
    queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    return [question] + queries[:3]    # 原问题 + 3个新的

# 5. 创建 LLM
llm = ChatOpenAI(model="deepseek-chat", temperature=0)

# 6. 用 LCEL 构建 RAG 链（新版写法）
rag_chain = (
    {"context": RunnableLambda(retrieve_and_rerank) | format_docs, "question": RunnablePassthrough()}
    | query_prompt
    | llm
)

# 7. 开问
# 问答函数
def ask(question, history):
    context_docs = retrieve_and_rerank(question)
    context = format_docs(context_docs)

    messages = prompt.format_messages(context=context, question=question)

    response = ""
    for chunk in llm.stream(messages):
        response += chunk.content
        yield response
# 启动网页
gr.ChatInterface(ask, title="📚 文档问答助手").launch()