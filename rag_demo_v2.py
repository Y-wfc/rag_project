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
# Reranker 重排序（自动下载不存在时自动从 HuggingFace 拉取）
import os as _os
_reranker_model = _os.getenv("RERANKER_MODEL", "./models/bge-reranker")
if not _os.path.exists(_reranker_model):
    print("⚠️ 本地模型未找到，将从 HuggingFace 下载 BAAI/bge-reranker-v2-m3...")
    _reranker_model = "BAAI/bge-reranker-v2-m3"
reranker = HuggingFaceCrossEncoder(model_name=_reranker_model)

def retrieve_and_rerank(query):
    docs = retriever.invoke(query)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.score(pairs) ##对比对后的文档内容进行打分
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:3]] ##打分取前三

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 4. 创建 LLM（必须在 Phase 2 之前，因为 Phase 2 要用到 llm）
llm = ChatOpenAI(model="deepseek-chat", temperature=0)

# ============================================================
# Phase 2: Agentic RAG（面试核心亮点）
# ============================================================

# --- Agent 1: Query Expansion（多查询扩写）---
query_expansion_template = """你是一个问答助手。
用户的问题是：{question}
请生成3个不同的搜索查询，从不同角度来搜索相关信息。
每个查询需要严格联系问题，不要重复。

输出3个查询，每行一个："""

query_prompt = ChatPromptTemplate.from_template(query_expansion_template)

def expand_queries(question):
    """把用户问题扩写成多个搜索角度"""
    messages = query_prompt.format_messages(question=question)
    response = llm.invoke(messages)
    queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    return [question] + queries[:3]    # 原问题 + 3个新的


# # --- Agent 2: Multi-Query Retrieval（多路检索 + 重排序）---
# def multi_query_retrieve(question):
#     """多查询分别检索 → 合并 → 去重 → 重排序"""
#     # 1. 生成多个搜索查询
#     queries = expand_queries(question)
#
#     # 2. 每个查询都去检索一遍
#     all_docs = []
#     for q in queries:
#         all_docs.extend(retriever.invoke(q))
#
#     # 3. 去重（按内容前100字判断是否重复）
#     seen = set()
#     unique_docs = []
#     for doc in all_docs:
#         key = doc.page_content[:100]
#         if key not in seen:
#             seen.add(key)
#             unique_docs.append(doc)
#
#     # 4. 用 Reranker 重新排序
#     pairs = [[question, doc.page_content] for doc in unique_docs]
#     scores = reranker.score(pairs)
#     scored_docs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
#     return [doc for doc, score in scored_docs[:5]]

# --- Agent 2: Multi-Query Retrieval（多路检索 + 重排序）---
def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = sum(ai * ai for ai in a) ** 0.5
    norm_b = sum(bi * bi for bi in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

def multi_query_retrieve(question):
    # 1. 生成多个搜索查询
    queries = expand_queries(question)

    # 2. 每个查询都去检索一遍
    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))

    # 3. 用 embedding 相似度去重（免费，比前100字更准）
    unique_docs = []
    for doc in all_docs:
        doc_vec = embeddings.embed_query(doc.page_content)
        is_dup = False
        for u in unique_docs:
            u_vec = embeddings.embed_query(u.page_content)
            if cosine_similarity(doc_vec, u_vec) > 0.9:  # 相似度 > 0.9 视为重复
                is_dup = True
                break
        if not is_dup:
            unique_docs.append(doc)

    # 4. 用 Reranker 重新排序
    pairs = [[question, doc.page_content] for doc in unique_docs]
    scores = reranker.score(pairs)
    scored_docs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:5]]


# --- Agent 3: Relevance Judge（文档质量判断）---
judge_template = """你是一个文档审核员。
判断以下"上下文"是否包含足够的信息来回答"问题"。
只回答 yes 或 no。

问题：{question}

上下文（截取部分）：
{context}

相关吗？（yes/no）："""

judge_prompt = ChatPromptTemplate.from_template(judge_template)

def is_relevant(question, docs):
    """判断检索到的文档是否与问题相关"""
    if not docs:
        return False
    context = "\n\n".join([doc.page_content[:300] for doc in docs])
    messages = judge_prompt.format_messages(question=question, context=context)
    response = llm.invoke(messages)
    return response.content.strip().lower().startswith("y")


# --- Agent 4: Adaptive Generation（自适应生成）---
def agentic_ask(question, history):
    """Agentic RAG 主流程"""
    # 步骤1：多路检索
    docs = multi_query_retrieve(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 步骤2：判断文档质量
    relevant = is_relevant(question, docs)

    if relevant:
        # 有相关文档 → 正常回答
        template = """你是一个文档问答助手。
请根据以下上下文回答问题。

上下文：
{context}

问题：
{question}

回答："""
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(context=context, question=question)
    else:
        # 无相关文档 → 诚实告知
        template = """你是一个文档问答助手。
以下文档中没有找到与问题直接相关的内容。
请根据你自己的知识尝试回答，但务必开头说明"文档中未找到相关信息，以下回答来自我的一般知识"。
如果连你也不确定，就说不知道。

问题：
{question}

你的回答："""
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(question=question)

    # 流式输出
    response = ""
    for chunk in llm.stream(messages):
        response += chunk.content
        yield response

# ============================================================
# Phase 1 原有代码（基础 RAG）
# ============================================================

# 5. 定义 Prompt 模板（基础 RAG 用）
template = """你是一个文档问答助手。
请根据以下上下文内容回答问题。如果你不知道答案，就说不知道，不要编造。

上下文：
{context}

问题：
{question}

回答："""
prompt = ChatPromptTemplate.from_template(template)

# 6. 用 LCEL 构建 RAG 链（基础 RAG 用）
rag_chain = (
    {"context": RunnableLambda(retrieve_and_rerank) | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 7. 问答函数（支持模式切换）
def ask(question, history, mode):
    if mode == "Agentic RAG":
        yield from agentic_ask(question, history)
    else:
        # 基础 RAG
        context_docs = retrieve_and_rerank(question)
        context = format_docs(context_docs)
        messages = prompt.format_messages(context=context, question=question)
        response = ""
        for chunk in llm.stream(messages):
            response += chunk.content
            yield response

# 8. 启动网页（带模式切换）
if __name__ == "__main__":
    gr.ChatInterface(
        ask,
        additional_inputs=gr.Radio(["基础 RAG", "Agentic RAG"], label="选择模式", value="Agentic RAG"),
        title="📚 Agentic RAG 问答助手"
    ).launch(server_name="0.0.0.0")