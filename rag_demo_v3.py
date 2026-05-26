import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.bm25 import BM25Retriever
from langgraph.graph import StateGraph, END
from typing import List, TypedDict, Literal
import gradio as gr

# ============================================================
# 配置
# ============================================================
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

# ============================================================
# Phase 0 — 文档加载 + 向量库（同 v2）
# ============================================================
if os.path.exists("RAG_Guide.pdf"):
    loader = PyPDFLoader("RAG_Guide.pdf")
    documents = loader.load()
else:
    loader = TextLoader("test.txt")
    documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = DashScopeEmbeddings(model="text-embedding-v2")
texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]

chorma_db = './chroma_db'
if os.path.exists(chorma_db):
    vectorstore = Chroma(persist_directory=chorma_db, embedding_function=embeddings)
    print('✅ 加载已有向量库')
else:
    vectorstore = Chroma.from_texts(texts, embeddings, metadatas=metadatas,
                                    persist_directory=chorma_db)
    print('✅ 已保存向量库')

# ============================================================
# Phase 1 — Hybrid Search + Reranker（同 v2）
# ============================================================
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_texts(texts)
bm25_retriever.k = 3

retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
)

_reranker_model = os.getenv("RERANKER_MODEL", "./models/bge-reranker")
if not os.path.exists(_reranker_model):
    print("⚠️ 本地模型未找到，将从 HuggingFace 下载 BAAI/bge-reranker-v2-m3...")
    _reranker_model = "BAAI/bge-reranker-v2-m3"
reranker = HuggingFaceCrossEncoder(model_name=_reranker_model)

llm = ChatOpenAI(model="deepseek-chat", temperature=0)


# ============================================================
# Phase 2 — ⭐ LangGraph Agentic RAG ⭐
# ============================================================

# --- 工具函数 ---
def cosine_similarity(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = sum(ai * ai for ai in a) ** 0.5
    norm_b = sum(bi * bi for bi in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0


# --- State 定义 ---
class AgentState(TypedDict):
    question: str
    queries: List[str]
    documents: List[Document]
    context: str
    is_relevant: bool
    retry_count: int
    generation: str


# --- Node 1: 查询扩展 + 多路检索 ---
query_expansion_template = """你是一个问答助手。
用户的问题是：{question}
请生成3个不同的搜索查询，从不同角度来搜索相关信息。
每个查询需要严格联系问题，不要重复。

输出3个查询，每行一个："""
query_prompt = ChatPromptTemplate.from_template(query_expansion_template)


def expand_queries(question: str) -> List[str]:
    messages = query_prompt.format_messages(question=question)
    response = llm.invoke(messages)
    queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    return [question] + queries[:3]


def retrieve_node(state: AgentState) -> AgentState:
    """检索节点：扩写查询 → 多路检索 → embedding去重 → Reranker"""
    # 如果是 Corrective RAG 的重新检索，递增 retry_count
    retry_count = state["retry_count"]
    if state["context"]:  # 已有检索结果 → 这是重试
        retry_count += 1
    print(f"  🔍 检索节点 (retry #{retry_count})")

    # 1. 扩写查询
    queries = expand_queries(state["question"])

    # 2. 每个查询都去检索
    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))

    # 3. embedding 相似度去重
    unique_docs = []
    for doc in all_docs:
        doc_vec = embeddings.embed_query(doc.page_content)
        is_dup = False
        for u in unique_docs:
            u_vec = embeddings.embed_query(u.page_content)
            if cosine_similarity(doc_vec, u_vec) > 0.9:
                is_dup = True
                break
        if not is_dup:
            unique_docs.append(doc)

    # 4. Reranker 排序
    pairs = [[state["question"], doc.page_content] for doc in unique_docs]
    scores = reranker.score(pairs)
    scored_docs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:5]]

    return {
        **state,
        "queries": queries,
        "documents": top_docs,
        "context": "\n\n".join([doc.page_content for doc in top_docs]),
        "retry_count": retry_count,
    }


# --- Node 2: 相关性打分（Relevance Judge）---
judge_template = """你是一个文档审核员。
判断以下"上下文"是否包含足够的信息来回答"问题"。
只回答 yes 或 no。

问题：{question}

上下文（截取部分）：
{context}

相关吗？（yes/no）："""
judge_prompt = ChatPromptTemplate.from_template(judge_template)


def grade_node(state: AgentState) -> AgentState:
    """打分节点：判断检索结果是否足够回答问题"""
    print(f"  📐 打分节点")

    docs = state["documents"]
    if not docs:
        return {**state, "is_relevant": False}

    context = "\n\n".join([doc.page_content[:300] for doc in docs])
    messages = judge_prompt.format_messages(question=state["question"], context=context)
    response = llm.invoke(messages)
    relevant = response.content.strip().lower().startswith("y")

    print(f"  → 结果相关性: {'✅ 相关' if relevant else '❌ 不相关'}")
    return {**state, "is_relevant": relevant}


# --- 路由函数（条件边） ---
def route_after_grade(state: AgentState) -> Literal["generate", "fallback", "retrieve"]:
    """根据打分结果决定下一步"""
    if state["is_relevant"]:
        return "generate"
    elif state["retry_count"] < 1:
        return "retrieve"
    else:
        return "fallback"


# --- Node 3: 正常回答生成 ---
def generate_node(state: AgentState) -> AgentState:
    """生成节点：基于检索结果回答"""
    print(f"  ✍️ 生成节点（有文档）")

    template = """你是一个文档问答助手。
请根据以下上下文回答问题。

上下文：
{context}

问题：
{question}

回答："""
    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(
        context=state["context"], question=state["question"]
    )

    response = ""
    for chunk in llm.stream(messages):
        response += chunk.content

    return {**state, "generation": response}


# --- Node 4: 兜底回答生成 ---
def fallback_node(state: AgentState) -> AgentState:
    """兜底节点：没有相关文档时的回答"""
    print(f"  ✍️ 生成节点（无文档/兜底）")

    template = """你是一个文档问答助手。
以下文档中没有找到与问题直接相关的内容。
请根据你自己的知识尝试回答，但务必开头说明"文档中未找到相关信息，以下回答来自我的一般知识"。
如果连你也不确定，就说不知道。

问题：
{question}

你的回答："""
    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(question=state["question"])

    response = ""
    for chunk in llm.stream(messages):
        response += chunk.content

    return {**state, "generation": response}


# --- 构建 LangGraph ---
def build_agentic_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    # 设置入口
    graph.set_entry_point("retrieve")

    # 固定边
    graph.add_edge("retrieve", "grade")

    # 条件边：根据相关性路由
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "fallback": "fallback",
            "retrieve": "retrieve",  # 不相关 → 重新检索（Corrective RAG）
        },
    )

    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


agentic_graph = build_agentic_graph()


# --- 面向 Gradio 的封装（流式输出）---
def agentic_ask(question: str, history):
    """LangGraph Agentic RAG 主入口"""
    print(f"\n{'='*50}")
    print(f"🤖 LangGraph Agentic RAG 开始")
    print(f"❓ 问题: {question}")
    print(f"{'='*50}")

    # 运行图
    result = agentic_graph.invoke({
        "question": question,
        "queries": [],
        "documents": [],
        "context": "",
        "is_relevant": False,
        "retry_count": 0,
        "generation": "",
    })

    # 流式输出给 Gradio
    full_response = result["generation"]
    for i in range(1, len(full_response) + 1):
        yield full_response[:i]


# ============================================================
# 基础 RAG（Phase 0+1，保留作为对照）
# ============================================================
def retrieve_and_rerank(query):
    docs = retriever.invoke(query)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.score(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:3]]


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


template = """你是一个文档问答助手。
请根据以下上下文内容回答问题。如果你不知道答案，就说不知道，不要编造。

上下文：
{context}

问题：
{question}

回答："""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": RunnableLambda(retrieve_and_rerank) | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)


# ============================================================
# 模式切换
# ============================================================
def ask(question, history, mode):
    if mode == "基础 RAG":
        context_docs = retrieve_and_rerank(question)
        context = format_docs(context_docs)
        messages = prompt.format_messages(context=context, question=question)
        response = ""
        for chunk in llm.stream(messages):
            response += chunk.content
            yield response

    else:  # LangGraph Agentic RAG
        yield from agentic_ask(question, history)


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    gr.ChatInterface(
        ask,
        additional_inputs=gr.Radio([
            "LangGraph Agentic RAG",
            "基础 RAG",
        ], label="选择模式", value="LangGraph Agentic RAG"),
        title="🤖 LangGraph Agentic RAG 问答助手"
    ).launch(server_name="0.0.0.0")
