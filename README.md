# RAG 智能问答系统

> 基于 LangChain + DeepSeek 的检索增强生成（RAG）问答系统，支持多格式文档加载、混合检索、重排序与 Docker 一键部署。

## 📌 项目简介

本项目实现了一套完整的 RAG（Retrieval-Augmented Generation）问答流程，能够上传 PDF/TXT 文档并基于文档内容进行智能问答。适用于企业知识库问答、文档智能检索等场景。

## ✨ 功能亮点

| 特性 | 说明 |
|------|------|
| 📄 **多格式文档加载** | 支持 PDF、TXT 格式文档导入解析 |
| 🔍 **Hybrid Search** | 向量检索 + BM25 关键词检索融合，权重可调 |
| 🎯 **Reranker 重排序** | Cross-Encoder 逐对打分，海选→决赛两阶段检索 |
| 🤖 **Agentic RAG** | LangGraph 驱动的自主检索决策，支持 Self-RAG / Corrective RAG |
| 📊 **评估体系** | RAGAS 框架量化评估（Faithfulness、Answer Relevancy 等指标） |
| 🐳 **Docker 部署** | 一键容器化部署，开箱即用 |
| 💻 **Gradio UI** | 友好的 Web 交互界面，流式输出体验 |

## 🛠️ 技术栈

- **框架：** LangChain + LangGraph
- **向量库：** Chroma（持久化存储）
- **Embedding：** DashScope Embeddings
- **LLM：** DeepSeek Chat
- **检索增强：** Hybrid Search（Vector + BM25）+ Reranker（bge-reranker-v2-m3）
- **评估：** RAGAS
- **前端：** Gradio
- **部署：** Docker

## 🚀 快速开始

### 方式一：Docker 部署（推荐）

```bash
# 克隆仓库
git clone https://github.com/<你的用户名>/rag-project.git
cd rag-project

# 启动服务
docker-compose up
```

访问 `http://localhost:7860` 即可使用。

### 方式二：本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行
python rag_demo.py
```

## 📁 项目结构

```
rag-project/
├── rag_demo.py              # 主程序入口
├── requirements.txt         # Python 依赖
├── Dockerfile               # Docker 构建文件
├── docker-compose.yml       # Docker Compose 配置
├── models/                  # 本地模型
│   └── bge-reranker/       # Reranker 重排序模型
├── chroma_db/              # Chroma 向量库持久化数据
└── README.md               # 本文件
```

## 📖 使用说明

1. **上传文档：** 在 Web 界面上传 PDF 或 TXT 文件
2. **自动处理：** 系统自动完成文档加载→分块→向量化存储
3. **提问：** 输入问题，系统检索相关文档内容并生成回答
4. **流式输出：** 答案逐 token 流式展示，响应迅速

### 分块策略

- 块大小（chunk_size）：500
- 重叠（overlap）：100

### 检索流程

1. **召回阶段：** Hybrid Search 从向量库和 BM25 索引中混合检索
2. **精排阶段：** Reranker Cross-Encoder 对候选文档逐对打分
3. **生成阶段：** 结合精排结果与原始问题，LLM 生成最终答案

## 📊 评估结果

基于 RAGAS 框架的量化评估指标：

| 指标 | 说明 |
|------|------|
| Faithfulness | 答案忠实于检索内容 |
| Answer Relevancy | 答案与问题的相关性 |
| Context Precision | 检索结果精确度 |
| Context Recall | 检索结果召回率 |

（具体数值因文档和测试集而异）

## 📌 后续方向

- [ ] Dify/Coze 低代码平台集成
- [ ] 支持更多文档格式（Docx、Markdown、HTML）
- [ ] 多轮对话记忆
- [ ] Web 端搜索增强

## 📄 License

MIT
"# rag_project" 
