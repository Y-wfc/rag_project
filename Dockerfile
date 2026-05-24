# ============================================
# Dockerfile - RAG Demo 项目
# ============================================
FROM docker.m.daocloud.io/library/python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- 先复制依赖文件，分层缓存 ----
COPY requirements.txt .

# 装核心依赖 + torch(cpu) + sentence-transformers（代码改动不影响这层缓存）
RUN pip install --no-cache-dir --default-timeout=120 --timeout=120 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt \
    && pip install --no-cache-dir --default-timeout=120 \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --default-timeout=120 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    sentence-transformers==5.5.0

# 创建 models 目录 + 测试文件
RUN mkdir -p models && \
    echo "这是一个测试文档，用于验证 RAG 知识库问答系统的功能。" > /app/test.txt

# ---- 最后复制代码（这层会因代码改动而失效，但上面 pip 层不受影响）----
COPY . .

# Gradio 端口
EXPOSE 7860

# 启动命令
CMD ["python", "-u", "/app/rag_demo_v2.py"]
