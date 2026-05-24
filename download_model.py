import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
import shutil, glob

# 下载到临时目录
path = snapshot_download(repo_id="BAAI/bge-reranker-v2-m3")
print("下载完成，路径:", path)

# 复制到项目目录下
target = "./models/bge-reranker"
if os.path.exists(target):
    shutil.rmtree(target)
shutil.copytree(path, target)
print("已复制到:", target)