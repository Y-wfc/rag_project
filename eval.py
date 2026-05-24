import sys
import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

from rag_demo_v2 import multi_query_retrieve, llm

# 测试问题
test_questions = [
    "RAG 和微调有什么区别？",
    "什么是向量检索？",
    "Reranker 的作用是什么？",
    "什么时候用 RAG 不用微调？",
]

print("正在跑评估，请稍等...\n")

# ---------- 第1步：跑系统，收集结果 ----------
results = []
for i, q in enumerate(test_questions, 1):
    print(f"[{i}/{len(test_questions)}] 测试问题：{q}")

    docs = multi_query_retrieve(q)
    context = [doc.page_content for doc in docs]

    template = """请根据以下上下文回答问题。

上下文：
{context}

问题：
{question}

回答："""
    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(context="\n\n".join(context), question=q)
    answer = llm.invoke(messages).content

    results.append({
        "question": q,
        "answer": answer,
        "contexts": context
    })

# ---------- 第2步：让 DeepSeek 自己评估 faithfulness（回答是否忠于文档）----------
print("\n正在评估 faithfulness...")
faithfulness_scores = []
for r in results:
    eval_prompt = f"""你是一个评估员。判断以下回答是否忠于提供的上下文（没有编造内容）。

上下文：
{" ".join(r["contexts"][:2])}

回答：
{r["answer"]}

请打分（0-1分），只输出一个数字：
- 1分：完全基于上下文，没有编造
- 0.5分：大部分基于上下文，但有少量推断
- 0分：有明显编造内容

分数："""
    eval_messages = [{"role": "user", "content": eval_prompt}]
    score = llm.invoke(eval_messages).content.strip()
    try:
        faithfulness_scores.append(float(score))
    except:
        faithfulness_scores.append(0.5)
    print(f"  [{r['question'][:20]}...] faithfulness: {faithfulness_scores[-1]}")

# ---------- 第3步：让 DeepSeek 自己评估 answer_relevancy（是否答非所问）----------
print("\n正在评估 answer_relevancy...")
relevancy_scores = []
for r in results:
    eval_prompt = f"""你是一个评估员。判断以下回答是否回答了用户的问题。

问题：
{r["question"]}

回答：
{r["answer"]}

请打分（0-1分），只输出一个数字：
- 1分：完全切题，回答了问题
- 0.5分：部分相关，但没有完全回答问题
- 0分：答非所问

分数："""
    eval_messages = [{"role": "user", "content": eval_prompt}]
    score = llm.invoke(eval_messages).content.strip()
    try:
        relevancy_scores.append(float(score))
    except:
        relevancy_scores.append(0.5)
    print(f"  [{r['question'][:20]}...] answer_relevancy: {relevancy_scores[-1]}")

# ---------- 输出结果 ----------
print("\n" + "=" * 50)
print("评估结果")
print("=" * 50)
avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
avg_rel = sum(relevancy_scores) / len(relevancy_scores)

for i, r in enumerate(results):
    print(f"\n问题：{r['question']}")
    print(f"  faithfulness:     {faithfulness_scores[i]:.2f}")
    print(f"  answer_relevancy: {relevancy_scores[i]:.2f}")
    print(f"  context个数：{len(r['contexts'])}")

print("\n" + "-" * 30)
print(f"平均 faithfulness:     {avg_faith:.2f}")
print(f"平均 answer_relevancy: {avg_rel:.2f}")

if avg_faith < 0.7:
    print("⚠️ faithfulness 偏低 → 加强 prompt 约束")
if avg_rel < 0.7:
    print("⚠️ answer_relevancy 偏低 → 检查检索质量")
print("✅ 评估完成")
