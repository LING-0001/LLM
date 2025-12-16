"""
练习3：文本相似度应用
实战：使用Embedding进行文本匹配、去重、分类等任务
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("="*70)
print(" "*20 + "文本相似度应用")
print("="*70)
print()

# 加载模型
print("正在加载模型...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("✅ 模型加载完成\n")

# 应用1：智能问答匹配
print("="*70)
print("应用1：智能问答匹配（FAQ系统）")
print("="*70)
print()

# 知识库（常见问题及答案）
faq_database = [
    {
        "question": "如何安装Python",
        "answer": "访问python.org下载安装包，运行安装程序即可。"
    },
    {
        "question": "Python怎么定义函数",
        "answer": "使用def关键字，例如：def my_function(): pass"
    },
    {
        "question": "什么是列表推导式",
        "answer": "列表推导式是Python的一种简洁语法：[x*2 for x in range(10)]"
    },
    {
        "question": "如何读取文件",
        "answer": "使用open()函数：with open('file.txt', 'r') as f: content = f.read()"
    },
    {
        "question": "Python的优点有哪些",
        "answer": "简单易学、库丰富、社区活跃、应用广泛。"
    },
]

# 预先计算知识库的向量
print("正在为FAQ知识库生成向量...")
faq_questions = [item["question"] for item in faq_database]
faq_vectors = model.encode(faq_questions)
print(f"✅ 已为 {len(faq_database)} 个问题生成向量\n")

def find_answer(user_query, threshold=0.5):
    """根据用户问题找到最匹配的答案"""
    # 把用户问题转成向量
    query_vector = model.encode(user_query)
    
    # 计算与知识库的相似度
    similarities = cosine_similarity([query_vector], faq_vectors)[0]
    
    # 找到最相似的
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    if best_score >= threshold:
        return faq_database[best_idx], best_score
    else:
        return None, best_score

# 测试不同的用户问题
test_queries = [
    "Python如何安装",        # 相似问题
    "怎么写函数",            # 相似问题
    "Python有什么好处",      # 相似问题
    "如何学习机器学习",      # 不在知识库
]

print("测试FAQ匹配：")
print()

for query in test_queries:
    print(f"用户问：{query}")
    result, score = find_answer(query, threshold=0.5)
    
    if result:
        print(f"  ✅ 匹配成功 (相似度: {score:.3f})")
        print(f"  📌 匹配到的问题：{result['question']}")
        print(f"  💬 答案：{result['answer']}")
    else:
        print(f"  ❌ 未找到匹配 (最高相似度: {score:.3f})")
        print(f"  💡 建议：转人工客服")
    print()

# 应用2：文本去重
print("="*70)
print("应用2：文本去重")
print("="*70)
print()

documents = [
    "Python是一种编程语言",
    "Python是一门编程语言",  # 几乎相同
    "Java也是编程语言",
    "机器学习很有趣",
    "Python是一种高级语言",  # 相似
    "我喜欢吃苹果",
]

print("原始文档：")
for i, doc in enumerate(documents, 1):
    print(f"  {i}. {doc}")
print()

# 生成向量
doc_vectors = model.encode(documents)

# 计算相似度矩阵
sim_matrix = cosine_similarity(doc_vectors)

# 去重（相似度>0.8认为是重复）
threshold = 0.8
duplicates = []
unique_indices = []

for i in range(len(documents)):
    is_duplicate = False
    for j in unique_indices:
        if sim_matrix[i][j] > threshold:
            duplicates.append((i, j, sim_matrix[i][j]))
            is_duplicate = True
            break
    if not is_duplicate:
        unique_indices.append(i)

print(f"去重结果（阈值={threshold}）：")
print()
print("保留的文档：")
for idx in unique_indices:
    print(f"  ✅ {documents[idx]}")

if duplicates:
    print()
    print("检测到的重复：")
    for i, j, score in duplicates:
        print(f"  ❌ '{documents[i]}' 与 '{documents[j]}' 相似度 {score:.3f}")
print()

# 应用3：文本分类
print("="*70)
print("应用3：零样本文本分类")
print("="*70)
print()

# 类别定义
categories = [
    "编程技术",
    "生活日常",
    "科学研究",
    "娱乐休闲",
]

# 待分类的文本
texts_to_classify = [
    "学习Python很有用",
    "今天天气真好",
    "深度学习算法",
    "周末去看电影",
    "如何写好代码",
    "晚餐吃什么",
]

print("分类标签：", categories)
print()

# 生成类别向量
category_vectors = model.encode(categories)

# 分类
print("分类结果：")
print()

for text in texts_to_classify:
    text_vector = model.encode(text)
    similarities = cosine_similarity([text_vector], category_vectors)[0]
    
    best_category_idx = similarities.argmax()
    best_category = categories[best_category_idx]
    confidence = similarities[best_category_idx]
    
    print(f"  '{text}'")
    print(f"    → {best_category} (置信度: {confidence:.3f})")
    
    # 显示所有类别的得分
    print(f"    详细: ", end="")
    for cat, sim in zip(categories, similarities):
        print(f"{cat}:{sim:.2f} ", end="")
    print("\n")

# 应用4：语义搜索
print("="*70)
print("应用4：语义搜索引擎")
print("="*70)
print()

# 文档库
documents_db = [
    "Python是一种解释型、面向对象的编程语言",
    "机器学习是人工智能的一个分支",
    "深度学习使用神经网络处理数据",
    "自然语言处理让计算机理解人类语言",
    "计算机视觉使机器能够识别图像",
    "强化学习通过奖励机制训练模型",
    "数据科学结合统计和编程分析数据",
    "Web开发包括前端和后端技术",
]

print("文档库：")
for i, doc in enumerate(documents_db, 1):
    print(f"  {i}. {doc}")
print()

# 预计算向量
db_vectors = model.encode(documents_db)

def semantic_search(query, top_k=3):
    """语义搜索"""
    query_vector = model.encode(query)
    similarities = cosine_similarity([query_vector], db_vectors)[0]
    
    # 排序
    ranked_indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for idx in ranked_indices:
        results.append({
            "doc": documents_db[idx],
            "score": similarities[idx],
            "rank": len(results) + 1
        })
    return results

# 测试搜索
search_queries = [
    "AI和机器学习",
    "编程语言",
    "如何让电脑看懂图片",
]

for query in search_queries:
    print(f"搜索：{query}")
    results = semantic_search(query, top_k=3)
    
    for res in results:
        emoji = "🥇" if res['rank'] == 1 else "🥈" if res['rank'] == 2 else "🥉"
        print(f"  {emoji} [{res['score']:.3f}] {res['doc']}")
    print()

print("💡 观察：")
print("   • '让电脑看懂图片' 能匹配到 '计算机视觉'")
print("   • 不需要关键词完全匹配")
print("   • 这就是语义搜索的强大之处！")
print()

# 应用5：相似度阈值选择
print("="*70)
print("应用5：如何选择相似度阈值")
print("="*70)
print()

test_pairs = [
    ("Python很好用", "Python非常实用"),
    ("Python很好用", "Java也很强"),
    ("Python很好用", "今天天气好"),
    ("北京是首都", "北京在中国"),
    ("北京是首都", "上海是城市"),
]

print("不同文本对的相似度：")
print()

for text1, text2 in test_pairs:
    vec1 = model.encode(text1)
    vec2 = model.encode(text2)
    sim = cosine_similarity([vec1], [vec2])[0][0]
    
    if sim > 0.7:
        level = "高度相似"
        emoji = "✅"
    elif sim > 0.5:
        level = "中度相关"
        emoji = "⚠️"
    else:
        level = "不太相关"
        emoji = "❌"
    
    print(f"  {emoji} [{sim:.3f}] {level}")
    print(f"     '{text1}' vs '{text2}'")
    print()

print("💡 阈值建议：")
print("   • > 0.8: 几乎相同（去重）")
print("   • 0.6-0.8: 高度相关（推荐）")
print("   • 0.4-0.6: 中度相关（可考虑）")
print("   • < 0.4: 不太相关（忽略）")
print()

print("="*70)
print("✅ 练习3完成！")
print()
print("💡 关键收获：")
print("   • FAQ智能匹配：提高客服效率")
print("   • 文本去重：清理重复数据")
print("   • 零样本分类：无需训练数据")
print("   • 语义搜索：理解用户意图")
print("   • 阈值选择：根据应用场景调整")
print()
print("📍 下一步：python 04_batch_embedding.py")
print("   学习高效处理大量文本！")
print("="*70)

