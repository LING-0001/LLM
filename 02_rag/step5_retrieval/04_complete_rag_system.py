#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5.4: 完整的RAG系统
学习目标：整合所有优化，构建生产级RAG类
"""

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os
import time

print("=" * 60)
print("🏗️  完整RAG系统")
print("=" * 60)

# ============================================================
# RAG系统类定义
# ============================================================

class TrafficLawRAG:
    """
    交通法RAG问答系统
    
    整合了所有优化策略：
    - 优化的检索（Top-K + 阈值 + 去重）
    - 优化的Prompt（防幻觉 + 格式化）
    - 异常处理
    - 日志记录
    """
    
    def __init__(
        self,
        db_path,
        embedding_model_name,
        llm_path,
        collection_name="traffic_law"
    ):
        """
        初始化RAG系统
        
        Args:
            db_path: 向量数据库路径
            embedding_model_name: Embedding模型名称
            llm_path: LLM模型路径
            collection_name: 集合名称
        """
        print("\n🚀 初始化RAG系统...")
        
        # 加载向量数据库
        print("   [1/3] 加载向量数据库...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        print(f"         ✅ 数据库包含 {self.collection.count()} 个文档")
        
        # 加载Embedding模型
        print("   [2/3] 加载Embedding模型...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("         ✅ Embedding模型加载完成")
        
        # 加载LLM
        print("   [3/3] 加载LLM...")
        self.llm = Llama(
            model_path=llm_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )
        print("         ✅ LLM加载完成")
        
        print("\n✅ RAG系统初始化完成！\n")
    
    def retrieve(
        self,
        question,
        top_k=10,
        threshold=0.7,
        max_results=3
    ):
        """
        优化的检索函数
        
        Args:
            question: 用户问题
            top_k: 初始检索数量
            threshold: 相似度阈值
            max_results: 最终返回数量
        
        Returns:
            检索到的文档列表
        """
        try:
            # 向量化问题
            question_vector = self.embedding_model.encode(
                [question],
                show_progress_bar=False
            )
            
            # 检索
            results = self.collection.query(
                query_embeddings=question_vector.tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 过滤和格式化
            retrieved_docs = []
            for i in range(len(results['ids'][0])):
                similarity = 1 - results['distances'][0][i]
                
                if similarity >= threshold:
                    retrieved_docs.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'chapter': results['metadatas'][0][i]['chapter'],
                        'similarity': similarity
                    })
            
            # 返回Top-N
            return retrieved_docs[:max_results]
        
        except Exception as e:
            print(f"⚠️  检索错误: {e}")
            return []
    
    def generate(self, question, context):
        """
        基于上下文生成答案
        
        Args:
            question: 用户问题
            context: 检索到的上下文
        
        Returns:
            生成的答案
        """
        # 构建Prompt
        prompt = f"""你是一个专业的交通法规助手，专门解答中国道路交通安全法相关问题。

【参考资料】
{context}

【回答规则】
1. 严格依据参考资料回答，不编造信息
2. 如果参考资料中没有相关内容，明确回答「参考资料中未提及此内容」
3. 回答要准确、简洁、分点列出
4. 保持客观中立的语气

【用户问题】
{question}

【你的回答】
"""
        
        try:
            # 生成答案
            output = self.llm(
                prompt,
                max_tokens=300,
                temperature=0.2,  # 低温度，更确定
                stop=["【", "\n\n\n"],
                echo=False,
                stream=False
            )
            
            answer = output['choices'][0]['text'].strip()
            return answer
        
        except Exception as e:
            print(f"⚠️  生成错误: {e}")
            return "抱歉，生成答案时出错了。"
    
    def query(
        self,
        question,
        top_k=10,
        threshold=0.7,
        max_results=3,
        show_details=False
    ):
        """
        完整的RAG查询流程
        
        Args:
            question: 用户问题
            top_k: 检索数量
            threshold: 相似度阈值
            max_results: 最终文档数
            show_details: 是否显示详细信息
        
        Returns:
            dict: 包含答案和元数据
        """
        start_time = time.time()
        
        # Step 1: 检索
        retrieved_docs = self.retrieve(
            question,
            top_k=top_k,
            threshold=threshold,
            max_results=max_results
        )
        
        retrieve_time = time.time() - start_time
        
        # Step 2: 处理检索结果
        if len(retrieved_docs) == 0:
            return {
                'question': question,
                'answer': "抱歉，我在文档中没有找到相关信息。建议您：\n1. 尝试用不同方式表达问题\n2. 检查问题是否在交通法规范围内",
                'sources': [],
                'retrieve_time': retrieve_time,
                'generate_time': 0,
                'total_time': retrieve_time
            }
        
        # Step 3: 构建上下文
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        
        # Step 4: 生成答案
        generate_start = time.time()
        answer = self.generate(question, context)
        generate_time = time.time() - generate_start
        
        total_time = time.time() - start_time
        
        # 返回结果
        result = {
            'question': question,
            'answer': answer,
            'sources': retrieved_docs,
            'retrieve_time': retrieve_time,
            'generate_time': generate_time,
            'total_time': total_time
        }
        
        # 显示详细信息
        if show_details:
            self._print_result(result)
        
        return result
    
    def _print_result(self, result):
        """打印查询结果"""
        print(f"\n{'='*60}")
        print(f"❓ 问题: {result['question']}")
        print(f"{'='*60}")
        
        # 检索的文档
        print(f"\n📚 检索到 {len(result['sources'])} 个相关文档:")
        for i, doc in enumerate(result['sources'], 1):
            print(f"\n[{i}] 相似度: {doc['similarity']*100:.1f}% | {doc['chapter']}")
            preview = doc['content'][:80] + "..." if len(doc['content']) > 80 else doc['content']
            print(f"    {preview}")
        
        # 答案
        print(f"\n💡 答案:")
        print(f"{'='*60}")
        print(result['answer'])
        print(f"{'='*60}")
        
        # 性能统计
        print(f"\n⏱️  性能:")
        print(f"   • 检索时间: {result['retrieve_time']*1000:.0f}ms")
        print(f"   • 生成时间: {result['generate_time']*1000:.0f}ms")
        print(f"   • 总时间: {result['total_time']:.2f}s")


# ============================================================
# 测试RAG系统
# ============================================================

print("【测试完整RAG系统】")
print("=" * 60)

# 初始化系统
rag_system = TrafficLawRAG(
    db_path="../step4_vectorstore/data/chroma_traffic_law",
    embedding_model_name="shibing624/text2vec-base-chinese",
    llm_path="/Users/a58/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf"
)

# 测试问题
test_questions = [
    "酒驾的处罚是什么？",
    "交通事故后应该怎么处理？",
    "驾驶证扣满12分怎么办？",
]

print(f"\n测试 {len(test_questions)} 个问题:")

for idx, question in enumerate(test_questions, 1):
    print(f"\n\n{'#'*60}")
    print(f"# 测试 {idx}/{len(test_questions)}")
    print(f"{'#'*60}")
    
    result = rag_system.query(
        question,
        top_k=10,
        threshold=0.7,
        max_results=3,
        show_details=True
    )

# ============================================================
# 总结
# ============================================================

print("\n\n" + "=" * 60)
print("🎉 完整RAG系统构建完成！")
print("=" * 60)

print("\n✅ 系统特性:")
print("   1. 面向对象设计，易于复用")
print("   2. 整合所有优化策略")
print("   3. 完善的异常处理")
print("   4. 性能统计")
print("   5. 灵活的参数配置")

print("\n💡 使用方法:")
print("""
# 初始化
rag = TrafficLawRAG(db_path, model_name, llm_path)

# 查询
result = rag.query(
    "你的问题",
    top_k=10,
    threshold=0.7,
    max_results=3,
    show_details=True
)

# 获取答案
answer = result['answer']
sources = result['sources']
""")

print("\n🎯 下一步:")
print("   运行: python 05_interactive_qa.py")
print("   体验交互式问答界面")

print("\n" + "=" * 60)

