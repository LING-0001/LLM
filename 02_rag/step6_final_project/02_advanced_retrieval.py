#!/usr/bin/env python3
"""
RAG最终项目 - 高级检索策略

功能：
1. 混合检索（向量 + 关键词）
2. 结果重排序（Reranking）
3. 上下文窗口优化
4. 性能对比

这是提升RAG准确率的关键技术
"""

import os
import time
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple


class AdvancedRetriever:
    """高级检索器：实现多种检索策略"""
    
    def __init__(self, 
                 chroma_path: str = "./data/document_store",
                 collection_name: str = "documents"):
        """初始化检索器"""
        print("📦 加载向量模型...")
        self.embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
        # 设置归一化：输出的向量自动L2归一化到单位长度
        self.embedding_model.encode_kwargs = {'normalize_embeddings': True}
        
        print(f"💾 连接文档库...")
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✅ 检索器初始化完成！文档块数：{self.collection.count()}\n")
        except:
            print(f"❌ 文档库不存在，请先运行 01_document_manager.py")
            raise
    
    def vector_search(self, 
                     query: str, 
                     n_results: int = 10) -> List[Dict[str, Any]]:
        """
        纯向量检索（基础方法）
        
        Args:
            query: 查询文本
            n_results: 返回结果数
            
        Returns:
            检索结果列表
        """
        start_time = time.time()
        
        # 生成查询向量（已归一化）
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 向量检索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            
            # 关键修正：归一化向量的L2距离 转 余弦相似度
            # 对于归一化向量: cosine_similarity = 1 - (L2_distance^2 / 2)
            # 简化：当向量已归一化，L2距离很小时，cos_sim ≈ 1 - distance/2
            # 但ChromaDB返回的是平方距离，所以：
            import math
            # 如果distance是L2距离的平方，则：cos_sim = 1 - distance/2
            # 如果distance是L2距离，则：cos_sim = 1 - distance^2/2
            
            # 保守处理：将距离映射到[0,1]
            # 对于归一化向量，L2距离范围是[0, 2]（最大是反向）
            # 相似度 = (2 - distance) / 2 = 1 - distance/2
            similarity = max(0, min(1, 1 - distance / 2))
            
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'distance': distance,
                'similarity': similarity,
                'metadata': results['metadatas'][0][i],
                'method': 'vector_only'
            })
        
        elapsed = time.time() - start_time
        return formatted_results, elapsed
    
    def keyword_search(self, 
                       query: str, 
                       n_results: int = 10) -> List[Dict[str, Any]]:
        """
        关键词检索（基于文本匹配）
        
        Args:
            query: 查询文本
            n_results: 返回结果数
            
        Returns:
            检索结果列表
        """
        start_time = time.time()
        
        # 获取所有文档
        all_docs = self.collection.get()
        
        # 计算关键词匹配分数
        results_with_score = []
        query_lower = query.lower()
        query_chars = set(query)
        
        for i, doc in enumerate(all_docs['documents']):
            doc_lower = doc.lower()
            
            # 计算匹配分数
            score = 0
            
            # 1. 完全匹配（最高分）
            if query in doc:
                score += 100
            
            # 2. 包含所有查询字符
            doc_chars = set(doc)
            char_overlap = len(query_chars & doc_chars) / len(query_chars)
            score += char_overlap * 50
            
            # 3. 查询词出现次数
            for char in query:
                score += doc.count(char) * 2
            
            if score > 0:
                results_with_score.append({
                    'id': all_docs['ids'][i],
                    'document': doc,
                    'score': score,
                    'similarity': min(score / 100, 1.0),  # 归一化到0-1
                    'metadata': all_docs['metadatas'][i],
                    'method': 'keyword_only'
                })
        
        # 按分数排序
        results_with_score.sort(key=lambda x: x['score'], reverse=True)
        
        elapsed = time.time() - start_time
        return results_with_score[:n_results], elapsed
    
    def hybrid_search(self, 
                     query: str, 
                     n_results: int = 10,
                     vector_weight: float = 0.7,
                     keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        混合检索（向量 + 关键词）
        
        Args:
            query: 查询文本
            n_results: 返回结果数
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            
        Returns:
            检索结果列表
        """
        start_time = time.time()
        
        # 1. 分别执行两种检索
        vector_results, _ = self.vector_search(query, n_results=20)
        keyword_results, _ = self.keyword_search(query, n_results=20)
        
        # 2. 合并结果
        all_results = {}
        
        # 添加向量检索结果
        for result in vector_results:
            doc_id = result['id']
            all_results[doc_id] = {
                'id': doc_id,
                'document': result['document'],
                'metadata': result['metadata'],
                'vector_score': result['similarity'],
                'keyword_score': 0,
                'method': 'hybrid'
            }
        
        # 添加关键词检索结果
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in all_results:
                all_results[doc_id]['keyword_score'] = result['similarity']
            else:
                all_results[doc_id] = {
                    'id': doc_id,
                    'document': result['document'],
                    'metadata': result['metadata'],
                    'vector_score': 0,
                    'keyword_score': result['similarity'],
                    'method': 'hybrid'
                }
        
        # 3. 计算混合分数
        for doc_id, result in all_results.items():
            result['hybrid_score'] = (
                result['vector_score'] * vector_weight +
                result['keyword_score'] * keyword_weight
            )
            result['similarity'] = result['hybrid_score']
        
        # 4. 排序并返回
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )
        
        elapsed = time.time() - start_time
        return sorted_results[:n_results], elapsed
    
    def rerank_results(self, 
                      query: str,
                      results: List[Dict[str, Any]],
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        重排序：使用交叉编码器重新排序结果
        
        这里使用简化版本：基于查询与文档的详细匹配度
        
        Args:
            query: 查询文本
            results: 初始检索结果
            top_k: 返回前K个结果
            
        Returns:
            重排序后的结果
        """
        start_time = time.time()
        
        # 计算重排序分数
        for result in results:
            doc = result['document']
            
            # 基于多个因素计算新分数
            rerank_score = result.get('similarity', 0.5) * 0.5  # 原始分数占50%
            
            # 1. 查询词完全匹配（+30%）
            if query in doc:
                rerank_score += 0.3
            
            # 2. 查询词字符覆盖率（+20%）
            query_chars = set(query)
            doc_chars = set(doc)
            overlap = len(query_chars & doc_chars) / len(query_chars)
            rerank_score += overlap * 0.2
            
            result['rerank_score'] = min(rerank_score, 1.0)
        
        # 重新排序
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        elapsed = time.time() - start_time
        
        # 标记为重排序结果
        for result in results[:top_k]:
            result['method'] = result.get('method', 'unknown') + '+rerank'
            result['similarity'] = result['rerank_score']
        
        return results[:top_k], elapsed
    
    def search_with_context(self, 
                           query: str,
                           n_results: int = 5,
                           context_window: int = 1) -> List[Dict[str, Any]]:
        """
        带上下文窗口的检索
        
        获取匹配块及其前后相邻的块，提供更完整的上下文
        
        Args:
            query: 查询文本
            n_results: 返回结果数
            context_window: 上下文窗口大小（前后各N块）
            
        Returns:
            包含上下文的检索结果
        """
        start_time = time.time()
        
        # 1. 先进行混合检索
        results, _ = self.hybrid_search(query, n_results=n_results)
        
        # 2. 为每个结果添加上下文
        for result in results:
            metadata = result['metadata']
            doc_name = metadata.get('doc_name')
            chunk_index = metadata.get('chunk_index')
            chunk_total = metadata.get('chunk_total')
            
            if doc_name is None or chunk_index is None:
                result['context_before'] = []
                result['context_after'] = []
                continue
            
            # 获取前后文档块
            context_before = []
            context_after = []
            
            # 获取前面的块
            for i in range(max(0, chunk_index - context_window), chunk_index):
                ctx_results = self.collection.get(
                    where={
                        "$and": [
                            {"doc_name": doc_name},
                            {"chunk_index": i}
                        ]
                    }
                )
                if ctx_results['documents']:
                    context_before.append(ctx_results['documents'][0])
            
            # 获取后面的块
            for i in range(chunk_index + 1, min(chunk_total, chunk_index + context_window + 1)):
                ctx_results = self.collection.get(
                    where={
                        "$and": [
                            {"doc_name": doc_name},
                            {"chunk_index": i}
                        ]
                    }
                )
                if ctx_results['documents']:
                    context_after.append(ctx_results['documents'][0])
            
            result['context_before'] = context_before
            result['context_after'] = context_after
            result['full_context'] = ''.join(context_before) + result['document'] + ''.join(context_after)
        
        elapsed = time.time() - start_time
        return results, elapsed


def demo():
    """演示高级检索策略"""
    print("=" * 60)
    print("RAG最终项目 - 高级检索策略演示")
    print("=" * 60)
    
    # 初始化检索器
    retriever = AdvancedRetriever()
    
    # 测试查询
    test_queries = [
        "醉驾的处罚是什么",
        "工作时间",
        "加班费"
    ]
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"🔍 查询: {query}")
        print("=" * 60)
        
        # 1. 纯向量检索
        print("\n📊 方法1: 纯向量检索")
        print("-" * 50)
        vector_results, vector_time = retriever.vector_search(query, n_results=3)
        for i, result in enumerate(vector_results, 1):
            print(f"\n结果 {i} (相似度: {result['similarity']:.1%})")
            print(f"来源: {result['metadata'].get('doc_name', 'unknown')}")
            print(f"内容: {result['document'][:100]}...")
        print(f"\n⏱️  耗时: {vector_time*1000:.1f}ms")
        
        # 2. 纯关键词检索
        print("\n📊 方法2: 纯关键词检索")
        print("-" * 50)
        keyword_results, keyword_time = retriever.keyword_search(query, n_results=3)
        for i, result in enumerate(keyword_results, 1):
            print(f"\n结果 {i} (相似度: {result['similarity']:.1%})")
            print(f"来源: {result['metadata'].get('doc_name', 'unknown')}")
            print(f"内容: {result['document'][:100]}...")
        print(f"\n⏱️  耗时: {keyword_time*1000:.1f}ms")
        
        # 3. 混合检索
        print("\n📊 方法3: 混合检索 (向量70% + 关键词30%)")
        print("-" * 50)
        hybrid_results, hybrid_time = retriever.hybrid_search(query, n_results=3)
        for i, result in enumerate(hybrid_results, 1):
            print(f"\n结果 {i} (混合分: {result['similarity']:.1%})")
            print(f"  向量分: {result.get('vector_score', 0):.1%}")
            print(f"  关键词分: {result.get('keyword_score', 0):.1%}")
            print(f"来源: {result['metadata'].get('doc_name', 'unknown')}")
            print(f"内容: {result['document'][:100]}...")
        print(f"\n⏱️  耗时: {hybrid_time*1000:.1f}ms")
        
        # 4. 混合检索 + 重排序
        print("\n📊 方法4: 混合检索 + 重排序")
        print("-" * 50)
        hybrid_results, _ = retriever.hybrid_search(query, n_results=10)
        reranked_results, rerank_time = retriever.rerank_results(query, hybrid_results, top_k=3)
        for i, result in enumerate(reranked_results, 1):
            print(f"\n结果 {i} (重排分: {result['similarity']:.1%})")
            print(f"  原始混合分: {result.get('hybrid_score', 0):.1%}")
            print(f"来源: {result['metadata'].get('doc_name', 'unknown')}")
            print(f"内容: {result['document'][:100]}...")
        print(f"\n⏱️  重排序耗时: {rerank_time*1000:.1f}ms")
    
    # 5. 上下文窗口演示
    print("\n" + "=" * 60)
    print("🪟 上下文窗口演示")
    print("=" * 60)
    query = "醉驾"
    print(f"\n查询: {query}")
    print("-" * 50)
    
    context_results, context_time = retriever.search_with_context(
        query, 
        n_results=2, 
        context_window=1
    )
    
    for i, result in enumerate(context_results, 1):
        print(f"\n结果 {i}:")
        print(f"来源: {result['metadata'].get('doc_name', 'unknown')}")
        
        if result['context_before']:
            print(f"\n⬆️  前文:")
            for ctx in result['context_before']:
                print(f"  {ctx[:80]}...")
        
        print(f"\n📌 匹配块 (相似度: {result['similarity']:.1%}):")
        print(f"  {result['document'][:150]}...")
        
        if result['context_after']:
            print(f"\n⬇️  后文:")
            for ctx in result['context_after']:
                print(f"  {ctx[:80]}...")
    
    print(f"\n⏱️  耗时: {context_time*1000:.1f}ms")
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    print("\n💡 学到的知识:")
    print("   1. 纯向量检索：语义理解好，但可能错过关键词")
    print("   2. 纯关键词检索：精确匹配，但缺乏语义理解")
    print("   3. 混合检索：结合两者优势，效果更好")
    print("   4. 重排序：进一步优化结果顺序")
    print("   5. 上下文窗口：提供更完整的上下文信息")
    print("\n🎯 最佳实践:")
    print("   - 短查询/精确查询：使用混合检索")
    print("   - 长查询/语义查询：向量检索权重更高")
    print("   - 需要完整信息：启用上下文窗口")
    print("   - 对准确率要求高：加入重排序")
    print("\n下一步：构建完整RAG应用 (03_rag_application.py)")


if __name__ == "__main__":
    demo()

