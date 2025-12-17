#!/usr/bin/env python3
"""
RAG最终项目 - 文档管理系统

功能：
1. 多文档导入和管理
2. 元数据管理（文档名、类型、日期等）
3. 文档更新和删除
4. 向量库维护

这是生产级RAG系统的基础组件
"""

import os
import json
import chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


class DocumentManager:
    """文档管理器：管理多个文档的导入、存储和维护"""
    
    def __init__(self, 
                 chroma_path: str = "./data/document_store",
                 collection_name: str = "documents"):
        """
        初始化文档管理器
        
        Args:
            chroma_path: ChromaDB存储路径
            collection_name: 集合名称
        """
        # 初始化向量模型
        print("📦 加载向量模型...")
        self.embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
        # 重要：输出归一化的向量
        self.embedding_model.encode_kwargs = {'normalize_embeddings': True}
        
        # 初始化ChromaDB
        print(f"💾 初始化文档库: {chroma_path}")
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "多文档RAG系统"}
        )
        
        print(f"✅ 文档管理器初始化完成！当前文档数：{self.collection.count()}\n")
    
    def add_document(self, 
                     content: str, 
                     doc_name: str,
                     doc_type: str = "text",
                     metadata: Dict[str, Any] = None,
                     chunk_size: int = 200,
                     chunk_overlap: int = 50) -> Dict[str, Any]:
        """
        添加新文档到系统
        
        Args:
            content: 文档内容
            doc_name: 文档名称
            doc_type: 文档类型（text, pdf, url等）
            metadata: 额外的元数据
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            
        Returns:
            添加结果统计
        """
        print(f"\n📄 开始处理文档: {doc_name}")
        print(f"   文档类型: {doc_type}")
        print(f"   文档长度: {len(content)} 字符")
        
        # 1. 智能分块
        chunks = self._smart_chunk(content, chunk_size, chunk_overlap)
        print(f"   ✂️  分块完成: {len(chunks)} 个块")
        
        # 2. 生成向量（归一化）
        print("   🔄 生成向量...")
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 3. 准备元数据
        timestamp = datetime.now().isoformat()
        base_metadata = {
            "doc_name": doc_name,
            "doc_type": doc_type,
            "import_time": timestamp,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
        # 合并用户提供的元数据
        if metadata:
            base_metadata.update(metadata)
        
        # 4. 为每个块准备数据
        ids = []
        metadatas = []
        for i in range(len(chunks)):
            chunk_id = f"{doc_name}_{timestamp}_{i}"
            ids.append(chunk_id)
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_total": len(chunks)
            })
            metadatas.append(chunk_metadata)
        
        # 5. 添加到向量库
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        result = {
            "doc_name": doc_name,
            "chunks": len(chunks),
            "timestamp": timestamp,
            "total_docs": self.collection.count()
        }
        
        print(f"   ✅ 文档已添加！总文档块数：{result['total_docs']}")
        return result
    
    def _smart_chunk(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        智能分块：按句子边界分块
        
        Args:
            text: 文本内容
            chunk_size: 目标块大小
            overlap: 重叠大小
            
        Returns:
            分块列表
        """
        # 按句子分割
        sentences = []
        for sep in ['。', '！', '？', '\n\n']:
            if sep in text:
                text = text.replace(sep, sep + '|||')
        
        raw_sentences = text.split('|||')
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        # 组合成块
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # 保存当前块
                chunks.append(''.join(current_chunk))
                
                # 计算重叠部分
                overlap_text = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= overlap:
                        overlap_text.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_text
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # 添加最后一块
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        列出所有文档
        
        Returns:
            文档列表（去重后的文档元数据）
        """
        # 获取所有数据
        results = self.collection.get()
        
        if not results['metadatas']:
            return []
        
        # 按文档名分组
        docs_dict = {}
        for metadata in results['metadatas']:
            doc_name = metadata.get('doc_name', 'unknown')
            if doc_name not in docs_dict:
                docs_dict[doc_name] = {
                    'doc_name': doc_name,
                    'doc_type': metadata.get('doc_type', 'unknown'),
                    'import_time': metadata.get('import_time', 'unknown'),
                    'chunks': 0
                }
            docs_dict[doc_name]['chunks'] += 1
        
        return list(docs_dict.values())
    
    def delete_document(self, doc_name: str) -> Dict[str, Any]:
        """
        删除指定文档
        
        Args:
            doc_name: 文档名称
            
        Returns:
            删除结果
        """
        print(f"\n🗑️  删除文档: {doc_name}")
        
        # 查询该文档的所有块
        results = self.collection.get(
            where={"doc_name": doc_name}
        )
        
        if not results['ids']:
            print(f"   ⚠️  文档不存在: {doc_name}")
            return {"success": False, "message": "文档不存在"}
        
        # 删除所有块
        self.collection.delete(ids=results['ids'])
        
        print(f"   ✅ 已删除 {len(results['ids'])} 个文档块")
        return {
            "success": True,
            "doc_name": doc_name,
            "deleted_chunks": len(results['ids']),
            "remaining_total": self.collection.count()
        }
    
    def search_documents(self, 
                        query: str, 
                        n_results: int = 5,
                        doc_name: str = None) -> List[Dict[str, Any]]:
        """
        搜索文档
        
        Args:
            query: 查询文本
            n_results: 返回结果数
            doc_name: 限定文档名（可选）
            
        Returns:
            搜索结果列表
        """
        # 生成查询向量
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        )
        
        # 构建查询条件
        where = {"doc_name": doc_name} if doc_name else None
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            })
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            统计信息字典
        """
        docs = self.list_documents()
        
        return {
            "total_chunks": self.collection.count(),
            "total_documents": len(docs),
            "documents": docs
        }


def demo():
    """演示文档管理系统的功能"""
    print("=" * 60)
    print("RAG最终项目 - 文档管理系统演示")
    print("=" * 60)
    
    # 1. 初始化管理器
    manager = DocumentManager()
    
    # 2. 添加第一个文档（交通法）
    traffic_law = """
    道路交通安全法规定：
    
    第一章 总则
    机动车驾驶人应当遵守道路交通安全法律法规，按照操作规范安全驾驶、文明驾驶。
    饮酒、服用国家管制的精神药品或者麻醉药品，不得驾驶机动车。
    
    第二章 违法处罚
    醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，
    吊销机动车驾驶证，依法追究刑事责任；五年内不得重新取得机动车驾驶证。
    饮酒后驾驶营运机动车的，处十五日拘留，并处五千元罚款，
    吊销机动车驾驶证，五年内不得重新取得机动车驾驶证。
    
    第三章 特殊规定
    超速驾驶按照超速比例进行处罚。超速50%以上的，处以罚款并扣12分。
    闯红灯的，一次记6分，罚款200元。
    """
    
    manager.add_document(
        content=traffic_law,
        doc_name="交通法",
        doc_type="法律文本",
        metadata={"category": "法律", "version": "2023"}
    )
    
    # 3. 添加第二个文档（劳动法）
    labor_law = """
    劳动合同法规定：
    
    第一章 劳动合同订立
    建立劳动关系，应当订立书面劳动合同。已建立劳动关系，未同时订立书面劳动合同的，
    应当自用工之日起一个月内订立书面劳动合同。
    
    第二章 工作时间与休息休假
    国家实行劳动者每日工作时间不超过八小时、平均每周工作时间不超过四十四小时的工时制度。
    用人单位应当保证劳动者每周至少休息一日。
    劳动者连续工作一年以上的，享受带薪年休假。
    
    第三章 工资支付
    工资应当以货币形式按月支付给劳动者本人。不得克扣或者无故拖欠劳动者的工资。
    用人单位安排加班的，应当按照规定支付加班费。
    """
    
    manager.add_document(
        content=labor_law,
        doc_name="劳动法",
        doc_type="法律文本",
        metadata={"category": "法律", "version": "2023"}
    )
    
    # 4. 列出所有文档
    print("\n" + "=" * 60)
    print("📚 文档库清单")
    print("=" * 60)
    docs = manager.list_documents()
    for doc in docs:
        print(f"\n📄 {doc['doc_name']}")
        print(f"   类型: {doc['doc_type']}")
        print(f"   导入时间: {doc['import_time']}")
        print(f"   文档块数: {doc['chunks']}")
    
    # 5. 跨文档搜索
    print("\n" + "=" * 60)
    print("🔍 跨文档搜索测试")
    print("=" * 60)
    
    queries = [
        "工作时间有什么规定",
        "醉驾的处罚",
        "年假怎么算"
    ]
    
    for query in queries:
        print(f"\n问题: {query}")
        print("-" * 50)
        results = manager.search_documents(query, n_results=3)
        for i, result in enumerate(results, 1):
            similarity = 1 - result['distance']
            print(f"\n结果 {i} (相似度: {similarity:.1%})")
            print(f"来源: {result['metadata']['doc_name']}")
            print(f"内容: {result['document'][:100]}...")
    
    # 6. 单文档搜索
    print("\n" + "=" * 60)
    print("🎯 单文档搜索测试")
    print("=" * 60)
    
    query = "处罚"
    print(f"\n在「交通法」中搜索: {query}")
    print("-" * 50)
    results = manager.search_documents(query, n_results=3, doc_name="交通法")
    for i, result in enumerate(results, 1):
        similarity = 1 - result['distance']
        print(f"\n结果 {i} (相似度: {similarity:.1%})")
        print(f"内容: {result['document'][:150]}...")
    
    # 7. 统计信息
    print("\n" + "=" * 60)
    print("📊 系统统计")
    print("=" * 60)
    stats = manager.get_stats()
    print(f"\n总文档数: {stats['total_documents']}")
    print(f"总文档块数: {stats['total_chunks']}")
    print(f"平均每文档块数: {stats['total_chunks'] / stats['total_documents']:.1f}")
    
    # 8. 删除文档测试
    print("\n" + "=" * 60)
    print("🗑️  删除文档测试")
    print("=" * 60)
    
    # 删除劳动法
    result = manager.delete_document("劳动法")
    print(f"\n剩余总块数: {result['remaining_total']}")
    
    # 再次列出文档
    print("\n当前文档:")
    docs = manager.list_documents()
    for doc in docs:
        print(f"  - {doc['doc_name']} ({doc['chunks']} 块)")
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    print("\n💡 学到的知识:")
    print("   1. 多文档管理：可以导入多个文档并独立管理")
    print("   2. 元数据管理：每个文档都有丰富的元数据")
    print("   3. 跨文档搜索：可以在所有文档中搜索")
    print("   4. 单文档搜索：可以限定在特定文档中搜索")
    print("   5. 文档维护：支持删除和更新")
    print("\n下一步：学习高级检索策略 (02_advanced_retrieval.py)")


if __name__ == "__main__":
    demo()

