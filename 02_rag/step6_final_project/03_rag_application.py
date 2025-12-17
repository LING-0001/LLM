#!/usr/bin/env python3
"""
RAG最终项目 - 完整RAG应用

这是一个生产级的RAG问答系统，整合所有学到的知识：
- 文档管理
- 高级检索
- Prompt优化
- 流式输出
- 性能监控
- 用户体验优化
"""

import os
import sys
import time
from llama_cpp import Llama
from typing import List, Dict, Any, Optional

# 导入前面开发的模块
from pathlib import Path
import importlib.util

# 动态导入同目录的模块
current_dir = Path(__file__).parent
retrieval_module_path = current_dir / "02_advanced_retrieval.py"
spec = importlib.util.spec_from_file_location("advanced_retrieval", retrieval_module_path)
advanced_retrieval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(advanced_retrieval)
AdvancedRetriever = advanced_retrieval.AdvancedRetriever


class ProductionRAG:
    """生产级RAG系统"""
    
    def __init__(self, 
                 model_path: str,
                 chroma_path: str = "./data/document_store",
                 collection_name: str = "documents"):
        """
        初始化RAG系统
        
        Args:
            model_path: LLM模型路径
            chroma_path: ChromaDB路径
            collection_name: 集合名称
        """
        print("=" * 60)
        print("🚀 初始化生产级RAG系统")
        print("=" * 60)
        
        # 1. 加载LLM
        print("\n📦 加载语言模型...")
        print(f"   模型: {os.path.basename(model_path)}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=0,
            verbose=False
        )
        print("   ✅ 模型加载完成")
        
        # 2. 初始化检索器
        print("\n🔍 初始化检索系统...")
        self.retriever = AdvancedRetriever(
            chroma_path=chroma_path,
            collection_name=collection_name
        )
        
        # 3. 系统配置
        self.config = {
            'retrieval_method': 'hybrid',  # vector, keyword, hybrid
            'n_results': 5,
            'use_rerank': True,
            'use_context_window': False,  # 暂时关闭上下文窗口，用混合检索
            'context_window_size': 1,
            'similarity_threshold': 0.3,  # 降低阈值，因为向量距离可能是负数
            'max_context_length': 2000,
            'llm_temperature': 0.3,
            'llm_max_tokens': 512
        }
        
        print("\n⚙️  系统配置:")
        for key, value in self.config.items():
            print(f"   {key}: {value}")
        
        print("\n" + "=" * 60)
        print("✅ RAG系统初始化完成！")
        print("=" * 60 + "\n")
    
    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        执行检索
        
        Args:
            query: 用户查询
            
        Returns:
            检索结果和统计信息
        """
        start_time = time.time()
        
        # 1. 根据配置选择检索方法
        method = self.config['retrieval_method']
        n_results = self.config['n_results']
        
        if self.config['use_context_window']:
            # 带上下文窗口的检索
            results, _ = self.retriever.search_with_context(
                query,
                n_results=n_results,
                context_window=self.config['context_window_size']
            )
        elif method == 'vector':
            results, _ = self.retriever.vector_search(query, n_results=n_results * 2)
        elif method == 'keyword':
            results, _ = self.retriever.keyword_search(query, n_results=n_results * 2)
        else:  # hybrid
            results, _ = self.retriever.hybrid_search(query, n_results=n_results * 2)
        
        # 2. 重排序（如果启用）
        if self.config['use_rerank'] and not self.config['use_context_window']:
            results, _ = self.retriever.rerank_results(query, results, top_k=n_results)
        
        # 3. 过滤低相似度结果
        threshold = self.config['similarity_threshold']
        filtered_results = [r for r in results if r.get('similarity', 0) >= threshold]
        
        retrieval_time = time.time() - start_time
        
        return {
            'results': filtered_results,
            'total_found': len(filtered_results),
            'retrieval_time': retrieval_time,
            'method': method + ('+rerank' if self.config['use_rerank'] else '')
        }
    
    def build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        构建优化的Prompt
        
        Args:
            query: 用户查询
            contexts: 检索到的上下文
            
        Returns:
            完整的prompt
        """
        if not contexts:
            # 没有检索到相关内容
            prompt = f"""你是一个专业的AI助手。请直接根据你的知识回答以下问题。

问题：{query}

请给出准确、专业的回答："""
            return prompt
        
        # 构建上下文
        context_text = ""
        max_length = self.config['max_context_length']
        current_length = 0
        
        for i, ctx in enumerate(contexts, 1):
            # 使用完整上下文（如果有）
            if 'full_context' in ctx:
                text = ctx['full_context']
            else:
                text = ctx['document']
            
            # 控制总长度
            if current_length + len(text) > max_length:
                remaining = max_length - current_length
                if remaining > 100:  # 至少保留100字符
                    text = text[:remaining] + "..."
                else:
                    break
            
            doc_name = ctx['metadata'].get('doc_name', '未知')
            similarity = ctx.get('similarity', 0)
            
            context_text += f"\n参考资料 {i} (来源:{doc_name}, 相关度:{similarity:.0%}):\n{text}\n"
            current_length += len(text)
        
        # 构建完整prompt
        prompt = f"""你是一个专业的AI助手。请基于提供的参考资料回答问题。

参考资料：
{context_text}

问题：{query}

回答要求：
1. 优先使用参考资料中的信息
2. 如果参考资料不够充分，可以结合你的知识补充
3. 回答要准确、专业、简洁
4. 如果参考资料与问题完全无关，说明情况后再用你的知识回答

请回答："""
        
        return prompt
    
    def generate(self, prompt: str, stream: bool = True):
        """
        生成回答
        
        Args:
            prompt: 完整的prompt
            stream: 是否流式输出
            
        Yields/Returns:
            流式输出文本块或完整文本
        """
        start_time = time.time()
        
        response = self.llm(
            prompt,
            max_tokens=self.config['llm_max_tokens'],
            temperature=self.config['llm_temperature'],
            stop=["问题：", "\n\n\n"],
            stream=stream
        )
        
        if stream:
            # 流式输出
            full_text = ""
            for chunk in response:
                text = chunk['choices'][0]['text']
                full_text += text
                yield {
                    'text': text,
                    'full_text': full_text,
                    'done': False
                }
            
            generation_time = time.time() - start_time
            yield {
                'text': '',
                'full_text': full_text,
                'done': True,
                'generation_time': generation_time
            }
        else:
            # 非流式输出
            full_text = response['choices'][0]['text']
            generation_time = time.time() - start_time
            return {
                'text': full_text,
                'generation_time': generation_time
            }
    
    def answer(self, query: str, stream: bool = True, verbose: bool = True):
        """
        回答问题（完整流程）
        
        Args:
            query: 用户问题
            stream: 是否流式输出
            verbose: 是否显示详细信息
            
        Yields/Returns:
            回答结果
        """
        total_start = time.time()
        
        # 1. 检索
        if verbose:
            print(f"🔍 检索中...", end='', flush=True)
        
        retrieval_result = self.retrieve(query)
        
        if verbose:
            print(f" 完成 ({retrieval_result['retrieval_time']*1000:.0f}ms)")
            print(f"   方法: {retrieval_result['method']}")
            print(f"   找到: {retrieval_result['total_found']} 条相关内容")
            
            if retrieval_result['total_found'] > 0:
                print("\n📚 检索结果:")
                for i, result in enumerate(retrieval_result['results'][:3], 1):
                    doc_name = result['metadata'].get('doc_name', '未知')
                    similarity = result.get('similarity', 0)
                    content = result['document'][:80].replace('\n', ' ')
                    print(f"   {i}. [{doc_name}] ({similarity:.0%}) {content}...")
            else:
                print("   ⚠️  未找到相关内容，将使用LLM直接回答")
        
        # 2. 构建Prompt
        prompt = self.build_prompt(query, retrieval_result['results'])
        
        if verbose:
            print(f"\n📝 Prompt长度: {len(prompt)} 字符")
        
        # 3. 生成回答
        if verbose:
            print(f"\n💬 AI回答:")
            print("-" * 60)
        
        if stream:
            # 流式输出
            for chunk in self.generate(prompt, stream=True):
                if not chunk['done']:
                    if verbose:
                        print(chunk['text'], end='', flush=True)
                    yield chunk
                else:
                    if verbose:
                        print()
                        print("-" * 60)
                    
                    total_time = time.time() - total_start
                    
                    result = {
                        'query': query,
                        'answer': chunk['full_text'],
                        'retrieval_time': retrieval_result['retrieval_time'],
                        'generation_time': chunk['generation_time'],
                        'total_time': total_time,
                        'num_contexts': retrieval_result['total_found'],
                        'method': retrieval_result['method']
                    }
                    
                    if verbose:
                        print(f"\n⏱️  性能统计:")
                        print(f"   检索: {result['retrieval_time']*1000:.0f}ms")
                        print(f"   生成: {result['generation_time']*1000:.0f}ms")
                        print(f"   总计: {result['total_time']*1000:.0f}ms")
                    
                    yield result
        else:
            # 非流式输出
            generation_result = self.generate(prompt, stream=False)
            total_time = time.time() - total_start
            
            result = {
                'query': query,
                'answer': generation_result['text'],
                'retrieval_time': retrieval_result['retrieval_time'],
                'generation_time': generation_result['generation_time'],
                'total_time': total_time,
                'num_contexts': retrieval_result['total_found'],
                'method': retrieval_result['method']
            }
            
            if verbose:
                print(result['answer'])
                print("-" * 60)
                print(f"\n⏱️  性能统计:")
                print(f"   检索: {result['retrieval_time']*1000:.0f}ms")
                print(f"   生成: {result['generation_time']*1000:.0f}ms")
                print(f"   总计: {result['total_time']*1000:.0f}ms")
            
            return result
    
    def interactive_mode(self):
        """交互式问答模式"""
        print("\n" + "=" * 60)
        print("💬 进入交互式问答模式")
        print("=" * 60)
        print("\n命令:")
        print("  - 输入问题进行提问")
        print("  - 输入 'config' 查看/修改配置")
        print("  - 输入 'quit' 或 'exit' 退出")
        print("\n" + "=" * 60 + "\n")
        
        while True:
            try:
                # 获取用户输入
                query = input("👤 你: ").strip()
                
                if not query:
                    continue
                
                # 处理命令
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 再见！")
                    break
                
                if query.lower() == 'config':
                    self._show_config_menu()
                    continue
                
                # 回答问题
                print()
                final_result = None
                for result in self.answer(query, stream=True, verbose=True):
                    if isinstance(result, dict) and 'total_time' in result:
                        final_result = result
                
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
    
    def _show_config_menu(self):
        """显示配置菜单"""
        print("\n⚙️  当前配置:")
        for i, (key, value) in enumerate(self.config.items(), 1):
            print(f"   {i}. {key}: {value}")
        print("\n提示: 配置修改功能可以进一步开发")


def demo():
    """演示完整RAG应用"""
    print("=" * 60)
    print("RAG最终项目 - 完整RAG应用演示")
    print("=" * 60)
    
    # 检查模型路径
    model_path = os.path.expanduser("~/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf")
    if not os.path.exists(model_path):
        print(f"\n❌ 模型文件不存在: {model_path}")
        print("请确保模型文件已下载")
        return
    
    # 初始化RAG系统
    rag = ProductionRAG(model_path=model_path)
    
    # 测试几个问题
    test_queries = [
        "醉驾会受到什么处罚？",
        "劳动者的工作时间有什么规定？",
        "什么是人工智能？"  # 文档中没有的问题
    ]
    
    print("\n📋 批量测试模式")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(test_queries)}")
        print(f"{'='*60}")
        print(f"\n问题: {query}\n")
        
        # 回答问题
        for result in rag.answer(query, stream=True, verbose=True):
            if isinstance(result, dict) and 'total_time' in result:
                pass  # 最终结果已经在answer中打印
        
        if i < len(test_queries):
            print("\n" + "="*60)
    
    # 进入交互模式
    print("\n" + "=" * 60)
    choice = input("\n是否进入交互模式？(y/n): ").strip().lower()
    if choice == 'y':
        rag.interactive_mode()
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    print("\n🎉 恭喜！你已经完成了RAG完整学习路径！")
    print("\n💡 你现在掌握了:")
    print("   ✅ RAG完整原理和实现")
    print("   ✅ 文档管理和向量存储")
    print("   ✅ 多种检索策略和优化")
    print("   ✅ Prompt工程和生成优化")
    print("   ✅ 生产级系统设计")
    print("\n🚀 下一步:")
    print("   - Phase 3: Fine-tuning (模型微调)")
    print("   - 将RAG应用到实际项目")
    print("   - 探索更多优化技术")


if __name__ == "__main__":
    demo()

