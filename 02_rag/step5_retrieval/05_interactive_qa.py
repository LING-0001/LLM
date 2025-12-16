#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5.5: 交互式问答系统
学习目标：构建命令行交互界面，体验完整RAG
"""

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os
import time
from datetime import datetime

# ============================================================
# 导入RAG系统类
# ============================================================

class TrafficLawRAG:
    """交通法RAG问答系统（从04脚本复制）"""
    
    def __init__(self, db_path, embedding_model_name, llm_path, collection_name="traffic_law"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm = Llama(
            model_path=llm_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )
        self.history = []  # 对话历史
    
    def retrieve(self, question, top_k=10, threshold=0.5, max_results=3):
        question_vector = self.embedding_model.encode([question], show_progress_bar=False)
        results = self.collection.query(
            query_embeddings=question_vector.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
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
        
        return retrieved_docs[:max_results]
    
    def generate(self, question, context, stream=True):
        prompt = f"""你是一个专业的交通法规助手，专门解答中国道路交通安全法相关问题。

【参考资料】
{context if context else "（无相关文档）"}

【回答规则】
1. 严格依据参考资料回答，不编造信息
2. 尽量根据参考资料回答，可以适当推理
3. 回答要准确、简洁、分点列出
4. 保持客观中立的语气

【用户问题】
{question}

【你的回答】
"""
        
        if stream:
            # 流式输出
            output = self.llm(
                prompt,
                max_tokens=300,
                temperature=0.2,
                stop=["【", "\n\n\n"],
                echo=False,
                stream=True
            )
            
            answer = ""
            for chunk in output:
                text = chunk['choices'][0]['text']
                print(text, end="", flush=True)
                answer += text
            print()  # 换行
            
            return answer, prompt
        else:
            output = self.llm(
                prompt,
                max_tokens=300,
                temperature=0.2,
                stop=["【", "\n\n\n"],
                echo=False,
                stream=False
            )
            
            return output['choices'][0]['text'].strip(), prompt
    
    def query(self, question, show_sources=True):
        start_time = time.time()
        
        # 检索
        retrieved_docs = self.retrieve(question)
        
        # 显示检索结果
        if len(retrieved_docs) == 0:
            print(f"\n❌ 未检索到文档 (相似度都<50%)")
            print(f"💬 直接让LLM回答\n")
            context = ""
            sources = []
        else:
            print(f"\n✅ 检索到 {len(retrieved_docs)} 个文档 (相似度>50%):")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"   [{i}] {doc['chapter'][:20]}... ({doc['similarity']:.0%})")
            
            context = "\n\n".join([doc['content'] for doc in retrieved_docs])
            sources = retrieved_docs
            
            # 显示喂给LLM的内容
            print(f"\n📝 喂给LLM (共{len(context)}字，前80字):")
            print(f"   {context[:80].replace(chr(10), ' ')}...")
        
        # 生成
        print(f"\n💡 答案:")
        print("=" * 60)
        answer, prompt = self.generate(question, context, stream=True)
        print("=" * 60)
        
        total_time = time.time() - start_time
        
        # 记录历史
        self.history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'question': question,
            'answer': answer,
            'num_sources': len(sources),
            'time': total_time
        })
        
        print(f"\n⏱️  耗时: {total_time:.2f}秒")
        
        return answer, sources


# ============================================================
# 交互式界面
# ============================================================

def print_banner():
    """打印欢迎界面"""
    print("\n" + "=" * 60)
    print("🚗 交通法RAG问答系统")
    print("=" * 60)
    print("\n欢迎使用！我是交通法规助手，可以回答关于中国道路交通安全法的问题。\n")
    print("💡 提示:")
    print("   • 输入问题并回车")
    print("   • 输入 'help' 查看帮助")
    print("   • 输入 'history' 查看历史")
    print("   • 输入 'exit' 退出")
    print("\n" + "=" * 60 + "\n")


def print_help():
    """打印帮助信息"""
    print("\n" + "=" * 60)
    print("📖 帮助信息")
    print("=" * 60)
    print("\n可用命令:")
    print("   • help      - 显示此帮助")
    print("   • history   - 查看对话历史")
    print("   • clear     - 清空屏幕")
    print("   • sources   - 显示/隐藏参考来源")
    print("   • exit/quit - 退出系统")
    print("\n示例问题:")
    print("   • 酒驾的处罚是什么？")
    print("   • 闯红灯要扣几分？")
    print("   • 交通事故后怎么处理？")
    print("   • 驾驶证扣满12分怎么办？")
    print("\n" + "=" * 60 + "\n")


def print_history(rag_system):
    """打印对话历史"""
    if not rag_system.history:
        print("\n📋 还没有对话历史\n")
        return
    
    print("\n" + "=" * 60)
    print(f"📋 对话历史 (共{len(rag_system.history)}条)")
    print("=" * 60)
    
    for i, item in enumerate(rag_system.history, 1):
        print(f"\n[{i}] {item['timestamp']}")
        print(f"   Q: {item['question']}")
        print(f"   A: {item['answer'][:60]}...")
        print(f"   来源: {item['num_sources']}个 | 耗时: {item['time']:.2f}s")
    
    print("\n" + "=" * 60 + "\n")


def save_history(rag_system, filename="qa_history.txt"):
    """保存对话历史到文件"""
    if not rag_system.history:
        print("\n⚠️  没有对话历史可以保存\n")
        return
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("交通法RAG问答系统 - 对话历史\n")
            f.write("=" * 60 + "\n\n")
            
            for i, item in enumerate(rag_system.history, 1):
                f.write(f"[{i}] {item['timestamp']}\n")
                f.write(f"问题: {item['question']}\n")
                f.write(f"答案: {item['answer']}\n")
                f.write(f"来源: {item['num_sources']}个文档\n")
                f.write(f"耗时: {item['time']:.2f}秒\n")
                f.write("\n" + "-" * 60 + "\n\n")
        
        print(f"\n✅ 对话历史已保存到: {filename}\n")
    except Exception as e:
        print(f"\n❌ 保存失败: {e}\n")


def main():
    """主函数"""
    # 打印欢迎界面
    print_banner()
    
    # 初始化系统
    print("🚀 正在加载系统...")
    print("   [1/3] 加载向量数据库...")
    print("   [2/3] 加载Embedding模型...")
    print("   [3/3] 加载LLM...")
    
    try:
        rag_system = TrafficLawRAG(
            db_path="../step4_vectorstore/data/chroma_traffic_law",
            embedding_model_name="shibing624/text2vec-base-chinese",
            llm_path="/Users/a58/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf"
        )
        print("\n✅ 系统加载完成！\n")
    except Exception as e:
        print(f"\n❌ 系统加载失败: {e}")
        print("\n请检查:")
        print("   1. 向量数据库是否存在")
        print("   2. LLM模型路径是否正确")
        print("   3. 依赖包是否安装完整")
        return
    
    show_sources = True  # 是否显示来源
    
    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = input("💬 你: ").strip()
            
            # 处理空输入
            if not user_input:
                continue
            
            # 处理命令
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 感谢使用！再见！\n")
                
                # 询问是否保存历史
                if rag_system.history:
                    save_choice = input("是否保存对话历史？(y/n): ").strip().lower()
                    if save_choice == 'y':
                        save_history(rag_system)
                
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'history':
                print_history(rag_system)
                continue
            
            elif user_input.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                print_banner()
                continue
            
            elif user_input.lower() == 'sources':
                show_sources = not show_sources
                print(f"\n✅ 参考来源显示已{'开启' if show_sources else '关闭'}\n")
                continue
            
            # 处理问题
            print(f"\n🤔 正在思考...")
            answer, sources = rag_system.query(user_input, show_sources=show_sources)
            
        except KeyboardInterrupt:
            print("\n\n👋 检测到Ctrl+C，退出系统\n")
            break
        
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")
            continue


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    main()

