#!/usr/bin/env python3
"""
RAG 交互式学习 - 亲手调参，理解原理

功能：
1. 对比不同检索策略的效果
2. 调整权重和阈值，实时看结果
3. 可视化展示各种分数
"""

import os
from pathlib import Path
import importlib.util

# 动态导入检索器
current_dir = Path(__file__).parent
retrieval_module_path = current_dir / "02_advanced_retrieval.py"
spec = importlib.util.spec_from_file_location("advanced_retrieval", retrieval_module_path)
advanced_retrieval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(advanced_retrieval)
AdvancedRetriever = advanced_retrieval.AdvancedRetriever


def print_separator(title="", char="="):
    """打印分隔线"""
    if title:
        print(f"\n{char * 60}")
        print(f"{title:^60}")
        print(f"{char * 60}\n")
    else:
        print(f"{char * 60}")


def display_result(result, index):
    """格式化显示单个检索结果"""
    doc_name = result['metadata'].get('doc_name', '未知')
    similarity = result.get('similarity', 0)
    vector_score = result.get('vector_score', 'N/A')
    keyword_score = result.get('keyword_score', 'N/A')
    
    print(f"\n结果 {index}:")
    print(f"  📄 来源: {doc_name}")
    print(f"  🎯 总分: {similarity:.1%}")
    
    if isinstance(vector_score, float):
        print(f"     ├─ 向量分: {vector_score:.1%}")
    if isinstance(keyword_score, float):
        print(f"     └─ 关键词分: {keyword_score:.1%}")
    
    content = result['document'][:120].replace('\n', ' ')
    print(f"  📝 内容: {content}...")


def compare_strategies(retriever, query):
    """对比不同检索策略"""
    print_separator("🔬 检索策略对比实验")
    print(f"查询: {query}\n")
    
    # 1. 纯向量检索
    print("1️⃣  纯向量检索 (语义理解)")
    print("-" * 60)
    vector_results, vector_time = retriever.vector_search(query, n_results=3)
    for i, result in enumerate(vector_results, 1):
        display_result(result, i)
    print(f"\n⏱️  耗时: {vector_time*1000:.0f}ms")
    
    # 2. 纯关键词检索
    print("\n2️⃣  纯关键词检索 (精确匹配)")
    print("-" * 60)
    keyword_results, keyword_time = retriever.keyword_search(query, n_results=3)
    if keyword_results:
        for i, result in enumerate(keyword_results, 1):
            display_result(result, i)
    else:
        print("  ⚠️  未找到匹配结果")
    print(f"\n⏱️  耗时: {keyword_time*1000:.0f}ms")
    
    # 3. 混合检索 (70% + 30%)
    print("\n3️⃣  混合检索 (向量70% + 关键词30%)")
    print("-" * 60)
    hybrid_results, hybrid_time = retriever.hybrid_search(query, n_results=3)
    for i, result in enumerate(hybrid_results, 1):
        display_result(result, i)
    print(f"\n⏱️  耗时: {hybrid_time*1000:.0f}ms")
    
    # 4. 混合 + 重排序
    print("\n4️⃣  混合检索 + 重排序 (最优)")
    print("-" * 60)
    hybrid_results_full, _ = retriever.hybrid_search(query, n_results=10)
    reranked_results, rerank_time = retriever.rerank_results(query, hybrid_results_full, top_k=3)
    for i, result in enumerate(reranked_results, 1):
        display_result(result, i)
    print(f"\n⏱️  重排序耗时: {rerank_time*1000:.0f}ms")
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 速度对比:")
    print(f"   关键词: {keyword_time*1000:>6.0f}ms  (最快)")
    print(f"   向量:   {vector_time*1000:>6.0f}ms")
    print(f"   混合:   {hybrid_time*1000:>6.0f}ms  (推荐)")
    print("=" * 60)


def experiment_weights(retriever, query):
    """实验不同的权重组合"""
    print_separator("⚖️  权重调整实验")
    print(f"查询: {query}\n")
    
    weight_configs = [
        (0.3, 0.7, "关键词优先（精确匹配场景）"),
        (0.5, 0.5, "平衡模式"),
        (0.7, 0.3, "向量优先（语义查询场景）- 默认"),
        (0.9, 0.1, "几乎纯向量"),
    ]
    
    for vector_w, keyword_w, desc in weight_configs:
        print(f"\n配置: {desc}")
        print(f"向量权重={vector_w}, 关键词权重={keyword_w}")
        print("-" * 60)
        
        results, _ = retriever.hybrid_search(
            query, 
            n_results=2,
            vector_weight=vector_w,
            keyword_weight=keyword_w
        )
        
        for i, result in enumerate(results, 1):
            display_result(result, i)


def experiment_threshold(retriever, query):
    """实验不同的阈值"""
    print_separator("🎚️  阈值调整实验")
    print(f"查询: {query}\n")
    
    # 先获取所有结果
    results, _ = retriever.hybrid_search(query, n_results=10)
    
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    for threshold in thresholds:
        print(f"\n阈值 = {threshold} ({threshold*100:.0f}%)")
        print("-" * 60)
        
        filtered = [r for r in results if r.get('similarity', 0) >= threshold]
        
        if filtered:
            print(f"找到 {len(filtered)} 条结果:")
            for i, result in enumerate(filtered[:3], 1):
                similarity = result.get('similarity', 0)
                content = result['document'][:80].replace('\n', ' ')
                print(f"  {i}. ({similarity:.1%}) {content}...")
        else:
            print("  ❌ 无结果（阈值太高）")
    
    print("\n" + "=" * 60)
    print("💡 建议:")
    print("   - 阈值太低(0.2): 可能有噪音")
    print("   - 阈值中等(0.3-0.4): 平衡 ✅")
    print("   - 阈值太高(0.6+): 可能找不到结果")
    print("=" * 60)


def interactive_menu():
    """交互式菜单"""
    print_separator("🎓 RAG 交互式学习系统", "=")
    
    # 初始化检索器
    print("📦 正在加载...")
    retriever = AdvancedRetriever()
    
    test_queries = {
        '1': ('醉驾', '短查询'),
        '2': ('醉驾的处罚是什么', '长查询'),
        '3': ('工作时间', '关键词查询'),
        '4': ('加班费怎么算', '语义查询'),
    }
    
    while True:
        print("\n" + "=" * 60)
        print("选择实验:")
        print("=" * 60)
        print("\n实验类型:")
        print("  [1] 🔬 检索策略对比 - 看懂4种方法的区别")
        print("  [2] ⚖️  权重调整实验 - 理解向量和关键词的平衡")
        print("  [3] 🎚️  阈值调整实验 - 学会控制结果质量")
        print("  [4] 🎯 自定义查询")
        print("  [0] 退出")
        
        choice = input("\n选择 [0-4]: ").strip()
        
        if choice == '0':
            print("\n👋 再见！记得看 LEARNING_NOTES.md 学习笔记！")
            break
        
        if choice not in ['1', '2', '3', '4']:
            print("❌ 无效选择")
            continue
        
        # 选择查询
        if choice == '4':
            # 自定义查询：先输入查询，再选实验类型
            query = input("\n输入你的查询: ").strip()
            if not query:
                continue
            
            print(f"\n✅ 你的查询: {query}")
            print("\n选择实验类型:")
            print("  [1] 🔬 检索策略对比")
            print("  [2] ⚖️  权重调整实验")
            print("  [3] 🎚️  阈值调整实验")
            
            exp_choice = input("选择 [1-3]: ").strip()
            if exp_choice not in ['1', '2', '3']:
                print("❌ 无效选择")
                continue
            
            # 用自定义查询执行对应的实验
            if exp_choice == '1':
                compare_strategies(retriever, query)
            elif exp_choice == '2':
                experiment_weights(retriever, query)
            elif exp_choice == '3':
                experiment_threshold(retriever, query)
            
            input("\n按 Enter 继续...")
            continue  # 跳过后面的执行，回到主菜单
        else:
            # 预设查询：先选实验，再选查询
            print("\n选择测试查询:")
            for key, (q, desc) in test_queries.items():
                print(f"  [{key}] {q} ({desc})")
            
            q_choice = input("选择 [1-4]: ").strip()
            if q_choice not in test_queries:
                print("❌ 无效选择")
                continue
            
            query = test_queries[q_choice][0]
        
        # 执行实验（只处理预设查询的情况，自定义查询已在上面处理）
        try:
            if choice == '1':
                compare_strategies(retriever, query)
            elif choice == '2':
                experiment_weights(retriever, query)
            elif choice == '3':
                experiment_threshold(retriever, query)
            
            input("\n按 Enter 继续...")
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def quick_demo():
    """快速演示模式"""
    print_separator("🚀 快速演示模式", "=")
    
    print("📦 加载检索器...")
    retriever = AdvancedRetriever()
    
    # 实验1: 策略对比
    compare_strategies(retriever, "醉驾的处罚")
    input("\n按 Enter 继续下一个实验...")
    
    # 实验2: 权重调整
    experiment_weights(retriever, "醉驾")
    input("\n按 Enter 继续下一个实验...")
    
    # 实验3: 阈值调整
    experiment_threshold(retriever, "处罚")
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    print("\n💡 现在你应该理解了:")
    print("   1. 向量检索 vs 关键词检索 vs 混合检索")
    print("   2. 权重如何影响结果排序")
    print("   3. 阈值如何控制结果数量和质量")
    print("\n📚 详细原理请阅读: LEARNING_NOTES.md")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        # 快速演示模式
        quick_demo()
    else:
        # 交互式模式
        try:
            interactive_menu()
        except KeyboardInterrupt:
            print("\n\n👋 再见！")

