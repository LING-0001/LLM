#!/usr/bin/env python3
"""
Fine-tuning 基础 - 什么是微调

本脚本通过可视化演示帮助理解：
1. 预训练模型的能力
2. 微调如何改变模型行为
3. 微调的应用场景
"""

def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60 + "\n")


def explain_pretraining_vs_finetuning():
    """解释预训练和微调的区别"""
    print_section("🧠 预训练 vs 微调")
    
    print("【预训练模型】就像一个「通才」:")
    print("  - 知识广泛但不深入")
    print("  - 可以完成各种基础任务")
    print("  - 但在专业领域表现一般")
    print()
    
    print("示例: 通用模型回答医疗问题")
    print("-" * 60)
    print("问: 患者出现持续性胸痛，应该如何处理？")
    print("答: 胸痛可能由多种原因引起，建议及时就医检查。")
    print("     可能需要做心电图、血液检查等。")
    print()
    print("⚠️  问题: 回答过于笼统，缺乏专业性")
    print()
    
    print("=" * 60)
    print()
    
    print("【微调后的模型】就像一个「专家」:")
    print("  - 在特定领域深入专业")
    print("  - 理解领域术语和规范")
    print("  - 能给出专业的回答")
    print()
    
    print("示例: 医疗领域微调后的模型")
    print("-" * 60)
    print("问: 患者出现持续性胸痛，应该如何处理？")
    print("答: 持续性胸痛需立即评估：")
    print("     1. 首先排查急性冠脉综合征(ACS)")
    print("     2. 立即进行心电图检查")
    print("     3. 检测心肌标志物(肌钙蛋白)")
    print("     4. 评估TIMI风险评分")
    print("     5. 根据结果决定是否需要紧急介入治疗")
    print()
    print("✅ 改进: 专业术语、标准流程、可执行步骤")


def explain_how_finetuning_works():
    """解释微调的工作原理"""
    print_section("⚙️  微调如何工作")
    
    print("【训练过程】")
    print()
    print("步骤1: 准备数据")
    print("  输入: 患者出现持续性胸痛，应该如何处理？")
    print("  期望输出: 持续性胸痛需立即评估...")
    print()
    
    print("步骤2: 计算误差")
    print("  模型当前输出: 胸痛可能由多种原因引起...")
    print("  期望输出:     持续性胸痛需立即评估...")
    print("  误差: 太大! 需要调整模型")
    print()
    
    print("步骤3: 更新模型参数")
    print("  通过反向传播算法，调整模型内部参数")
    print("  使模型输出更接近期望输出")
    print()
    
    print("步骤4: 重复训练")
    print("  用成千上万个这样的例子反复训练")
    print("  直到模型能够给出专业的回答")
    print()
    
    print("【核心原理】")
    print("  微调 = 在预训练模型的基础上，用专业数据继续训练")
    print("  就像让「通才」去读「专业教科书」变成「专家」")


def explain_finetuning_types():
    """解释微调的类型"""
    print_section("📚 微调的类型")
    
    types = [
        {
            "name": "全量微调 (Full Fine-tuning)",
            "desc": "更新模型的所有参数",
            "pros": "效果最好",
            "cons": "需要大量GPU内存和时间",
            "suitable": "有充足资源，追求最佳效果"
        },
        {
            "name": "LoRA (Low-Rank Adaptation)",
            "desc": "只训练小部分额外参数",
            "pros": "资源需求低，训练快",
            "cons": "效果略逊于全量微调",
            "suitable": "资源有限，追求效率 ⭐推荐"
        },
        {
            "name": "Prompt Tuning",
            "desc": "只优化输入的提示词",
            "pros": "资源需求极低",
            "cons": "效果有限",
            "suitable": "极度资源受限"
        }
    ]
    
    for i, t in enumerate(types, 1):
        print(f"{i}. {t['name']}")
        print(f"   原理: {t['desc']}")
        print(f"   优点: {t['pros']}")
        print(f"   缺点: {t['cons']}")
        print(f"   适用: {t['suitable']}")
        print()


def explain_use_cases():
    """解释微调的应用场景"""
    print_section("🎯 什么时候需要微调")
    
    scenarios = [
        {
            "scenario": "专业领域问答",
            "example": "医疗、法律、金融等专业咨询",
            "why": "需要专业术语和深度理解",
            "method": "✅ 微调"
        },
        {
            "scenario": "特定风格生成",
            "example": "诗歌、文言文、特定作家风格",
            "why": "需要学习特定的表达风格",
            "method": "✅ 微调"
        },
        {
            "scenario": "格式化输出",
            "example": "JSON、SQL、代码生成",
            "why": "需要严格遵守格式规范",
            "method": "✅ 微调"
        },
        {
            "scenario": "多轮对话",
            "example": "客服机器人、助手",
            "why": "需要理解上下文和意图",
            "method": "✅ 微调"
        },
        {
            "scenario": "简单事实查询",
            "example": "公司政策、产品说明",
            "why": "只需检索文档",
            "method": "❌ RAG即可"
        },
        {
            "scenario": "频繁更新的知识",
            "example": "新闻、实时数据",
            "why": "微调更新成本高",
            "method": "❌ RAG即可"
        }
    ]
    
    for s in scenarios:
        print(f"场景: {s['scenario']}")
        print(f"  例子: {s['example']}")
        print(f"  原因: {s['why']}")
        print(f"  方案: {s['method']}")
        print()


def explain_rag_vs_finetuning():
    """对比RAG和Fine-tuning"""
    print_section("⚖️  RAG vs Fine-tuning 对比")
    
    print("┌─────────────┬──────────────────┬──────────────────┐")
    print("│   维度      │       RAG        │   Fine-tuning    │")
    print("├─────────────┼──────────────────┼──────────────────┤")
    print("│ 知识来源    │ 外部文档         │ 模型内部         │")
    print("│ 知识深度    │ 浅层（检索）     │ 深层（理解）     │")
    print("│ 更新成本    │ 低（换文档）     │ 高（重训练）     │")
    print("│ 响应速度    │ 慢（检索+生成）  │ 快（直接生成）   │")
    print("│ 硬件要求    │ 低               │ 中高             │")
    print("│ 适用场景    │ 事实查询         │ 风格学习         │")
    print("│ 可解释性    │ 高（可追溯来源） │ 低（黑盒）       │")
    print("└─────────────┴──────────────────┴──────────────────┘")
    print()
    
    print("💡 最佳实践: RAG + Fine-tuning 结合")
    print("  - 用RAG处理事实性知识（可更新）")
    print("  - 用Fine-tuning学习表达风格和专业理解（稳定）")
    print("  - 举例: 法律咨询系统")
    print("    · RAG: 检索最新法律条文")
    print("    · Fine-tuning: 学习法律术语和推理方式")


def show_next_steps():
    """显示下一步"""
    print_section("📝 总结与下一步")
    
    print("✅ 你现在应该理解了:")
    print("  1. 什么是微调 - 让通用模型变成专业模型")
    print("  2. 微调的类型 - LoRA是资源友好的选择")
    print("  3. 应用场景 - 专业领域、特定风格、格式化输出")
    print("  4. RAG vs 微调 - 各有优势，可以结合")
    print()
    
    print("🚀 下一步:")
    print("  1. 准备环境: python 02_prepare_environment.py")
    print("  2. 学习数据准备: cd ../step2_data_preparation")
    print()
    
    print("💡 小测试:")
    print("  想一想: 你想用微调解决什么问题？")
    print("  - 是否真的需要微调，还是RAG就够了？")
    print("  - 有没有足够的训练数据（至少几百条）？")
    print("  - 有没有GPU或者能等待较长训练时间？")


def main():
    """主函数"""
    print("=" * 60)
    print("Fine-tuning 基础教程".center(60))
    print("什么是微调？为什么需要微调？".center(60))
    print("=" * 60)
    
    # 1. 预训练vs微调
    explain_pretraining_vs_finetuning()
    input("\n按 Enter 继续...")
    
    # 2. 微调工作原理
    explain_how_finetuning_works()
    input("\n按 Enter 继续...")
    
    # 3. 微调类型
    explain_finetuning_types()
    input("\n按 Enter 继续...")
    
    # 4. 应用场景
    explain_use_cases()
    input("\n按 Enter 继续...")
    
    # 5. RAG vs Fine-tuning
    explain_rag_vs_finetuning()
    input("\n按 Enter 继续...")
    
    # 6. 下一步
    show_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 下次见！")

