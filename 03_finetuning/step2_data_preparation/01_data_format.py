#!/usr/bin/env python3
"""
数据格式详解

展示不同的训练数据格式，理解它们的区别和应用场景
"""

import json


def show_format_examples():
    """展示不同的数据格式"""
    print("=" * 60)
    print("训练数据格式详解".center(60))
    print("=" * 60)
    
    # 格式1: 对话格式 (Chat Format)
    print("\n" + "=" * 60)
    print("格式1: 对话格式 (Chat Format)")
    print("=" * 60)
    print("\n适用场景: 多轮对话、聊天机器人\n")
    
    chat_example = {
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的交通法律顾问"
            },
            {
                "role": "user",
                "content": "醉驾会受到什么处罚？"
            },
            {
                "role": "assistant",
                "content": "醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；五年内不得重新取得机动车驾驶证。"
            }
        ]
    }
    
    print("示例:")
    print(json.dumps(chat_example, ensure_ascii=False, indent=2))
    
    print("\n优点:")
    print("  ✅ 支持多轮对话")
    print("  ✅ 可以设置system角色（定义助手人设）")
    print("  ✅ 标准格式，兼容性好")
    
    print("\n缺点:")
    print("  ⚠️  格式相对复杂")
    
    # 格式2: 指令格式 (Instruction Format)
    print("\n" + "=" * 60)
    print("格式2: 指令格式 (Instruction Format)")
    print("=" * 60)
    print("\n适用场景: 单轮问答、任务执行\n")
    
    instruction_example = {
        "instruction": "回答以下交通法问题",
        "input": "醉驾会受到什么处罚？",
        "output": "醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；五年内不得重新取得机动车驾驶证。"
    }
    
    print("示例:")
    print(json.dumps(instruction_example, ensure_ascii=False, indent=2))
    
    print("\n优点:")
    print("  ✅ 格式简单")
    print("  ✅ 适合单轮任务")
    
    print("\n缺点:")
    print("  ⚠️  不支持多轮对话")
    
    # 格式3: 简化格式
    print("\n" + "=" * 60)
    print("格式3: 简化格式 (Simplified Format)")
    print("=" * 60)
    print("\n适用场景: 快速实验、简单任务\n")
    
    simple_example = {
        "prompt": "问题：醉驾会受到什么处罚？\n回答：",
        "completion": "醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；五年内不得重新取得机动车驾驶证。"
    }
    
    print("示例:")
    print(json.dumps(simple_example, ensure_ascii=False, indent=2))
    
    print("\n优点:")
    print("  ✅ 最简单")
    print("  ✅ 灵活自定义格式")
    
    print("\n缺点:")
    print("  ⚠️  需要自己设计prompt格式")


def show_real_examples():
    """展示真实场景的数据示例"""
    print("\n" + "=" * 60)
    print("真实场景示例".center(60))
    print("=" * 60)
    
    examples = [
        {
            "scenario": "法律咨询",
            "data": {
                "messages": [
                    {"role": "system", "content": "你是一个专业的交通法律顾问"},
                    {"role": "user", "content": "闯红灯会扣多少分？"},
                    {"role": "assistant", "content": "闯红灯一次记6分，罚款200元。"}
                ]
            }
        },
        {
            "scenario": "代码生成",
            "data": {
                "instruction": "将以下Python函数转换为JavaScript",
                "input": "def add(a, b):\n    return a + b",
                "output": "function add(a, b) {\n    return a + b;\n}"
            }
        },
        {
            "scenario": "文本总结",
            "data": {
                "instruction": "总结以下文本的要点",
                "input": "道路交通安全法规定，机动车驾驶人应当遵守道路交通安全法律法规...",
                "output": "核心要点：1. 遵守交通法规 2. 安全驾驶 3. 不得酒驾"
            }
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\n示例 {i}: {ex['scenario']}")
        print("-" * 60)
        print(json.dumps(ex['data'], ensure_ascii=False, indent=2))


def explain_our_choice():
    """解释我们的选择"""
    print("\n" + "=" * 60)
    print("我们的方案".center(60))
    print("=" * 60)
    
    print("\n对于交通法问答系统，我们选择：")
    print()
    print("📝 对话格式 (Chat Format)")
    print()
    print("理由:")
    print("  1. ✅ 可以设置专业人设 (system message)")
    print("  2. ✅ 标准格式，工具链成熟")
    print("  3. ✅ 未来可以扩展多轮对话")
    print("  4. ✅ 兼容性好")
    print()
    
    print("数据结构:")
    example = {
        "messages": [
            {"role": "system", "content": "你是专业的交通法律顾问，回答要准确、专业、简洁。"},
            {"role": "user", "content": "用户的问题"},
            {"role": "assistant", "content": "助手的回答"}
        ]
    }
    print(json.dumps(example, ensure_ascii=False, indent=2))


def show_data_file_structure():
    """展示数据文件结构"""
    print("\n" + "=" * 60)
    print("数据文件结构".center(60))
    print("=" * 60)
    
    print("\n我们将创建以下文件:")
    print()
    print("📁 data/")
    print("  ├── train.jsonl          # 训练数据 (90%)")
    print("  └── eval.jsonl           # 评估数据 (10%)")
    print()
    
    print("JSONL格式说明:")
    print("  - 每行一个JSON对象")
    print("  - 方便逐行读取")
    print("  - 适合大规模数据")
    print()
    
    print("示例 (train.jsonl):")
    print("-" * 60)
    line1 = {"messages": [{"role": "system", "content": "你是交通法律顾问"}, {"role": "user", "content": "醉驾处罚？"}, {"role": "assistant", "content": "吊销驾照..."}]}
    line2 = {"messages": [{"role": "system", "content": "你是交通法律顾问"}, {"role": "user", "content": "闯红灯扣分？"}, {"role": "assistant", "content": "扣6分..."}]}
    print(json.dumps(line1, ensure_ascii=False))
    print(json.dumps(line2, ensure_ascii=False))
    print("...")


def show_tips():
    """展示注意事项"""
    print("\n" + "=" * 60)
    print("⚠️  注意事项".center(60))
    print("=" * 60)
    
    tips = [
        ("数据质量 > 数据量", "100条高质量数据优于1000条低质量数据"),
        ("保持一致性", "所有数据的格式、风格要统一"),
        ("避免偏见", "数据要平衡，不要过度集中在某一类问题"),
        ("检查准确性", "所有答案必须正确，错误数据会教坏模型"),
        ("适当多样性", "同一问题用不同表达方式，增加泛化能力"),
    ]
    
    print()
    for i, (title, desc) in enumerate(tips, 1):
        print(f"{i}. {title}")
        print(f"   → {desc}")
        print()


def main():
    """主函数"""
    show_format_examples()
    
    input("\n按 Enter 查看真实场景示例...")
    show_real_examples()
    
    input("\n按 Enter 查看我们的方案...")
    explain_our_choice()
    
    input("\n按 Enter 查看文件结构...")
    show_data_file_structure()
    
    input("\n按 Enter 查看注意事项...")
    show_tips()
    
    print("\n" + "=" * 60)
    print("📝 总结".center(60))
    print("=" * 60)
    print()
    print("你现在应该理解了:")
    print("  ✅ 三种主要数据格式及其应用场景")
    print("  ✅ 我们选择对话格式 (Chat Format)")
    print("  ✅ 数据文件结构 (JSONL)")
    print("  ✅ 数据质量的重要性")
    print()
    print("🚀 下一步:")
    print("  python 02_create_dataset.py  # 创建训练数据集")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 再见！")
    except EOFError:
        print("\n\n✅ 内容已全部展示")
        print("🚀 下一步: python 02_create_dataset.py")

