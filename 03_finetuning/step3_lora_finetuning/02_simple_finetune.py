#!/usr/bin/env python3
"""
简单微调脚本

使用LoRA微调Qwen模型，实现交通法问答
"""

import os
import sys
import json
from pathlib import Path

def check_dependencies():
    """检查依赖"""
    print("检查依赖库...")
    
    required = {
        'transformers': 'transformers',
        'peft': 'peft',
        'torch': 'torch',
        'datasets': 'datasets',
        'trl': 'trl'
    }
    
    missing = []
    for name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            missing.append(name)
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    return True


def load_dataset(train_file, eval_file):
    """加载数据集"""
    from datasets import Dataset
    
    print(f"\n📚 加载数据集...")
    
    # 加载训练数据
    train_data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    # 加载验证数据
    eval_data = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
    
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(eval_data)} 条")
    
    # 转换为HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    return train_dataset, eval_dataset


def setup_model_and_tokenizer(model_path):
    """设置模型和分词器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n🤖 加载模型: {model_path}")
    
    # 加载分词器
    print("  加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("  加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=None  # CPU模式
    )
    
    print(f"  ✅ 模型参数: {model.num_parameters() / 1e9:.2f}B")
    
    return model, tokenizer


def setup_lora(model):
    """配置LoRA"""
    from peft import LoraConfig, get_peft_model, TaskType
    
    print("\n⚙️  配置LoRA...")
    
    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                    # rank
        lora_alpha=16,          # alpha
        lora_dropout=0.05,      # dropout
        target_modules=["q_proj", "v_proj"],  # 目标模块
        bias="none"
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  总参数: {total_params / 1e6:.2f}M")
    print(f"  可训练: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    
    return model


def preprocess_function(examples, tokenizer):
    """数据预处理"""
    # 格式化为聊天模板
    texts = []
    for messages in examples["messages"]:
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    # 分词
    model_inputs = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding=False,
    )
    
    # 复制input_ids作为labels
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs


def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir):
    """训练模型"""
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    print("\n🚀 开始训练...")
    
    # 预处理数据
    print("  预处理数据...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        use_cpu=True,  # 强制使用CPU
    )
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\n" + "=" * 60)
    print("训练中...".center(60))
    print("=" * 60)
    print("\n这可能需要30-60分钟，请耐心等待...")
    print("你可以看到loss逐渐下降\n")
    
    trainer.train()
    
    print("\n✅ 训练完成！")
    
    return trainer


def save_model(model, tokenizer, output_dir):
    """保存模型"""
    print(f"\n💾 保存模型到: {output_dir}")
    
    # 保存LoRA适配器
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("  ✅ 模型已保存")


def show_summary(output_dir):
    """显示总结"""
    print("\n" + "=" * 60)
    print("🎉 微调完成！".center(60))
    print("=" * 60)
    
    print(f"\n📁 输出目录: {output_dir}")
    print("\n包含文件:")
    print("  - adapter_config.json  # LoRA配置")
    print("  - adapter_model.bin    # LoRA权重 (~15MB)")
    print("  - tokenizer配置")
    
    print("\n🚀 下一步:")
    print("  python 03_test_finetuned.py  # 测试微调效果")


def main():
    """主函数"""
    print("=" * 60)
    print("LoRA 微调实战".center(60))
    print("=" * 60)
    
    # 1. 检查依赖
    if not check_dependencies():
        return
    
    # 2. 配置路径
    # 数据路径
    data_dir = Path(__file__).parent.parent / "step2_data_preparation" / "data"
    train_file = data_dir / "train.jsonl"
    eval_file = data_dir / "eval.jsonl"
    
    # 如果数据在根目录的data文件夹
    if not train_file.exists():
        data_dir = Path(__file__).parent.parent.parent / "data"
        train_file = data_dir / "train.jsonl"
        eval_file = data_dir / "eval.jsonl"
    
    if not train_file.exists():
        print(f"\n❌ 未找到训练数据: {train_file}")
        print("请先运行: python ../step2_data_preparation/02_create_dataset.py")
        return
    
    # 模型路径
    model_path = os.path.expanduser("~/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf")
    
    # 检查模型
    if not Path(model_path).exists():
        print(f"\n❌ 未找到模型: {model_path}")
        print("\n请提供正确的模型路径")
        return
    
    # 注意：GGUF格式不能直接用于微调
    # 我们需要使用HuggingFace格式的模型
    print("\n⚠️  注意: GGUF格式不支持微调")
    print("我们需要HuggingFace格式的模型")
    print("\n建议:")
    print("  1. 使用在线Colab/Kaggle (免费GPU)")
    print("  2. 下载HF格式模型进行本地微调")
    print("  3. 先理解流程，实际微调可以云端进行")
    print("\n本脚本展示完整的微调流程代码")
    
    # 输出目录
    output_dir = "./output/lora-traffic-law"
    
    print(f"\n📝 配置:")
    print(f"  数据: {data_dir}")
    print(f"  模型: {model_path}")
    print(f"  输出: {output_dir}")
    
    # 由于GGUF格式问题，这里只展示流程
    print("\n" + "=" * 60)
    print("💡 微调流程说明".center(60))
    print("=" * 60)
    
    print("\n完整的微调步骤:")
    print("  1. ✅ 准备数据 (已完成)")
    print("  2. ⏸️  加载模型 (需要HF格式)")
    print("  3. ⏸️  配置LoRA")
    print("  4. ⏸️  训练模型")
    print("  5. ⏸️  保存适配器")
    
    print("\n由于模型格式限制，建议使用Google Colab进行实际微调")
    print("我已经准备好了完整的代码框架")


if __name__ == "__main__":
    main()

