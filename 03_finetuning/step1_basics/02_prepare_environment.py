#!/usr/bin/env python3
"""
Fine-tuning 环境准备

检查并安装微调所需的库
"""

import sys
import subprocess
import importlib


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60 + "\n")


def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name:20s} 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name:20s} 未安装")
        return False


def install_package(package_name):
    """安装包"""
    print(f"\n📦 正在安装 {package_name}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name, "-q"]
        )
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package_name} 安装失败")
        return False


def check_environment():
    """检查环境"""
    print_section("🔍 检查当前环境")
    
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    print()
    
    # 检查必需的包
    required_packages = {
        "transformers": "transformers",
        "peft": "peft",
        "datasets": "datasets",
        "torch": "torch",
        "trl": "trl",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
    }
    
    print("检查必需的包:")
    print("-" * 60)
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    return missing_packages


def install_missing_packages(missing_packages):
    """安装缺失的包"""
    if not missing_packages:
        print("\n✅ 所有必需的包都已安装！")
        return True
    
    print_section(f"📦 需要安装 {len(missing_packages)} 个包")
    
    print("将安装以下包:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    print()
    
    # 特殊处理：bitsandbytes 在 Mac 上可能不可用
    if "bitsandbytes" in missing_packages and sys.platform == "darwin":
        print("⚠️  注意: bitsandbytes 在 macOS 上不可用")
        print("   我们将跳过它，使用纯PyTorch进行训练（会慢一些）")
        missing_packages.remove("bitsandbytes")
    
    choice = input("\n是否现在安装？(y/n): ").strip().lower()
    if choice != 'y':
        print("❌ 已取消安装")
        return False
    
    print("\n开始安装...")
    print("=" * 60)
    
    success_count = 0
    for package in missing_packages:
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"安装完成: {success_count}/{len(missing_packages)} 成功")
    
    return success_count == len(missing_packages)


def show_hardware_info():
    """显示硬件信息"""
    print_section("💻 硬件信息")
    
    try:
        import torch
        
        print(f"PyTorch 版本: {torch.__version__}")
        print()
        
        # 检查CUDA
        if torch.cuda.is_available():
            print("✅ CUDA 可用")
            print(f"   设备数量: {torch.cuda.device_count()}")
            print(f"   当前设备: {torch.cuda.current_device()}")
            print(f"   设备名称: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        # 检查MPS (Apple Silicon)
        elif torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) 可用")
            print("   但微调时我们使用CPU（更稳定）")
            device = "cpu"
        else:
            print("⚠️  仅CPU可用")
            print("   训练会比较慢，但可以完成")
            device = "cpu"
        
        print()
        print(f"推荐设备: {device}")
        
    except ImportError:
        print("❌ PyTorch 未安装，无法检测硬件")


def estimate_requirements():
    """估算资源需求"""
    print_section("📊 资源需求估算")
    
    print("微调小型模型 (1.5B参数) 使用LoRA:")
    print("-" * 60)
    print("  内存需求:    ~8-16GB RAM")
    print("  显存需求:    ~4-8GB VRAM (如果用GPU)")
    print("  磁盘空间:    ~10GB (模型 + 数据)")
    print("  训练时间:    CPU: 1-2小时")
    print("               GPU: 10-30分钟")
    print()
    
    print("数据需求:")
    print("-" * 60)
    print("  最少:        100条高质量样本")
    print("  推荐:        500-1000条样本")
    print("  理想:        5000+条样本")
    print()
    
    print("💡 我们的方案:")
    print("  - 使用 Qwen2.5-1.5B (已下载)")
    print("  - LoRA微调（低资源消耗）")
    print("  - 准备300条训练数据")
    print("  - CPU训练（1小时左右）")


def show_next_steps():
    """显示下一步"""
    print_section("🚀 下一步")
    
    print("环境准备完成！接下来你将学习:")
    print()
    print("Step 2: 数据准备")
    print("  - 了解训练数据格式")
    print("  - 创建自己的训练数据集")
    print("  - 保证数据质量")
    print()
    print("Step 3: LoRA微调")
    print("  - 理解LoRA原理")
    print("  - 实战微调模型")
    print("  - 调整超参数")
    print()
    print("命令:")
    print("  cd ../step2_data_preparation")
    print("  python 01_data_format.py")


def main():
    """主函数"""
    print("=" * 60)
    print("Fine-tuning 环境准备".center(60))
    print("=" * 60)
    
    # 1. 检查环境
    missing_packages = check_environment()
    
    # 2. 安装缺失的包
    if missing_packages:
        install_missing_packages(missing_packages)
    
    # 3. 显示硬件信息
    show_hardware_info()
    
    # 4. 估算资源需求
    estimate_requirements()
    
    # 5. 下一步
    show_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已取消")

