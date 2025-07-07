#!/usr/bin/env python3
"""
EnvClutter 项目启动脚本
提供简单的命令行接口来运行不同的功能
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_test():
    """运行测试"""
    print("🧪 运行环境测试...")
    subprocess.run([sys.executable, "test_env.py"])

def run_training(args):
    """运行训练"""
    print("🏋️ 开始训练...")
    
    cmd = [sys.executable, "training.py"]
    
    # 添加参数
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.num_envs:
        cmd.extend(["--num_envs", str(args.num_envs)])
    if args.config:
        cmd.extend(["--config", args.config])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.log_dir:
        cmd.extend(["--log_dir", args.log_dir])
    
    subprocess.run(cmd)

def run_inference(args):
    """运行推理"""
    print("🔮 开始推理...")
    
    if not args.model_path:
        print("❌ 需要指定模型路径 (--model_path)")
        return
    
    cmd = [sys.executable, "inference.py", "--model_path", args.model_path]
    
    # 添加参数
    if args.mode:
        cmd.extend(["--mode", args.mode])
    if args.num_episodes:
        cmd.extend(["--num_episodes", str(args.num_episodes)])
    if args.render:
        cmd.append("--render")
    if args.record_video:
        cmd.append("--record_video")
    if args.video_dir:
        cmd.extend(["--video_dir", args.video_dir])
    
    subprocess.run(cmd)

def setup_directories():
    """创建必要的目录"""
    directories = [
        "logs",
        "models", 
        "videos",
        "results",
        "configs"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("📁 目录创建完成")

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    try:
        import torch
        import gymnasium
        import numpy as np
        import mani_skill
        print("✅ 核心依赖检查通过")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA可用 (版本: {torch.version.cuda})")
        else:
            print("⚠️  CUDA不可用，将使用CPU")
            
    except ImportError as e:
        print(f"❌ 依赖检查失败: {e}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def show_help():
    """显示帮助信息"""
    help_text = """
🤖 EnvClutter 环境使用指南

基本命令:
  python run.py test                    # 运行环境测试
  python run.py setup                   # 设置项目目录
  python run.py check                   # 检查依赖
  python run.py train                   # 开始训练
  python run.py infer --model_path <路径> # 运行推理

训练选项:
  --epochs 1000                         # 训练轮数
  --num_envs 8                          # 并行环境数
  --config default                      # 配置预设
  --device cuda                         # 使用设备

推理选项:
  --model_path <路径>                   # 模型文件路径 (必需)
  --mode demo                           # 运行模式 (demo/eval/benchmark)
  --num_episodes 100                    # 评估episode数
  --render                              # 显示渲染
  --record_video                        # 录制视频

示例:
  # 快速开始
  python run.py setup
  python run.py test
  python run.py train --epochs 100 --num_envs 4
  
  # 训练完成后推理
  python run.py infer --model_path models/ppo_model.pth --mode demo --render
  
  # 批量评估
  python run.py infer --model_path models/ppo_model.pth --mode eval --num_episodes 100

配置文件:
  config.py                             # 主配置文件
  
输出目录:
  logs/                                 # 训练日志
  models/                               # 保存的模型
  videos/                               # 录制的视频
  results/                              # 评估结果

更多信息请查看 README.md
"""
    print(help_text)

def main():
    parser = argparse.ArgumentParser(description='EnvClutter 项目启动脚本')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='运行环境测试')
    
    # 设置命令
    setup_parser = subparsers.add_parser('setup', help='设置项目目录')
    
    # 检查命令
    check_parser = subparsers.add_parser('check', help='检查依赖')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='开始训练')
    train_parser.add_argument('--epochs', type=int, help='训练轮数')
    train_parser.add_argument('--num_envs', type=int, help='并行环境数')
    train_parser.add_argument('--config', type=str, help='配置预设')
    train_parser.add_argument('--device', type=str, help='使用设备')
    train_parser.add_argument('--log_dir', type=str, help='日志目录')
    
    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='运行推理')
    infer_parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    infer_parser.add_argument('--mode', type=str, default='demo', 
                             choices=['demo', 'eval', 'benchmark', 'interactive'],
                             help='运行模式')
    infer_parser.add_argument('--num_episodes', type=int, help='评估episode数')
    infer_parser.add_argument('--render', action='store_true', help='显示渲染')
    infer_parser.add_argument('--record_video', action='store_true', help='录制视频')
    infer_parser.add_argument('--video_dir', type=str, help='视频目录')
    
    # 帮助命令
    help_parser = subparsers.add_parser('help', help='显示详细帮助')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        run_test()
    elif args.command == 'setup':
        setup_directories()
    elif args.command == 'check':
        check_dependencies()
    elif args.command == 'train':
        if check_dependencies():
            run_training(args)
    elif args.command == 'infer':
        if check_dependencies():
            run_inference(args)
    elif args.command == 'help':
        show_help()
    else:
        # 显示基本帮助
        print("🤖 EnvClutter 环境")
        print("使用 'python run.py help' 查看详细帮助")
        print("使用 'python run.py <command> --help' 查看命令帮助")
        print("\n可用命令:")
        print("  test     - 运行环境测试")
        print("  setup    - 设置项目目录")
        print("  check    - 检查依赖")
        print("  train    - 开始训练")
        print("  infer    - 运行推理")
        print("  help     - 显示详细帮助")

if __name__ == "__main__":
    main() 