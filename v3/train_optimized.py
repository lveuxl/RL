#!/usr/bin/env python3
"""
优化的训练启动脚本 - 一键启动完整的抓取顺序学习训练
"""

import os
import sys
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """主函数 - 启动优化训练"""
    parser = argparse.ArgumentParser(description='启动优化的抓取顺序学习训练')
    
    # 快速启动选项
    parser.add_argument('--quick', action='store_true', 
                       help='快速测试模式（较少步数和环境）')
    parser.add_argument('--full', action='store_true', 
                       help='完整训练模式（推荐）')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式（单环境，详细日志）')
    
    # 自定义参数
    parser.add_argument('--total_timesteps', type=int, 
                       help='覆盖默认训练步数')
    parser.add_argument('--num_envs', type=int,
                       help='覆盖默认环境数量')
    parser.add_argument('--log_dir', type=str, default='./logs/optimized_training',
                       help='日志目录')
    parser.add_argument('--model_dir', type=str, default='./models/optimized_training',
                       help='模型保存目录')
    
    args = parser.parse_args()
    
    # 根据模式设置参数
    if args.quick:
        total_timesteps = args.total_timesteps or 100_000
        num_envs = args.num_envs or 8
        eval_freq = 10_000
        save_freq = 25_000
        print("🚀 快速测试模式")
    elif args.debug:
        total_timesteps = args.total_timesteps or 50_000
        num_envs = args.num_envs or 1
        eval_freq = 5_000
        save_freq = 10_000
        print("🐛 调试模式")
    else:  # 完整模式（默认）
        total_timesteps = args.total_timesteps or 2_000_000
        num_envs = args.num_envs or 32
        eval_freq = 50_000
        save_freq = 100_000
        print("💪 完整训练模式")
    
    print(f"训练配置:")
    print(f"  总步数: {total_timesteps:,}")
    print(f"  环境数: {num_envs}")
    print(f"  日志目录: {args.log_dir}")
    print(f"  模型目录: {args.model_dir}")
    
    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 构建训练命令
    cmd_args = [
        '--mode', 'train',
        '--total_timesteps', str(total_timesteps),
        '--num_envs', str(num_envs),
        '--eval_freq', str(eval_freq),
        '--save_freq', str(save_freq),
        '--log_dir', args.log_dir,
        '--model_dir', args.model_dir,
        '--record_video'  # 启用视频录制
    ]
    
    # 调用优化的训练脚本
    try:
        from training import main as train_main
        
        # 模拟命令行参数
        original_argv = sys.argv
        sys.argv = ['training.py'] + cmd_args
        
        print("\n开始训练...")
        print("=" * 60)
        train_main()
        
        # 恢复原始参数
        sys.argv = original_argv
        
        print("\n训练完成！")
        print(f"模型保存在: {args.model_dir}")
        print(f"日志保存在: {args.log_dir}")
        print(f"可以使用以下命令进行推理:")
        print(f"  python inference.py --model_path {args.model_dir}/ppo_envclutter_final.zip --mode demo")
        
    except KeyboardInterrupt:
        print("\n用户中断训练")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()