#!/usr/bin/env python3
"""
堆叠抓取训练启动脚本
使用方法:
  python run_training.py           # 开始训练
  python run_training.py --eval    # 评估模型
"""

import os
import sys
import subprocess

def main():
    """主启动函数"""
    print("=== 堆叠抓取强化学习训练系统 ===")
    
    # 检查是否在正确目录
    if not os.path.exists("train_optimized.py"):
        print("错误: 请在包含train_optimized.py的目录下运行此脚本")
        sys.exit(1)
    
    # 检查环境
    try:
        import torch
        import stable_baselines3
        import mani_skill
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ Stable-Baselines3版本: {stable_baselines3.__version__}")
        print(f"✓ ManiSkill环境: OK")
        if torch.cuda.is_available():
            print(f"✓ GPU可用: {torch.cuda.get_device_name()}")
        else:
            print("! GPU不可用，将使用CPU训练")
    except ImportError as e:
        print(f"错误: 缺少依赖库 {e}")
        print("请安装所需依赖: pip install torch stable-baselines3 mani-skill")
        sys.exit(1)
    
    # 解析参数
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        mode = "eval"
        print("\n=== 评估模式 ===")
        print("将评估已训练的模型...")
    else:
        mode = "train"
        print("\n=== 训练模式 ===")
        print("配置信息:")
        print("- 环境: EnvClutterOptimized-v1 (9物体堆叠抓取)")
        print("- 算法: PPO (Proximal Policy Optimization)")
        print("- 并行环境: 64个")
        print("- 总训练步数: 2,000,000步")
        print("- 预计训练时间: ~2-4小时 (GPU)")
        print("- Tensorboard日志: ./tensorboard_logs/")
        print("- 模型保存路径: ./models/")
    
    # 构建命令
    cmd = [sys.executable, "train_optimized.py", "--mode", mode]
    
    if torch.cuda.is_available():
        cmd.append("--gpu")
    
    print(f"\n执行命令: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # 运行训练脚本
        subprocess.run(cmd, check=True)
        
        if mode == "train":
            print("\n" + "=" * 50)
            print("🎉 训练完成！")
            print("📊 查看训练曲线: tensorboard --logdir ./tensorboard_logs")
            print("🤖 最佳模型已保存到: ./models/best_model")
            print("🧪 运行评估: python run_training.py --eval")
        else:
            print("\n" + "=" * 50)
            print("✅ 评估完成！")
            
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败，退出码: {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
