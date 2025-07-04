#!/usr/bin/env python3
"""
ManiSkill可视化性能修复脚本
一键解决可视化卡顿问题
"""

import os
import sys
import time
import argparse
import subprocess
from typing import Dict, Any

def check_system_performance():
    """检查系统性能"""
    print("🔍 检查系统性能...")
    
    # 检查GPU状态
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU可用")
            # 提取GPU利用率
            lines = result.stdout.split('\n')
            for line in lines:
                if 'MiB' in line and '%' in line:
                    print(f"   GPU状态: {line.strip()}")
        else:
            print("❌ NVIDIA GPU不可用")
    except FileNotFoundError:
        print("❌ nvidia-smi命令未找到")
    
    # 检查CPU使用率
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        print(f"💻 CPU使用率: {cpu_percent}%")
        print(f"🧠 内存使用率: {memory_percent}%")
        
        if cpu_percent > 80:
            print("⚠️  CPU使用率过高，可能影响渲染性能")
        if memory_percent > 80:
            print("⚠️  内存使用率过高，可能影响渲染性能")
            
    except ImportError:
        print("❌ psutil未安装，无法检查CPU/内存状态")

def fix_visualization_lag():
    """修复可视化卡顿"""
    print("\n🔧 应用可视化性能修复...")
    
    fixes_applied = []
    
    # 1. 创建优化的启动脚本
    optimized_script = """#!/bin/bash
# ManiSkill优化启动脚本 - 解决可视化卡顿

echo "🚀 启动ManiSkill优化训练..."

# 设置环境变量优化
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl  # 使用EGL渲染，避免窗口系统开销
export DISPLAY=:0.0   # 设置显示环境

# 优化参数启动
python ppo_maniskill_training.py \\
    --mode visualize \\
    --num_envs 1 \\
    --render_freq 1000 \\
    --total_timesteps 10000 \\
    --sim_freq 60 \\
    --control_freq 10 \\
    --enable_render

echo "✅ 训练完成"
"""
    
    with open("start_optimized_training.sh", "w") as f:
        f.write(optimized_script)
    os.chmod("start_optimized_training.sh", 0o755)
    fixes_applied.append("创建优化启动脚本")
    
    # 2. 创建无渲染训练脚本
    no_render_script = """#!/bin/bash
# ManiSkill无渲染训练脚本 - 最佳性能

echo "🚀 启动ManiSkill无渲染训练..."

python ppo_maniskill_training.py \\
    --mode train \\
    --num_envs 64 \\
    --total_timesteps 1000000 \\
    --no_render

echo "✅ 训练完成"
"""
    
    with open("start_no_render_training.sh", "w") as f:
        f.write(no_render_script)
    os.chmod("start_no_render_training.sh", 0o755)
    fixes_applied.append("创建无渲染训练脚本")
    
    # 3. 创建性能测试脚本
    test_script = """#!/usr/bin/env python3
import time
import gymnasium as gym
import mani_skill.envs

def test_rendering_performance():
    print("🧪 测试渲染性能...")
    
    # 测试不同配置的性能
    configs = [
        {"render_mode": "none", "desc": "无渲染"},
        {"render_mode": "rgb_array", "camera_width": 64, "camera_height": 64, "desc": "64x64渲染"},
        {"render_mode": "rgb_array", "camera_width": 128, "camera_height": 128, "desc": "128x128渲染"},
        {"render_mode": "rgb_array", "camera_width": 256, "camera_height": 256, "desc": "256x256渲染"},
    ]
    
    for config in configs:
        desc = config.pop("desc")
        print(f"\\n测试: {desc}")
        
        try:
            start_time = time.time()
            
            env = gym.make(
                "StackPickingManiSkill-v1",
                num_envs=1,
                obs_mode="state",
                max_objects=3,
                sim_backend="gpu",
                **config
            )
            
            env.reset()
            
            # 测试10次step
            for i in range(10):
                env.step(env.action_space.sample())
                if config.get("render_mode") == "rgb_array":
                    env.render()
            
            env.close()
            
            total_time = time.time() - start_time
            print(f"   耗时: {total_time:.2f}秒")
            
        except Exception as e:
            print(f"   失败: {e}")

if __name__ == "__main__":
    test_rendering_performance()
"""
    
    with open("test_rendering_performance.py", "w") as f:
        f.write(test_script)
    fixes_applied.append("创建性能测试脚本")
    
    return fixes_applied

def generate_usage_guide():
    """生成使用指南"""
    guide = """
# ManiSkill可视化性能优化指南

## 🚀 快速解决方案

### 1. 最佳实践（推荐）
```bash
# 使用优化的单环境可视化训练
./start_optimized_training.sh
```

### 2. 高性能训练（无可视化）
```bash
# 使用多环境无渲染训练
./start_no_render_training.sh
```

### 3. 性能测试
```bash
# 测试不同渲染配置的性能
python test_rendering_performance.py
```

## 🔧 手动优化参数

### 单环境可视化训练
```bash
python ppo_maniskill_training.py \\
    --mode visualize \\
    --num_envs 1 \\
    --render_freq 1000 \\
    --total_timesteps 10000 \\
    --enable_render
```

### 多环境无渲染训练
```bash
python ppo_maniskill_training.py \\
    --mode train \\
    --num_envs 64 \\
    --total_timesteps 1000000 \\
    --no_render
```

## 🎯 性能优化要点

1. **渲染频率**: 使用较低的render_freq（1000+）
2. **环境数量**: 可视化时使用单环境，训练时使用多环境
3. **分辨率**: 降低相机分辨率到128x128或更低
4. **物体数量**: 减少max_objects到3个
5. **仿真频率**: 降低sim_freq和control_freq

## 🐛 常见问题解决

### 问题1: 单环境仍然卡顿
解决方案:
- 进一步降低渲染频率到2000+
- 使用更低的分辨率（64x64）
- 关闭所有不必要的可视化效果

### 问题2: SSH环境无法显示窗口
解决方案:
- 使用X11转发: `ssh -X username@server`
- 或者使用VNC/远程桌面
- 或者保存图像到文件而不是显示窗口

### 问题3: GPU内存不足
解决方案:
- 减少并行环境数量
- 降低渲染分辨率
- 使用CPU后端: sim_backend="cpu"

## 📊 性能基准

| 配置 | 预期FPS | 内存使用 | 推荐场景 |
|------|---------|----------|----------|
| 无渲染 | 1000+ | 低 | 大规模训练 |
| 64x64渲染 | 100+ | 中 | 快速调试 |
| 128x128渲染 | 50+ | 中 | 正常可视化 |
| 256x256渲染 | 20+ | 高 | 高质量录制 |

记住：可视化主要用于调试和演示，实际训练时建议关闭渲染以获得最佳性能。
"""
    
    with open("PERFORMANCE_GUIDE.md", "w") as f:
        f.write(guide)
    
    return "生成性能优化指南"

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ManiSkill可视化性能修复工具')
    parser.add_argument('--check', action='store_true', help='检查系统性能')
    parser.add_argument('--fix', action='store_true', help='应用性能修复')
    parser.add_argument('--test', action='store_true', help='运行性能测试')
    parser.add_argument('--all', action='store_true', help='执行所有操作')
    
    args = parser.parse_args()
    
    if args.all or args.check:
        check_system_performance()
    
    if args.all or args.fix:
        fixes = fix_visualization_lag()
        print(f"\n✅ 应用了 {len(fixes)} 个修复:")
        for fix in fixes:
            print(f"   - {fix}")
        
        guide_fix = generate_usage_guide()
        print(f"   - {guide_fix}")
    
    if args.all or args.test:
        print("\n🧪 运行性能测试...")
        if os.path.exists("test_rendering_performance.py"):
            os.system("python test_rendering_performance.py")
        else:
            print("❌ 测试脚本不存在，请先运行 --fix")
    
    if not any([args.check, args.fix, args.test, args.all]):
        print("请指定操作: --check, --fix, --test, 或 --all")
        print("使用 --help 查看详细帮助")
    
    print("\n🎉 可视化性能修复完成!")
    print("📖 查看 PERFORMANCE_GUIDE.md 获取详细使用指南")

if __name__ == "__main__":
    main() 