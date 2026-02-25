#!/usr/bin/env python3
"""
性能测试和比较脚本
"""

import time
import subprocess
import sys

def test_environment_speed():
    """测试环境创建和执行速度"""
    print("=== 环境速度测试 ===")
    
    try:
        import torch
        import gymnasium as gym
        from env_clutter_optimized import EnvClutterOptimizedEnv
        import mani_skill.envs
        
        # 测试环境创建速度
        print("1. 测试环境创建速度...")
        start_time = time.time()
        
        env = gym.make(
            "EnvClutterOptimized-v1",
            num_envs=32,
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            reward_mode="dense",
            sim_backend="gpu",
        )
        
        create_time = time.time() - start_time
        print(f"   ✓ 环境创建: {create_time:.2f}秒")
        
        # 测试重置速度
        print("2. 测试环境重置速度...")
        reset_start = time.time()
        obs, info = env.reset()
        reset_time = time.time() - reset_start
        print(f"   ✓ 环境重置: {reset_time:.2f}秒")
        
        # 测试步骤执行速度
        print("3. 测试步骤执行速度...")
        step_times = []
        
        for i in range(50):
            actions = torch.randint(0, 9, (32,))
            step_start = time.time()
            obs, rewards, terminated, truncated, info = env.step(actions)
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            if i % 10 == 0:
                print(f"   步骤 {i}: {step_time*1000:.1f}ms")
        
        avg_step_time = sum(step_times) / len(step_times)
        steps_per_second = 1.0 / avg_step_time
        
        print(f"   ✓ 平均步骤时间: {avg_step_time*1000:.1f}ms")
        print(f"   ✓ 步骤执行速度: {steps_per_second:.0f} steps/s")
        
        env.close()
        
        return steps_per_second
        
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return 0


def compare_training_speeds():
    """比较不同配置的训练速度"""
    print("\n=== 训练速度对比测试 ===")
    
    configs = [
        ("CPU极速版", ["python", "train_high_speed.py"]),
        ("GPU极速版", ["python", "train_high_speed.py", "--gpu"]),
    ]
    
    results = {}
    
    for name, cmd in configs:
        print(f"\n🚀 测试 {name}...")
        try:
            start_time = time.time()
            
            # 修改为短时间测试
            test_cmd = cmd + ["--test"] if "--test" not in cmd else cmd
            result = subprocess.run(test_cmd, timeout=300, capture_output=True, text=True)
            
            end_time = time.time()
            
            if result.returncode == 0:
                duration = end_time - start_time
                results[name] = duration
                print(f"   ✓ {name}: {duration:.1f}秒")
            else:
                print(f"   ❌ {name}: 失败")
                print(f"   错误: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {name}: 超时")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    if results:
        print(f"\n📊 速度对比结果:")
        fastest = min(results.items(), key=lambda x: x[1])
        for name, time in results.items():
            speedup = fastest[1] / time if time > 0 else 0
            print(f"   {name}: {time:.1f}秒 ({speedup:.1f}x)")
        print(f"   🏆 最快: {fastest[0]}")


def run_quick_training_test():
    """运行快速训练测试"""
    print("\n=== 快速训练测试 ===")
    print("运行1万步训练测试...")
    
    try:
        # 修改训练脚本做短期测试
        cmd = ["python", "-c", """
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 使用CPU
exec(open('train_high_speed.py').read().replace('100_000', '1000'))  # 1000步测试
"""]
        
        start_time = time.time()
        result = subprocess.run(cmd, timeout=120, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode == 0:
            duration = end_time - start_time
            print(f"✓ 1000步训练耗时: {duration:.1f}秒")
            print(f"🚀 预估10万步耗时: {duration*100/60:.1f}分钟")
        else:
            print("❌ 快速训练测试失败")
            print(result.stderr[:300])
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def main():
    """主测试函数"""
    print("🔥 极速训练性能测试")
    print("="*50)
    
    # 1. 环境速度测试
    env_speed = test_environment_speed()
    
    if env_speed > 0:
        print(f"\n💡 环境性能: {env_speed:.0f} steps/s")
        if env_speed > 50:
            print("✅ 环境性能优秀！")
        elif env_speed > 20:
            print("⚡ 环境性能良好")
        else:
            print("⚠️  环境性能需要优化")
    
    # 2. 快速训练测试
    run_quick_training_test()
    
    print(f"\n🎯 建议配置:")
    print("- 使用CPU训练（对MLP策略更快）")
    print("- 环境数量: 128-256个")
    print("- 批次大小: 4096-8192")
    print("- 预计速度: 50-200 steps/s")
    
    print(f"\n🚀 开始极速训练:")
    print("python train_high_speed.py")


if __name__ == "__main__":
    main()

