#!/usr/bin/env python3
"""
测试优化可视化系统的性能
比较原始方法和优化方法的差异
"""

import time
import argparse
from ppo_maniskill_training import PPOTrainingConfig, train_ppo_model


def test_original_visualization():
    """测试原始可视化方法"""
    print("=== 测试原始可视化方法 ===")
    
    config = PPOTrainingConfig()
    config.num_envs = 1
    config.total_timesteps = 2000  # 短时间测试
    config.enable_render = True
    config.render_freq = 50  # 较高频率
    
    print("配置:")
    print(f"  环境数量: {config.num_envs}")
    print(f"  渲染频率: 每{config.render_freq}步")
    print(f"  总步数: {config.total_timesteps}")
    print(f"  预计渲染次数: {config.total_timesteps // config.render_freq}")
    
    start_time = time.time()
    train_ppo_model(config)
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每步耗时: {total_time/config.total_timesteps*1000:.2f}ms")
    
    return total_time


def test_optimized_visualization():
    """测试优化可视化方法"""
    print("\n=== 测试优化可视化方法 ===")
    
    config = PPOTrainingConfig()
    config.num_envs = 1
    config.total_timesteps = 2000  # 短时间测试
    config.enable_render = True
    config.render_freq = 200  # 降低频率
    
    print("配置:")
    print(f"  环境数量: {config.num_envs}")
    print(f"  渲染频率: 每{config.render_freq}步")
    print(f"  总步数: {config.total_timesteps}")
    print(f"  预计渲染次数: {config.total_timesteps // config.render_freq}")
    print("  优化特性: 异步处理 + 无磁盘I/O")
    
    start_time = time.time()
    train_ppo_model(config)
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每步耗时: {total_time/config.total_timesteps*1000:.2f}ms")
    
    return total_time


def performance_comparison():
    """性能对比测试"""
    print("=" * 60)
    print("可视化性能对比测试")
    print("=" * 60)
    
    # 测试不同配置的性能
    test_configs = [
        {
            "name": "无可视化基准",
            "num_envs": 1,
            "render_freq": 0,  # 不渲染
            "enable_render": False,
            "total_timesteps": 1000
        },
        {
            "name": "高频率可视化",
            "num_envs": 1,
            "render_freq": 10,
            "enable_render": True,
            "total_timesteps": 1000
        },
        {
            "name": "优化可视化",
            "num_envs": 1,
            "render_freq": 100,
            "enable_render": True,
            "total_timesteps": 1000
        },
        {
            "name": "多环境优化可视化",
            "num_envs": 4,
            "render_freq": 200,
            "enable_render": True,
            "total_timesteps": 1000
        }
    ]
    
    results = []
    
    for test_config in test_configs:
        print(f"\n--- 测试: {test_config['name']} ---")
        
        config = PPOTrainingConfig()
        config.num_envs = test_config['num_envs']
        config.total_timesteps = test_config['total_timesteps']
        config.enable_render = test_config['enable_render']
        config.render_freq = test_config['render_freq']
        
        print(f"环境数: {config.num_envs}, 渲染频率: {config.render_freq}, 可视化: {config.enable_render}")
        
        start_time = time.time()
        try:
            train_ppo_model(config)
        except Exception as e:
            print(f"测试失败: {e}")
            continue
        end_time = time.time()
        
        total_time = end_time - start_time
        steps_per_second = config.total_timesteps / total_time
        
        result = {
            "name": test_config['name'],
            "total_time": total_time,
            "steps_per_second": steps_per_second,
            "render_overhead": 0  # 将在后面计算
        }
        
        results.append(result)
        print(f"耗时: {total_time:.2f}s, 训练速度: {steps_per_second:.1f} steps/s")
    
    # 计算渲染开销
    if results:
        baseline_speed = results[0]['steps_per_second']  # 无可视化基准
        
        print(f"\n{'='*60}")
        print("性能对比结果:")
        print(f"{'='*60}")
        print(f"{'测试名称':<20} {'耗时(s)':<10} {'速度(steps/s)':<15} {'性能损失':<10}")
        print("-" * 60)
        
        for result in results:
            overhead = (baseline_speed - result['steps_per_second']) / baseline_speed * 100
            print(f"{result['name']:<20} {result['total_time']:<10.2f} {result['steps_per_second']:<15.1f} {overhead:<10.1f}%")
    
    return results


def quick_visualization_test():
    """快速可视化测试"""
    print("=== 快速可视化测试 ===")
    print("测试单环境可视化是否仍然卡顿")
    
    config = PPOTrainingConfig()
    config.num_envs = 1  # 单环境
    config.total_timesteps = 500  # 很短的测试
    config.enable_render = True
    config.render_freq = 50  # 中等频率
    
    print(f"配置: {config.num_envs}环境, 每{config.render_freq}步渲染一次")
    print("观察是否还有卡顿现象...")
    
    start_time = time.time()
    train_ppo_model(config)
    end_time = time.time()
    
    total_time = end_time - start_time
    expected_renders = config.total_timesteps // config.render_freq
    avg_render_time = total_time / expected_renders if expected_renders > 0 else 0
    
    print(f"\n测试结果:")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"渲染次数: {expected_renders}")
    print(f"平均每次渲染耗时: {avg_render_time:.3f}秒")
    
    if avg_render_time < 0.1:
        print("✅ 渲染性能良好，无明显卡顿")
    elif avg_render_time < 0.5:
        print("⚠️  渲染有轻微延迟")
    else:
        print("❌ 渲染仍有明显卡顿")


def main():
    parser = argparse.ArgumentParser(description='测试优化可视化系统')
    parser.add_argument('--test', type=str, default='quick',
                       choices=['quick', 'comparison', 'all'],
                       help='测试类型')
    
    args = parser.parse_args()
    
    if args.test == 'quick':
        quick_visualization_test()
    elif args.test == 'comparison':
        performance_comparison()
    elif args.test == 'all':
        quick_visualization_test()
        print("\n" + "="*60 + "\n")
        performance_comparison()


if __name__ == "__main__":
    main() 