#!/usr/bin/env python3
"""
测试超轻量级可视化系统
专门验证单环境可视化卡顿问题是否彻底解决
"""

import time
import argparse
import psutil
import os
from ppo_maniskill_training import PPOTrainingConfig, train_ppo_model


def monitor_system_resources():
    """监控系统资源使用情况"""
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    return {
        'cpu_percent': cpu_percent,
        'memory_mb': memory_mb,
        'memory_percent': process.memory_percent()
    }


def test_ultra_lightweight_single_env():
    """测试超轻量级单环境可视化"""
    print("=" * 60)
    print("测试：超轻量级单环境可视化")
    print("=" * 60)
    
    config = PPOTrainingConfig()
    config.num_envs = 1  # 单环境
    config.total_timesteps = 1000  # 短时间测试
    config.enable_render = True
    config.render_freq = 100  # 中等频率测试
    
    print("配置:")
    print(f"  环境数量: {config.num_envs}")
    print(f"  渲染频率: 每{config.render_freq}步")
    print(f"  总步数: {config.total_timesteps}")
    print("  可视化类型: 超轻量级（5FPS限制）")
    print("  预期效果: 无明显卡顿，流畅可视化")
    
    # 记录开始状态
    start_resources = monitor_system_resources()
    start_time = time.time()
    
    print(f"\n开始测试... (开始时间: {time.strftime('%H:%M:%S')})")
    print(f"初始资源使用: CPU {start_resources['cpu_percent']:.1f}%, 内存 {start_resources['memory_mb']:.1f}MB")
    
    try:
        # 运行训练
        model, run_number = train_ppo_model(config)
        
        end_time = time.time()
        end_resources = monitor_system_resources()
        
        # 计算结果
        total_time = end_time - start_time
        steps_per_second = config.total_timesteps / total_time
        expected_renders = config.total_timesteps // config.render_freq
        avg_render_time = total_time / expected_renders if expected_renders > 0 else 0
        
        print(f"\n测试完成! (结束时间: {time.strftime('%H:%M:%S')})")
        print("=" * 60)
        print("测试结果:")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  训练速度: {steps_per_second:.1f} steps/s")
        print(f"  预计渲染次数: {expected_renders}")
        print(f"  平均每次渲染耗时: {avg_render_time:.3f}秒")
        
        print(f"\n资源使用对比:")
        print(f"  CPU使用: {start_resources['cpu_percent']:.1f}% → {end_resources['cpu_percent']:.1f}%")
        print(f"  内存使用: {start_resources['memory_mb']:.1f}MB → {end_resources['memory_mb']:.1f}MB")
        
        # 性能评估
        if avg_render_time < 0.05:
            print("\n✅ 性能评估: 优秀 - 渲染非常流畅")
        elif avg_render_time < 0.1:
            print("\n✅ 性能评估: 良好 - 渲染基本流畅")
        elif avg_render_time < 0.2:
            print("\n⚠️  性能评估: 一般 - 有轻微延迟")
        else:
            print("\n❌ 性能评估: 较差 - 仍有明显卡顿")
        
        return {
            'total_time': total_time,
            'steps_per_second': steps_per_second,
            'avg_render_time': avg_render_time,
            'resource_usage': {
                'start': start_resources,
                'end': end_resources
            }
        }
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_different_render_frequencies():
    """测试不同渲染频率的性能"""
    print("\n" + "=" * 60)
    print("测试：不同渲染频率性能对比")
    print("=" * 60)
    
    test_configs = [
        {"freq": 50, "name": "高频率"},
        {"freq": 100, "name": "中频率"},
        {"freq": 200, "name": "低频率"},
        {"freq": 500, "name": "极低频率"}
    ]
    
    results = []
    
    for test_config in test_configs:
        print(f"\n--- 测试: {test_config['name']} (每{test_config['freq']}步) ---")
        
        config = PPOTrainingConfig()
        config.num_envs = 1
        config.total_timesteps = 500  # 更短的测试
        config.enable_render = True
        config.render_freq = test_config['freq']
        
        start_time = time.time()
        
        try:
            train_ppo_model(config)
            end_time = time.time()
            
            total_time = end_time - start_time
            steps_per_second = config.total_timesteps / total_time
            
            result = {
                'name': test_config['name'],
                'freq': test_config['freq'],
                'total_time': total_time,
                'steps_per_second': steps_per_second
            }
            results.append(result)
            
            print(f"结果: {total_time:.2f}s, {steps_per_second:.1f} steps/s")
            
        except Exception as e:
            print(f"测试失败: {e}")
            continue
    
    # 输出对比结果
    if results:
        print(f"\n{'='*60}")
        print("渲染频率性能对比:")
        print(f"{'='*60}")
        print(f"{'配置':<10} {'频率':<8} {'耗时(s)':<10} {'速度(steps/s)':<15}")
        print("-" * 50)
        
        for result in results:
            print(f"{result['name']:<10} {result['freq']:<8} {result['total_time']:<10.2f} {result['steps_per_second']:<15.1f}")
    
    return results


def test_memory_leak():
    """测试内存泄漏情况"""
    print("\n" + "=" * 60)
    print("测试：内存泄漏检测")
    print("=" * 60)
    
    print("运行多轮短期训练，监控内存使用变化...")
    
    memory_usage = []
    
    for round_num in range(5):
        print(f"\n第 {round_num + 1}/5 轮测试...")
        
        # 记录测试前内存
        before_resources = monitor_system_resources()
        
        config = PPOTrainingConfig()
        config.num_envs = 1
        config.total_timesteps = 200  # 很短的测试
        config.enable_render = True
        config.render_freq = 50
        
        try:
            train_ppo_model(config)
        except Exception as e:
            print(f"轮次 {round_num + 1} 失败: {e}")
            continue
        
        # 记录测试后内存
        after_resources = monitor_system_resources()
        
        memory_diff = after_resources['memory_mb'] - before_resources['memory_mb']
        memory_usage.append(after_resources['memory_mb'])
        
        print(f"内存使用: {before_resources['memory_mb']:.1f}MB → {after_resources['memory_mb']:.1f}MB (差异: {memory_diff:+.1f}MB)")
    
    if len(memory_usage) >= 3:
        memory_trend = memory_usage[-1] - memory_usage[0]
        print(f"\n内存趋势分析:")
        print(f"  起始内存: {memory_usage[0]:.1f}MB")
        print(f"  结束内存: {memory_usage[-1]:.1f}MB")
        print(f"  总体变化: {memory_trend:+.1f}MB")
        
        if abs(memory_trend) < 50:
            print("✅ 内存使用稳定，无明显泄漏")
        elif abs(memory_trend) < 100:
            print("⚠️  内存使用略有增加，需关注")
        else:
            print("❌ 检测到可能的内存泄漏")


def main():
    parser = argparse.ArgumentParser(description='测试超轻量级可视化系统')
    parser.add_argument('--test', type=str, default='single',
                       choices=['single', 'frequency', 'memory', 'all'],
                       help='测试类型')
    
    args = parser.parse_args()
    
    print("超轻量级可视化测试工具")
    print("专门验证单环境可视化卡顿问题的解决效果")
    
    if args.test == 'single':
        test_ultra_lightweight_single_env()
    elif args.test == 'frequency':
        test_different_render_frequencies()
    elif args.test == 'memory':
        test_memory_leak()
    elif args.test == 'all':
        test_ultra_lightweight_single_env()
        test_different_render_frequencies()
        test_memory_leak()


if __name__ == "__main__":
    main() 