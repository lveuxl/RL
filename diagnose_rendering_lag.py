#!/usr/bin/env python3
"""
ManiSkill渲染性能诊断工具
分析可视化卡顿的根本原因并提供解决方案
"""

import time
import os
import psutil
import torch
import numpy as np
import cv2
import gymnasium as gym
from typing import Dict, Any, List

# ManiSkill相关导入
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


class RenderingPerformanceDiagnostic:
    """渲染性能诊断工具"""
    
    def __init__(self):
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def diagnose_all(self) -> Dict[str, Any]:
        """执行完整诊断"""
        print("🔍 ManiSkill渲染性能诊断开始...")
        print("=" * 60)
        
        # 1. 系统环境检查
        self.results['system'] = self._check_system_environment()
        
        # 2. GPU和CUDA检查
        self.results['gpu'] = self._check_gpu_status()
        
        # 3. ManiSkill环境创建性能
        self.results['env_creation'] = self._test_env_creation_performance()
        
        # 4. 渲染调用性能
        self.results['rendering'] = self._test_rendering_performance()
        
        # 5. 不同渲染模式对比
        self.results['render_modes'] = self._test_different_render_modes()
        
        # 6. 内存使用分析
        self.results['memory'] = self._analyze_memory_usage()
        
        # 7. 生成诊断报告
        self._generate_diagnostic_report()
        
        return self.results
    
    def _check_system_environment(self) -> Dict[str, Any]:
        """检查系统环境"""
        print("1. 系统环境检查...")
        
        system_info = {
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'platform': os.name
        }
        
        # 检查OpenCV
        try:
            cv2_version = cv2.__version__
            system_info['opencv_version'] = cv2_version
        except:
            system_info['opencv_version'] = "未安装"
        
        print(f"   Python: {system_info['python_version']}")
        print(f"   CPU核心: {system_info['cpu_count']}")
        print(f"   内存: {system_info['memory_available_gb']:.1f}GB / {system_info['memory_total_gb']:.1f}GB")
        print(f"   OpenCV: {system_info['opencv_version']}")
        
        return system_info
    
    def _check_gpu_status(self) -> Dict[str, Any]:
        """检查GPU状态"""
        print("\n2. GPU和CUDA检查...")
        
        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': 0,
            'current_device': str(self.device)
        }
        
        if torch.cuda.is_available():
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            gpu_info['cuda_version'] = torch.version.cuda
            
            # GPU内存信息
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            
            gpu_info['memory_allocated_gb'] = memory_allocated
            gpu_info['memory_reserved_gb'] = memory_reserved
            
            print(f"   CUDA可用: ✅")
            print(f"   GPU设备: {gpu_info['device_name']}")
            print(f"   CUDA版本: {gpu_info['cuda_version']}")
            print(f"   GPU内存: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
        else:
            print(f"   CUDA可用: ❌ (可能影响渲染性能)")
        
        return gpu_info
    
    def _test_env_creation_performance(self) -> Dict[str, Any]:
        """测试环境创建性能"""
        print("\n3. 环境创建性能测试...")
        
        creation_times = []
        
        for i in range(3):
            start_time = time.time()
            
            try:
                # 创建最简单的环境
                env = gym.make(
                    "StackPickingManiSkill-v1",
                    num_envs=1,
                    obs_mode="state",
                    render_mode="rgb_array",
                    max_objects=3,
                    sim_backend="gpu" if torch.cuda.is_available() else "cpu"
                )
                
                # 包装环境
                vec_env = ManiSkillVectorEnv(env, 1, ignore_terminations=True)
                
                # 重置环境
                vec_env.reset()
                
                creation_time = time.time() - start_time
                creation_times.append(creation_time)
                
                print(f"   第{i+1}次创建: {creation_time:.2f}秒")
                
                # 关闭环境
                vec_env.close()
                
            except Exception as e:
                print(f"   第{i+1}次创建失败: {e}")
                creation_times.append(float('inf'))
        
        avg_creation_time = np.mean([t for t in creation_times if t != float('inf')])
        
        creation_info = {
            'creation_times': creation_times,
            'avg_creation_time': avg_creation_time,
            'creation_success_rate': len([t for t in creation_times if t != float('inf')]) / len(creation_times)
        }
        
        print(f"   平均创建时间: {avg_creation_time:.2f}秒")
        
        return creation_info
    
    def _test_rendering_performance(self) -> Dict[str, Any]:
        """测试渲染性能"""
        print("\n4. 渲染性能测试...")
        
        try:
            # 创建环境
            env = gym.make(
                "StackPickingManiSkill-v1",
                num_envs=1,
                obs_mode="state",
                render_mode="rgb_array",
                max_objects=3,
                sim_backend="gpu" if torch.cuda.is_available() else "cpu"
            )
            
            vec_env = ManiSkillVectorEnv(env, 1, ignore_terminations=True)
            vec_env.reset()
            
            # 测试多次渲染
            render_times = []
            render_sizes = []
            
            for i in range(10):
                start_time = time.time()
                
                # 执行渲染
                rendered = vec_env.render()
                
                render_time = time.time() - start_time
                render_times.append(render_time)
                
                if rendered is not None:
                    if isinstance(rendered, np.ndarray):
                        render_sizes.append(rendered.shape)
                    elif isinstance(rendered, list) and len(rendered) > 0:
                        render_sizes.append(rendered[0].shape if hasattr(rendered[0], 'shape') else 'unknown')
                
                print(f"   第{i+1}次渲染: {render_time*1000:.1f}ms")
            
            vec_env.close()
            
            avg_render_time = np.mean(render_times)
            max_render_time = np.max(render_times)
            min_render_time = np.min(render_times)
            
            render_info = {
                'render_times': render_times,
                'avg_render_time': avg_render_time,
                'max_render_time': max_render_time,
                'min_render_time': min_render_time,
                'render_sizes': render_sizes,
                'fps': 1.0 / avg_render_time if avg_render_time > 0 else 0
            }
            
            print(f"   平均渲染时间: {avg_render_time*1000:.1f}ms")
            print(f"   渲染FPS: {render_info['fps']:.1f}")
            print(f"   图像尺寸: {render_sizes[0] if render_sizes else 'unknown'}")
            
            return render_info
            
        except Exception as e:
            print(f"   渲染测试失败: {e}")
            return {'error': str(e)}
    
    def _test_different_render_modes(self) -> Dict[str, Any]:
        """测试不同渲染模式"""
        print("\n5. 不同渲染模式对比...")
        
        render_modes = ["rgb_array", "human"]
        mode_results = {}
        
        for mode in render_modes:
            print(f"   测试渲染模式: {mode}")
            
            try:
                start_time = time.time()
                
                env = gym.make(
                    "StackPickingManiSkill-v1",
                    num_envs=1,
                    obs_mode="state",
                    render_mode=mode,
                    max_objects=3,
                    sim_backend="gpu" if torch.cuda.is_available() else "cpu"
                )
                
                vec_env = ManiSkillVectorEnv(env, 1, ignore_terminations=True)
                vec_env.reset()
                
                # 测试几次渲染
                render_times = []
                for _ in range(3):
                    render_start = time.time()
                    vec_env.render()
                    render_times.append(time.time() - render_start)
                
                vec_env.close()
                
                total_time = time.time() - start_time
                avg_render_time = np.mean(render_times)
                
                mode_results[mode] = {
                    'total_time': total_time,
                    'avg_render_time': avg_render_time,
                    'success': True
                }
                
                print(f"     总时间: {total_time:.2f}s, 平均渲染: {avg_render_time*1000:.1f}ms")
                
            except Exception as e:
                print(f"     失败: {e}")
                mode_results[mode] = {
                    'error': str(e),
                    'success': False
                }
        
        return mode_results
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """分析内存使用"""
        print("\n6. 内存使用分析...")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        print(f"   初始内存: {initial_memory:.1f}MB")
        
        try:
            # 创建环境并运行一些步骤
            env = gym.make(
                "StackPickingManiSkill-v1",
                num_envs=1,
                obs_mode="state",
                render_mode="rgb_array",
                max_objects=3,
                sim_backend="gpu" if torch.cuda.is_available() else "cpu"
            )
            
            vec_env = ManiSkillVectorEnv(env, 1, ignore_terminations=True)
            vec_env.reset()
            
            after_creation_memory = process.memory_info().rss / (1024**2)
            print(f"   创建环境后: {after_creation_memory:.1f}MB (+{after_creation_memory-initial_memory:.1f}MB)")
            
            # 执行一些渲染
            for i in range(5):
                vec_env.render()
                current_memory = process.memory_info().rss / (1024**2)
                print(f"   第{i+1}次渲染后: {current_memory:.1f}MB")
            
            final_memory = process.memory_info().rss / (1024**2)
            vec_env.close()
            
            memory_info = {
                'initial_memory_mb': initial_memory,
                'after_creation_mb': after_creation_memory,
                'final_memory_mb': final_memory,
                'creation_overhead_mb': after_creation_memory - initial_memory,
                'total_increase_mb': final_memory - initial_memory
            }
            
            return memory_info
            
        except Exception as e:
            print(f"   内存分析失败: {e}")
            return {'error': str(e)}
    
    def _generate_diagnostic_report(self):
        """生成诊断报告"""
        print("\n" + "=" * 60)
        print("🔧 诊断报告和建议")
        print("=" * 60)
        
        # 系统建议
        if not self.results['gpu']['cuda_available']:
            print("⚠️  CUDA不可用 - 建议:")
            print("   • 安装支持CUDA的PyTorch版本")
            print("   • 检查GPU驱动程序")
            print("   • 使用GPU加速可显著提升渲染性能")
        
        # 渲染性能建议
        if 'rendering' in self.results and 'avg_render_time' in self.results['rendering']:
            avg_time = self.results['rendering']['avg_render_time']
            
            if avg_time > 0.1:  # 100ms
                print("❌ 渲染性能较差 - 建议:")
                print("   • 降低渲染分辨率")
                print("   • 减少渲染频率")
                print("   • 使用rgb_array而非human模式")
                print("   • 减少场景复杂度（物体数量）")
            elif avg_time > 0.05:  # 50ms
                print("⚠️  渲染性能一般 - 建议:")
                print("   • 考虑降低渲染频率")
                print("   • 使用异步渲染")
            else:
                print("✅ 渲染性能良好")
        
        # 内存使用建议
        if 'memory' in self.results and 'total_increase_mb' in self.results['memory']:
            memory_increase = self.results['memory']['total_increase_mb']
            
            if memory_increase > 500:  # 500MB
                print("⚠️  内存使用较高 - 建议:")
                print("   • 定期清理GPU缓存")
                print("   • 减少并行环境数量")
                print("   • 检查是否存在内存泄漏")
        
        # 具体优化建议
        print("\n🚀 针对可视化卡顿的具体优化建议:")
        print("1. 使用超轻量级可视化回调")
        print("2. 设置渲染频率为500步以上")
        print("3. 限制最大FPS为5")
        print("4. 使用rgb_array模式而非human模式")
        print("5. 降低渲染分辨率至256x256")
        print("6. 减少场景物体数量至6个以下")


def main():
    """主函数"""
    print("ManiSkill渲染性能诊断工具")
    print("分析可视化卡顿问题并提供解决方案")
    
    diagnostic = RenderingPerformanceDiagnostic()
    results = diagnostic.diagnose_all()
    
    # 保存诊断结果
    import json
    with open('rendering_diagnostic_report.json', 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_for_json(obj)
        
        json.dump(recursive_convert(results), f, indent=2)
    
    print(f"\n📄 完整诊断报告已保存至: rendering_diagnostic_report.json")


if __name__ == "__main__":
    main() 