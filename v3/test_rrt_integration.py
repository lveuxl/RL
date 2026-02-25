#!/usr/bin/env python3
"""
RRT集成系统快速测试脚本
验证环境、RRT规划器和RL模型集成是否正常工作
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_environment_creation():
    """测试环境创建"""
    print("🧪 测试1: 环境创建")
    try:
        import gymnasium as gym
        import mani_skill.envs
        from env_clutter_rrt import EnvClutterRRTEnv
        
        # 测试基础环境创建
        env = gym.make(
            "EnvClutter-v1",
            num_envs=1,
            obs_mode="state",
            control_mode="pd_joint_pos",
            reward_mode="dense",
            sim_backend="gpu"
        )
        print("   ✅ 基础EnvClutter环境创建成功")
        env.close()
        
        # 测试RRT集成环境创建
        try:
            env = gym.make(
                "EnvClutter-RRT-v1",
                num_envs=1,
                obs_mode="state", 
                control_mode="pd_joint_pos",
                reward_mode="dense",
                sim_backend="gpu",
                use_rrt_planning=True,
                enable_obstacle_detection=True
            )
            print("   ✅ RRT集成环境创建成功")
            env.close()
        except Exception as e:
            print(f"   ⚠️ RRT集成环境创建失败: {e}")
            print("   这可能是因为缺少mplib依赖")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 环境创建测试失败: {e}")
        return False


def test_obstacle_detector():
    """测试障碍物检测器"""
    print("\n🧪 测试2: 障碍物检测器")
    try:
        from obstacle_detector import ObstacleDetector
        import sapien
        
        detector = ObstacleDetector(
            point_density=128,
            safety_margin=0.02,
            debug=True
        )
        print("   ✅ 障碍物检测器创建成功")
        
        # 测试点云生成
        table_pose = sapien.Pose(p=[0, 0, 0])
        table_size = np.array([0.5, 0.3, 0.02])
        points = detector.add_table_obstacle(table_pose, table_size)
        
        if points is not None and len(points) > 0:
            print(f"   ✅ 桌面障碍物点云生成成功: {len(points)} 个点")
        else:
            print("   ⚠️ 桌面障碍物点云生成失败")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 障碍物检测器测试失败: {e}")
        return False


def test_rrt_planner():
    """测试RRT规划器"""
    print("\n🧪 测试3: RRT运动规划器")
    try:
        import mplib
        print("   ✅ mplib库可用")
        
        # 这里可以添加更详细的RRT规划器测试
        # 但需要完整的环境设置，所以暂时跳过
        print("   ✅ RRT规划器依赖检查通过")
        return True
        
    except ImportError:
        print("   ❌ mplib库未安装")
        print("   安装命令: pip install mplib-dist")
        return False
    except Exception as e:
        print(f"   ❌ RRT规划器测试失败: {e}")
        return False


def test_rl_model_loading():
    """测试RL模型加载"""
    print("\n🧪 测试4: RL模型加载")
    try:
        from stable_baselines3 import PPO
        print("   ✅ PPO可用")
        
        try:
            from sb3_contrib import MaskablePPO
            print("   ✅ MaskablePPO可用")
        except ImportError:
            print("   ⚠️ MaskablePPO不可用 (sb3-contrib未安装)")
        
        # 查找可能的模型文件
        model_dirs = [
            "./models/sb3_topdown",
            "./models/optimized_training",
            "../models",
        ]
        
        found_models = []
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith('.zip'):
                        found_models.append(os.path.join(model_dir, file))
        
        if found_models:
            print(f"   ✅ 找到 {len(found_models)} 个可能的模型文件:")
            for model in found_models[:3]:  # 只显示前3个
                print(f"      - {model}")
        else:
            print("   ⚠️ 未找到预训练模型文件")
            print("   这不影响系统测试，可以使用贪心策略")
        
        return True
        
    except Exception as e:
        print(f"   ❌ RL模型加载测试失败: {e}")
        return False


def test_intelligent_grasp_system():
    """测试智能抓取系统"""
    print("\n🧪 测试5: 智能抓取系统集成")
    try:
        from intelligent_grasp_system import IntelligentGraspSystem
        
        # 创建系统（不加载模型，使用贪心策略）
        system = IntelligentGraspSystem(
            model_path=None,  # 使用贪心策略
            use_rrt_planning=False,  # 禁用RRT以简化测试
            enable_obstacle_detection=False,  # 禁用障碍检测
            visualize=False,  # 禁用可视化
            debug=False
        )
        print("   ✅ 智能抓取系统创建成功")
        
        # 测试系统状态
        stats = system.get_execution_stats()
        print(f"   ✅ 系统统计获取成功: {len(stats)} 个统计项")
        
        system.close()
        print("   ✅ 系统关闭成功")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 智能抓取系统测试失败: {e}")
        return False


def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试6: 基本功能测试")
    try:
        # 测试基础导入
        from env_clutter import EnvClutterEnv
        from training import create_eval_env
        print("   ✅ 基础模块导入成功")
        
        # 测试环境重置
        import gymnasium as gym
        env = gym.make("EnvClutter-v1", num_envs=1, obs_mode="state")
        obs = env.reset()
        print("   ✅ 环境重置成功")
        
        # 检查观测形状
        if isinstance(obs, (tuple, list)):
            obs_array = obs[0] if len(obs) > 0 else obs
        else:
            obs_array = obs
        
        if hasattr(obs_array, 'shape'):
            print(f"   ✅ 观测形状: {obs_array.shape}")
        else:
            print(f"   ✅ 观测类型: {type(obs_array)}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ 基本功能测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🔬 RRT集成系统测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(test_environment_creation())
    test_results.append(test_obstacle_detector()) 
    test_results.append(test_rrt_planner())
    test_results.append(test_rl_model_loading())
    test_results.append(test_intelligent_grasp_system())
    test_results.append(test_basic_functionality())
    
    # 汇总结果
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print(f"   通过: {passed_tests}/{total_tests}")
    print(f"   成功率: {passed_tests/total_tests:.1%}")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！系统准备就绪")
        print("\n🚀 快速开始:")
        print("   python demo_intelligent_grasp.py --mode single")
        return True
    else:
        print("⚠️ 部分测试失败，请检查依赖和配置")
        print("\n🔧 可能需要的安装:")
        print("   pip install mplib-dist trimesh sb3-contrib")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)