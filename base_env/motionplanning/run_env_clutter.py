#!/usr/bin/env python3
"""
EnvClutter Motion Planning 演示脚本

运行方式:
python run_env_clutter.py --vis --max-objects 3 --episodes 5

功能特性:
1. 自动场景分析和物体识别
2. 智能抓取序列规划（顶层优先）
3. 防碰撞路径规划
4. 实时可视化和性能统计
"""

import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# 设置ManiSkill相关环境变量
os.environ["MANI_SKILL_ASSET_DIR"] = str(project_root / "data")

try:
    # 导入环境和求解器
    from env_clutter import EnvClutterOptimizedEnv  # 使用优化版本
    from motionplanning.env_clutter_solver import solve_env_clutter, EnvClutterMotionPlanner
    print("✓ 成功导入EnvClutter环境和Motion Planning求解器")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保路径设置正确，相关模块已安装")
    sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="EnvClutter Motion Planning演示")
    
    # 环境参数
    parser.add_argument("--env-name", type=str, default="EnvClutterOptimized-v1",
                       help="环境名称")
    parser.add_argument("--robot", type=str, default="panda", 
                       choices=["panda", "fetch"],
                       help="机器人类型")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos",
                       help="控制模式")
    
    # 任务参数
    parser.add_argument("--episodes", type=int, default=3,
                       help="运行回合数")
    parser.add_argument("--max-objects", type=int, default=3,
                       help="每回合最多抓取物体数")
    parser.add_argument("--seed", type=int, default=None,
                       help="随机种子")
    
    # 可视化参数
    parser.add_argument("--vis", action="store_true",
                       help="开启实时可视化")
    parser.add_argument("--debug", action="store_true",
                       help="开启调试模式（需手动确认每步）")
    parser.add_argument("--render-mode", type=str, default="human",
                       help="渲染模式")
    
    # 性能参数
    parser.add_argument("--sim-backend", type=str, default="auto",
                       choices=["auto", "cpu", "gpu"],
                       help="仿真后端")
    parser.add_argument("--joint-speed", type=float, default=0.8,
                       help="关节运动速度限制")
    
    # 输出参数
    parser.add_argument("--save-stats", action="store_true",
                       help="保存统计数据")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="输出目录")
    
    return parser.parse_args()

def create_env(args):
    """创建环境实例"""
    print(f"🤖 创建环境: {args.env_name}")
    print(f"   机器人: {args.robot}")
    print(f"   控制模式: {args.control_mode}")
    print(f"   仿真后端: {args.sim_backend}")
    
    try:
        env = gym.make(
            args.env_name,
            robot_uids=args.robot,
            control_mode=args.control_mode,
            render_mode=args.render_mode if args.vis else None,
            sim_backend=args.sim_backend,
        )
        print("✓ 环境创建成功")
        return env
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return None

def print_scene_analysis(scene_info):
    """打印场景分析结果"""
    print("\n📊 场景分析结果:")
    print(f"   发现物体数量: {len(scene_info.get('objects', []))}")
    
    layers = scene_info.get('layers', {})
    if layers:
        print(f"   物体层次分布:")
        for layer_idx, obj_ids in layers.items():
            print(f"     第{layer_idx}层: {len(obj_ids)}个物体 {obj_ids}")
    
    candidates = scene_info.get('grasp_candidates', [])
    if candidates:
        print(f"   抓取候选点: {len(candidates)}个")
        top_candidates = candidates[:3]  # 显示前3个最优候选
        for i, candidate in enumerate(top_candidates):
            print(f"     #{i+1}: 物体{candidate['object_id']}, "
                  f"质量{candidate['quality']:.2f}, "
                  f"{'优选' if candidate['is_preferred'] else '备选'}方向")
    
    sequence = scene_info.get('optimal_sequence', [])
    if sequence:
        print(f"   推荐抓取序列: {sequence}")

def run_single_episode(env, args, episode_idx):
    """运行单个回合"""
    print(f"\n🎯 开始第 {episode_idx + 1} 回合")
    
    # 设置随机种子
    seed = args.seed + episode_idx if args.seed is not None else None
    
    start_time = time.time()
    
    try:
        # 执行Motion Planning求解
        result = solve_env_clutter(
            env, 
            seed=seed,
            debug=args.debug,
            vis=args.vis,
            max_objects=args.max_objects
        )
        
        episode_time = time.time() - start_time
        
        # 打印结果
        success = result.get("success", False)
        success_rate = result.get("success_rate", 0)
        total_steps = result.get("total_steps", 0)
        grasped_objects = result.get("grasped_objects", 0)
        
        print(f"📈 回合结果:")
        print(f"   任务状态: {'✅ 成功' if success else '❌ 失败'}")
        print(f"   成功率: {success_rate:.1%}")
        print(f"   抓取物体: {grasped_objects}/{args.max_objects}")
        print(f"   执行步数: {total_steps}")
        print(f"   用时: {episode_time:.1f}秒")
        
        return {
            "episode": episode_idx + 1,
            "success": success,
            "success_rate": success_rate,
            "grasped_objects": grasped_objects,
            "total_steps": total_steps,
            "episode_time": episode_time,
            "details": result.get("details", {})
        }
        
    except Exception as e:
        print(f"❌ 回合执行失败: {e}")
        return {
            "episode": episode_idx + 1,
            "success": False,
            "error": str(e)
        }

def print_final_statistics(results):
    """打印最终统计结果"""
    if not results:
        print("没有有效结果可统计")
        return
    
    successful_episodes = [r for r in results if r.get("success", False)]
    total_episodes = len(results)
    
    print(f"\n📊 最终统计结果 ({total_episodes}回合):")
    print("=" * 50)
    
    # 成功率统计
    overall_success_rate = len(successful_episodes) / total_episodes
    print(f"整体成功率: {overall_success_rate:.1%}")
    
    if successful_episodes:
        # 性能统计
        avg_steps = np.mean([r["total_steps"] for r in successful_episodes])
        avg_time = np.mean([r["episode_time"] for r in successful_episodes])
        avg_objects = np.mean([r["grasped_objects"] for r in successful_episodes])
        
        print(f"成功回合平均指标:")
        print(f"   平均抓取物体数: {avg_objects:.1f}")
        print(f"   平均执行步数: {avg_steps:.0f}")
        print(f"   平均用时: {avg_time:.1f}秒")
        
        # 效率指标
        steps_per_object = avg_steps / max(avg_objects, 1)
        print(f"   步数效率: {steps_per_object:.0f}步/物体")

def save_statistics(results, args):
    """保存统计数据"""
    if not args.save_stats:
        return
        
    try:
        import json
        from datetime import datetime
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"env_clutter_mp_stats_{timestamp}.json"
        filepath = os.path.join(args.output_dir, filename)
        
        stats = {
            "timestamp": timestamp,
            "config": vars(args),
            "results": results,
            "summary": {
                "total_episodes": len(results),
                "successful_episodes": len([r for r in results if r.get("success", False)]),
                "overall_success_rate": len([r for r in results if r.get("success", False)]) / len(results) if results else 0
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"📁 统计数据已保存到: {filepath}")
        
    except Exception as e:
        print(f"⚠️  保存统计数据失败: {e}")

def main():
    """主函数"""
    args = parse_args()
    
    print("🚀 EnvClutter Motion Planning 演示启动")
    print("=" * 60)
    print(f"配置参数:")
    print(f"   环境: {args.env_name}")
    print(f"   回合数: {args.episodes}")
    print(f"   最大抓取物体: {args.max_objects}")
    print(f"   可视化: {'开启' if args.vis else '关闭'}")
    print(f"   调试模式: {'开启' if args.debug else '关闭'}")
    
    # 创建环境
    env = create_env(args)
    if env is None:
        return
    
    # 运行多个回合
    all_results = []
    
    try:
        for episode_idx in range(args.episodes):
            episode_result = run_single_episode(env, args, episode_idx)
            all_results.append(episode_result)
            
            # 显示进度
            if episode_idx < args.episodes - 1:
                print(f"\n⏳ 准备下一回合... ({episode_idx + 2}/{args.episodes})")
                time.sleep(1)  # 短暂暂停
        
        # 显示最终统计
        print_final_statistics(all_results)
        
        # 保存统计数据
        save_statistics(all_results, args)
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断执行，已完成 {len(all_results)} 回合")
        if all_results:
            print_final_statistics(all_results)
    
    finally:
        env.close()
        print("\n👋 演示完成，环境已关闭")

if __name__ == "__main__":
    main()
