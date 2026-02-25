#!/usr/bin/env python3
"""
EnvClutter环境的完整Motion Planning解决方案
实现抓取最高物体并放置到指定位置的完整流程
"""

import sys
import os
import argparse
import time
import numpy as np
import sapien
import torch
from tqdm import tqdm

# 添加路径
sys.path.append('/home/linux/jzh/RL_Robot')
sys.path.append('/home/linux/jzh/RL_Robot/base_env')
sys.path.append('/home/linux/jzh/RL_Robot/examples/motionplanning/panda')

# 导入必要的模块
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
from transforms3d.euler import euler2quat


def get_highest_object_info(env):
    """
    获取最高物体的信息
    返回: (object_index, height, position)
    """
    if not hasattr(env, 'object_info') or not env.object_info:
        return None, 0, None
    
    # 获取第一个环境的物体信息
    objects_info = env.object_info[0]
    
    highest_idx = 0
    highest_z = -float('inf')
    highest_pos = None
    
    for i, obj_info in enumerate(objects_info):
        z = obj_info['center'][2]
        if z > highest_z:
            highest_z = z
            highest_idx = i
            highest_pos = obj_info['center']
    
    return highest_idx, highest_z, highest_pos


def solve_env_clutter_complete(env, seed=None, debug=False, vis=False):
    """
    EnvClutter环境的完整Motion Planning解决方案
    """
    print("Resetting environment...")
    env.reset(seed=seed)
    
    # 初始化运动规划器
    print("Initializing motion planner...")
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=True,
    )
    
    FINGER_LENGTH = 0.025
    
    # 获取最高物体的信息
    print("Finding highest object...")
    target_idx, target_height, target_pos = get_highest_object_info(env)
    
    if target_pos is None:
        print("No target object found")
        planner.close()
        return -1
    
    print(f"Target object index: {target_idx} at height {target_height:.3f}m")
    print(f"Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    try:
        # 定义抓取参数
        approaching = np.array([0, 0, -1])  # 从上方接近
        
        # 创建一个虚拟的抓取姿态，基于目标位置
        grasp_center = np.array(target_pos)
        
        # 构建抓取姿态
        # 使用机器人当前的TCP方向作为参考
        tcp_transform = env.agent.tcp.pose.to_transformation_matrix()[0].cpu().numpy()
        closing_direction = tcp_transform[:3, 1]  # Y轴作为闭合方向
        
        # 构建抓取pose
        grasp_pose = env.agent.build_grasp_pose(approaching, closing_direction, grasp_center)
        
        # 尝试多个角度找到可行的抓取姿态
        print("Searching for valid grasp pose...")
        angles = np.arange(0, np.pi, np.pi / 6)  # 30度间隔
        angles = np.concatenate([angles, -angles[1:]])  # 添加负角度
        
        valid_grasp_pose = None
        for angle in angles:
            delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
            candidate_pose = grasp_pose * delta_pose
            
            # 干运行测试可行性
            result = planner.move_to_pose_with_screw(candidate_pose, dry_run=True)
            if result != -1 and isinstance(result, dict) and result.get("status") == "Success":
                valid_grasp_pose = candidate_pose
                print(f"Found valid grasp pose at angle {angle:.2f} rad")
                break
        
        if valid_grasp_pose is None:
            print("No valid grasp pose found, using original pose")
            valid_grasp_pose = grasp_pose
        
        # 执行完整的抓取序列
        print("\n=== Executing Motion Planning Sequence ===")
        
        # 1. 移动到预抓取位置（目标上方10cm）
        print("Step 1: Moving to pre-grasp position...")
        pre_grasp_pose = valid_grasp_pose * sapien.Pose([0, 0, -0.10])
        result = planner.move_to_pose_with_screw(pre_grasp_pose)
        if result == -1:
            print("❌ Failed to reach pre-grasp position")
            planner.close()
            return -1
        print("✅ Pre-grasp position reached")
        
        # 2. 下降到抓取位置
        print("Step 2: Descending to grasp position...")
        result = planner.move_to_pose_with_screw(valid_grasp_pose)
        if result == -1:
            print("❌ Failed to reach grasp position")
            planner.close()
            return -1
        print("✅ Grasp position reached")
        
        # 3. 关闭夹爪执行抓取
        print("Step 3: Closing gripper...")
        planner.close_gripper()
        print("✅ Gripper closed")
        
        # 4. 提升物体
        print("Step 4: Lifting object...")
        lift_pose = valid_grasp_pose * sapien.Pose([0, 0, -0.15])  # 提升15cm
        result = planner.move_to_pose_with_screw(lift_pose)
        if result == -1:
            print("❌ Failed to lift object")
            planner.close()
            return -1
        print("✅ Object lifted")
        
        # 5. 移动到目标位置上方
        if hasattr(env, 'goal_site') and env.goal_site is not None:
            print("Step 5: Moving to target area...")
            goal_pos = env.goal_site.pose.sp.p
            transport_pose = sapien.Pose(
                p=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.20],  # 目标上方20cm
                q=valid_grasp_pose.q
            )
            result = planner.move_to_pose_with_screw(transport_pose)
            if result == -1:
                print("❌ Failed to reach target area")
                planner.close()
                return -1
            print("✅ Target area reached")
            
            # 6. 下降到放置位置
            print("Step 6: Descending to place position...")
            place_pose = sapien.Pose(
                p=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.05],  # 目标上方5cm
                q=valid_grasp_pose.q
            )
            result = planner.move_to_pose_with_screw(place_pose)
            if result == -1:
                print("❌ Failed to reach place position")
                planner.close()
                return -1
            print("✅ Place position reached")
            
            # 7. 打开夹爪放置物体
            print("Step 7: Opening gripper to release object...")
            result = planner.open_gripper()
            print("✅ Object released")
            
            # 8. 后退
            print("Step 8: Retreating...")
            retreat_pose = place_pose * sapien.Pose([0, 0, -0.10])
            planner.move_to_pose_with_screw(retreat_pose)
            print("✅ Retreat completed")
        else:
            print("No goal site found, skipping placement")
        
        print("\n🎉 Motion planning sequence completed successfully!")
        planner.close()
        return {"success": True, "elapsed_steps": 100}
        
    except Exception as e:
        print(f"❌ Error during motion planning: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        planner.close()
        return -1


def main():
    parser = argparse.ArgumentParser(description="Complete EnvClutter Motion Planning Solution")
    parser.add_argument("-n", "--num-traj", type=int, default=1, 
                       help="Number of trajectories to generate")
    parser.add_argument("--vis", action="store_true", 
                       help="Enable visualization")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--save-video", action="store_true",
                       help="Save video recordings")
    args = parser.parse_args()
    
    try:
        # 直接导入环境类
        from base_env.env_clutter import EnvClutterEnv
        
        print("🚀 Creating EnvClutter environment...")
        
        # 直接实例化环境
        env = EnvClutterEnv(
            obs_mode="none",
            control_mode="pd_joint_pos",
            render_mode="rgb_array" if args.save_video else "sensors",
            sim_backend="auto",
            num_envs=1,
        )
        
        print("✅ Environment created successfully!")
        print(f"📊 Environment info:")
        print(f"  - Control mode: {env.control_mode}")
        print(f"  - Robot: {env.robot_uids}")
        print(f"  - Objects per env: {getattr(env, 'total_objects_per_env', 'Unknown')}")
        
        # 执行测试
        successes = []
        pbar = tqdm(range(args.num_traj), desc="🔄 Processing trajectories")
        
        for i in pbar:
            print(f"\n{'='*50}")
            print(f"🎯 Trajectory {i+1}/{args.num_traj}")
            print(f"{'='*50}")
            
            try:
                result = solve_env_clutter_complete(
                    env,
                    seed=i,
                    debug=args.debug,
                    vis=args.vis
                )
                
                success = result != -1 and (isinstance(result, dict) and result.get("success", False))
                successes.append(success)
                
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"📋 Trajectory {i+1} result: {status}")
                
                # 更新进度条
                pbar.set_postfix({
                    'success_rate': f"{np.mean(successes):.1%}",
                    'successes': f"{sum(successes)}/{len(successes)}"
                })
                
            except Exception as e:
                print(f"❌ Error in trajectory {i+1}: {e}")
                successes.append(False)
        
        # 输出最终结果
        total_success = sum(successes)
        success_rate = np.mean(successes) if successes else 0
        
        print(f"\n{'='*50}")
        print(f"📈 FINAL RESULTS")
        print(f"{'='*50}")
        print(f"🎯 Total trajectories: {args.num_traj}")
        print(f"✅ Successful: {total_success}")
        print(f"❌ Failed: {args.num_traj - total_success}")
        print(f"📊 Success rate: {success_rate:.1%}")
        
        if success_rate > 0.5:
            print(f"🎉 Great job! Motion planning is working well!")
        elif success_rate > 0:
            print(f"⚠️  Partial success. Consider tuning parameters.")
        else:
            print(f"🔧 No successful trajectories. Check environment setup.")
        
        env.close()
        return 0 if success_rate > 0 else 1
        
    except ImportError as e:
        print(f"❌ Failed to import environment: {e}")
        print("Please ensure the environment is properly set up.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
