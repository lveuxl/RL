#!/usr/bin/env python3
"""
EnvClutter环境的精确Motion Planning解决方案
使用OBB计算实现精确抓取
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


def find_highest_object_actor(env):
    """
    查找最高的物体actor对象
    返回: (target_actor, height)
    """
    if not hasattr(env, 'all_objects') or not env.all_objects:
        print("❌ No objects found in environment")
        return None, 0.0
    
    print(f"🔍 Searching through {len(env.all_objects)} objects...")
    
    highest_z = -float('inf')
    target_actor = None
    env_idx = 0  # 第一个环境
    
    for i, obj in enumerate(env.all_objects):
        if hasattr(obj, '_scene_idxs') and len(obj._scene_idxs) > 0:
            if obj._scene_idxs[0] == env_idx:  # 属于第一个环境
                try:
                    # 获取物体的Z坐标
                    obj_pose = obj.pose.p
                    if isinstance(obj_pose, torch.Tensor):
                        if len(obj_pose.shape) > 1:
                            obj_z = obj_pose[0, 2].item()
                        else:
                            obj_z = obj_pose[2].item()
                    else:
                        obj_z = obj_pose[2]
                    
                    print(f"  Object {i}: Z={obj_z:.3f}")
                    
                    if obj_z > highest_z:
                        highest_z = obj_z
                        target_actor = obj
                        print(f"  ✅ New highest object found: Z={obj_z:.3f}")
                        
                except Exception as e:
                    print(f"  ❌ Error accessing object {i}: {e}")
                    continue
    
    if target_actor is None:
        print("❌ No valid target object found")
        return None, 0.0
    
    print(f"🎯 Selected target object at height Z={highest_z:.3f}")
    return target_actor, highest_z


def solve_env_clutter_precision(env, seed=None, debug=False, vis=False):
    """
    EnvClutter环境的精确Motion Planning解决方案
    使用OBB计算实现真正的精确抓取
    """
    print("🚀 Starting precision motion planning...")
    obs, _ = env.reset(seed=seed)
    
    try:
        # 初始化运动规划器
        print("📋 Initializing motion planner...")
        planner = PandaArmMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=vis,
            print_env_info=debug,
            joint_vel_limits=0.8,
            joint_acc_limits=0.8,
        )
        
        # 找到最高的物体
        print("\n🔍 Finding target object...")
        target_actor, target_height = find_highest_object_actor(env)
        
        if target_actor is None:
            print("❌ No target object found, aborting")
            planner.close()
            return -1
        
        print(f"✅ Target object selected with height: {target_height:.3f}")
        
        # 使用官方OBB方法计算精确抓取信息
        print("\n🎯 Calculating precise grasp using OBB...")
        FINGER_LENGTH = 0.025  # Panda夹爪长度
        
        try:
            # 获取物体的OBB（Oriented Bounding Box）
            obb = get_actor_obb(target_actor)
            print("✅ OBB calculated successfully")
            
            # 定义抓取参数
            approaching = np.array([0, 0, -1])  # 从上方接近
            
            # 获取TCP姿态的变换矩阵来定义闭合方向
            tcp_transform = env.agent.tcp.pose.to_transformation_matrix()
            if isinstance(tcp_transform, torch.Tensor):
                target_closing = tcp_transform[0, :3, 1].cpu().numpy()
            else:
                target_closing = tcp_transform[:3, 1]
            
            # 计算精确的抓取信息
            grasp_info = compute_grasp_info_by_obb(
                obb,
                approaching=approaching,
                target_closing=target_closing,
                depth=FINGER_LENGTH,
            )
            
            closing = grasp_info["closing"]
            center = grasp_info["center"]
            
            print(f"✅ Grasp info calculated:")
            print(f"   Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
            print(f"   Closing direction: [{closing[0]:.3f}, {closing[1]:.3f}, {closing[2]:.3f}]")
            
            # 获取物体的实际位置
            target_pos = target_actor.pose.sp.p
            
            # 构建精确的抓取姿态
            grasp_pose = env.agent.build_grasp_pose(approaching, closing, target_pos)
            print("✅ Precise grasp pose built")
            
        except Exception as e:
            print(f"❌ Error in OBB calculation: {e}")
            print("🔄 Falling back to simple grasp calculation...")
            
            # 简单的后备方案
            approaching = np.array([0, 0, -1])
            closing = np.array([1, 0, 0])
            target_pos = target_actor.pose.sp.p
            grasp_pose = env.agent.build_grasp_pose(approaching, closing, target_pos)
        
        print("\n=== 🎯 Executing Precision Motion Planning ===")
        
        # 1. 移动到目标上方
        print("Step 1: Moving above target...")
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])  # 预抓取位置
        result = planner.move_to_pose_with_screw(reach_pose)
        if result == -1:
            print("❌ Failed to reach pre-grasp position")
            planner.close()
            return -1
        print("✅ Reached pre-grasp position")
        
        # 2. 精确移动到抓取位置
        print("Step 2: Moving to precise grasp position...")
        result = planner.move_to_pose_with_screw(grasp_pose)
        if result == -1:
            print("❌ Failed to reach grasp position")
            planner.close()
            return -1
        print("✅ Reached grasp position")
        
        # 3. 关闭夹爪抓取
        print("Step 3: Closing gripper...")
        result = planner.close_gripper()
        if result == -1:
            print("❌ Failed to close gripper")
            planner.close()
            return -1
        print("✅ Gripper closed")
        
        # 4. 提升物体
        print("Step 4: Lifting object...")
        lift_pose = grasp_pose * sapien.Pose([0, 0, -0.12])  # 上移12cm
        result = planner.move_to_pose_with_screw(lift_pose)
        if result == -1:
            print("❌ Failed to lift object")
            planner.close()
            return -1
        print("✅ Object lifted")
        
        # 5. 移动到放置位置
        print("Step 5: Moving to placement position...")
        # 检查是否有目标位置
        if hasattr(env, 'goal_site') and env.goal_site is not None:
            try:
                goal_pos = env.goal_site.pose.sp.p
                print(f"   Target goal: [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
                place_pose = sapien.Pose(p=goal_pos, q=grasp_pose.q)
                result = planner.move_to_pose_with_RRTConnect(place_pose)
                if result != -1:
                    print("✅ Moved to goal position")
                else:
                    print("⚠️  RRT failed, using screw motion")
                    result = planner.move_to_pose_with_screw(place_pose)
            except Exception as e:
                print(f"❌ Error moving to goal: {e}")
                # 使用当前位置作为放置位置
                place_pose = lift_pose
        else:
            print("   No goal site found, placing at current lifted position")
            place_pose = lift_pose
        
        # 6. 放下物体
        print("Step 6: Placing object...")
        lower_pose = place_pose * sapien.Pose([0, 0, 0.05])  # 下降5cm
        result = planner.move_to_pose_with_screw(lower_pose)
        if result == -1:
            print("❌ Failed to lower object")
            planner.close()
            return -1
        print("✅ Object lowered to placement position")
        
        # 7. 打开夹爪
        print("Step 7: Opening gripper...")
        result = planner.open_gripper()
        if result == -1:
            print("❌ Failed to open gripper")
            planner.close()
            return -1
        print("✅ Gripper opened")
        
        # 8. 安全后退
        print("Step 8: Safe retreat...")
        retreat_pose = lower_pose * sapien.Pose([0, 0, -0.10])  # 上移10cm
        result = planner.move_to_pose_with_screw(retreat_pose)
        if result == -1:
            print("⚠️  Retreat failed, but task may still be successful")
        else:
            print("✅ Safe retreat completed")
        
        planner.close()
        
        # 获取最终状态
        final_info = env.get_info()
        success = final_info.get("success", False)
        
        print(f"\n🎉 Motion planning completed!")
        print(f"   Success: {success}")
        
        return 0 if success else -1
        
    except Exception as e:
        print(f"❌ Critical error during motion planning: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        planner.close()
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-traj", type=int, default=1, help="Number of trajectories to generate")
    parser.add_argument("--vis", action="store_true", help="Visualize the solution")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--sim-backend", type=str, default="auto", help="Simulation backend")
    args = parser.parse_args()
    
    # 创建环境
    print("🏗️  Creating environment...")
    try:
        # 直接实例化环境
        sys.path.append('/home/linux/jzh/RL_Robot/base_env')
        from env_clutter import EnvClutterEnv
        env = EnvClutterEnv(
            obs_mode="none", 
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sim_backend=args.sim_backend,
            num_envs=1,
        )
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return
    
    print(f"🎯 Running precision motion planning on EnvClutter...")
    
    successful_trajectories = 0
    total_attempts = 0
    
    for i in tqdm(range(args.num_traj), desc="Generating trajectories"):
        total_attempts += 1
        seed = i
        
        print(f"\n📋 === Trajectory {i+1}/{args.num_traj} (seed={seed}) ===")
        
        result = solve_env_clutter_precision(
            env, 
            seed=seed, 
            debug=args.debug, 
            vis=args.vis
        )
        
        if result == 0:
            successful_trajectories += 1
            print(f"✅ Trajectory {i+1} successful!")
        else:
            print(f"❌ Trajectory {i+1} failed")
    
    # 最终统计
    success_rate = successful_trajectories / total_attempts * 100
    print(f"\n🎉 === Final Results ===")
    print(f"   Total attempts: {total_attempts}")
    print(f"   Successful: {successful_trajectories}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    env.close()


if __name__ == "__main__":
    main()
