#!/usr/bin/env python3
"""
EnvClutter环境的Motion Planning解决方案
基于官方solutions模式，实现抓取最高物体并放置到指定位置
"""

import argparse
import time
import numpy as np
import sapien
import gymnasium as gym
from tqdm import tqdm
import os.path as osp

from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb


def solve_env_clutter(env, seed=None, debug=False, vis=False):
    """
    EnvClutter环境的Motion Planning解决方案
    策略：抓取最高层（layer=2）的物体并放置到目标位置
    """
    env.reset(seed=seed)
    
    # 确保控制模式正确
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], f"Unsupported control mode: {env.unwrapped.control_mode}"
    
    # 初始化运动规划器
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    
    # 找到可选择的物体中最高的那个
    # EnvClutter环境中物体存储在selectable_objects中
    if not env.selectable_objects or not env.selectable_objects[0]:
        print("No selectable objects found")
        planner.close()
        return -1
    
    # 获取第一个环境的物体列表
    objects_in_env = env.selectable_objects[0]
    
    # 找到最高的物体（Z坐标最大）
    highest_z = -float('inf')
    target_obj = None
    target_obj_idx = 0
    
    for i, obj in enumerate(objects_in_env):
        obj_pos = obj.pose.p
        if obj_pos[2] > highest_z:
            highest_z = obj_pos[2]
            target_obj = obj
            target_obj_idx = i
    
    if target_obj is None:
        print("No target object found")
        planner.close()
        return -1
    
    print(f"Target object: {target_obj.name} at height {highest_z:.3f}m")
    
    # 获取物体的OBB (Oriented Bounding Box)
    obb = get_actor_obb(target_obj)
    
    # 定义抓取参数
    approaching = np.array([0, 0, -1])  # 从上方接近
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    # 计算抓取信息
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    
    # 尝试多个角度找到可行的抓取姿态（参考stack_cube.py的策略）
    from transforms3d.euler import euler2quat
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 4)  # 更密集的角度采样
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1  # 正负交替
    
    valid_grasp_pose = None
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        candidate_pose = grasp_pose * delta_pose
        
        # 干运行测试可行性
        result = planner.move_to_pose_with_screw(candidate_pose, dry_run=True)
        if result != -1 and result["status"] == "Success":
            valid_grasp_pose = candidate_pose
            print(f"Found valid grasp pose at angle {angle:.2f} rad")
            break
    
    if valid_grasp_pose is None:
        print("Failed to find valid grasp pose")
        planner.close()
        return -1
    
    # -------------------------------------------------------------------------- #
    # 执行抓取序列
    # -------------------------------------------------------------------------- #
    
    # 1. 移动到预抓取位置
    pre_grasp_pose = valid_grasp_pose * sapien.Pose([0, 0, -0.08])  # 8cm高度
    print("Moving to pre-grasp position...")
    result = planner.move_to_pose_with_screw(pre_grasp_pose)
    if result == -1:
        print("Failed to reach pre-grasp position")
        planner.close()
        return -1
    
    # 2. 下降到抓取位置
    print("Moving to grasp position...")
    result = planner.move_to_pose_with_screw(valid_grasp_pose)
    if result == -1:
        print("Failed to reach grasp position")
        planner.close()
        return -1
    
    # 3. 关闭夹爪
    print("Closing gripper...")
    planner.close_gripper()
    
    # 4. 提升物体
    lift_pose = valid_grasp_pose * sapien.Pose([0, 0, -0.12])  # 提升12cm
    print("Lifting object...")
    result = planner.move_to_pose_with_screw(lift_pose)
    if result == -1:
        print("Failed to lift object")
        planner.close()
        return -1
    
    # 5. 移动到目标位置上方
    goal_position = env.goal_site.pose.sp.p  # 目标位置
    transport_pose = sapien.Pose(
        p=[goal_position[0], goal_position[1], goal_position[2] + 0.15],  # 目标上方15cm
        q=valid_grasp_pose.q
    )
    print("Moving to target area...")
    result = planner.move_to_pose_with_screw(transport_pose)
    if result == -1:
        print("Failed to reach target area")
        planner.close()
        return -1
    
    # 6. 下降到目标位置
    place_pose = sapien.Pose(
        p=[goal_position[0], goal_position[1], goal_position[2] + 0.03],  # 目标位置上方3cm
        q=valid_grasp_pose.q
    )
    print("Placing object...")
    result = planner.move_to_pose_with_screw(place_pose)
    if result == -1:
        print("Failed to place object")
        planner.close()
        return -1
    
    # 7. 打开夹爪放置物体
    print("Opening gripper...")
    result = planner.open_gripper()
    
    # 8. 后退
    retreat_pose = place_pose * sapien.Pose([0, 0, -0.1])
    print("Retreating...")
    planner.move_to_pose_with_screw(retreat_pose)
    
    print("Motion planning sequence completed successfully!")
    planner.close()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="EnvClutter Motion Planning Solver")
    parser.add_argument("-n", "--num-traj", type=int, default=1, 
                       help="Number of trajectories to generate")
    parser.add_argument("--vis", action="store_true", 
                       help="Enable visualization")
    parser.add_argument("--save-video", action="store_true", 
                       help="Save video recordings")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto",
                       help="Simulation backend: auto, cpu, gpu")
    parser.add_argument("--record-dir", type=str, default="demos",
                       help="Directory to save recordings")
    parser.add_argument("--traj-name", type=str,
                       help="Custom trajectory name")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 导入环境定义
    import sys
    sys.path.append('/home/linux/jzh/RL_Robot/base_env')
    
    # 创建环境
    print("Creating EnvClutter-v1 environment...")
    env = gym.make(
        "EnvClutter-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        num_envs=1,
    )
    
    # 设置录制
    if args.save_video or args.record_dir:
        traj_name = args.traj_name or time.strftime("clutter_%Y%m%d_%H%M%S")
        env = RecordEpisode(
            env,
            output_dir=osp.join(args.record_dir, "EnvClutter-v1", "motionplanning"),
            trajectory_name=traj_name,
            save_video=args.save_video,
            source_type="motionplanning",
            source_desc="EnvClutter motion planning solution",
            video_fps=30,
            record_reward=False,
            save_on_reset=False
        )
    
    print(f"Running motion planning on EnvClutter-v1...")
    
    # 执行轨迹生成
    successes = []
    pbar = tqdm(range(args.num_traj), desc="Generating trajectories")
    
    for i in pbar:
        try:
            result = solve_env_clutter(
                env, 
                seed=i, 
                debug=args.debug, 
                vis=args.vis
            )
            
            success = result != -1
            successes.append(success)
            
            if hasattr(env, 'flush_trajectory'):
                env.flush_trajectory()
            if hasattr(env, 'flush_video') and args.save_video:
                env.flush_video()
                
        except Exception as e:
            print(f"Error in trajectory {i}: {e}")
            successes.append(False)
            
        # 更新进度条
        pbar.set_postfix({
            'success_rate': f"{np.mean(successes):.2f}",
            'successes': f"{sum(successes)}/{len(successes)}"
        })
    
    # 输出最终统计
    total_success = sum(successes)
    success_rate = np.mean(successes)
    
    print(f"\n=== Motion Planning Results ===")
    print(f"Total trajectories: {args.num_traj}")
    print(f"Successful: {total_success}")
    print(f"Success rate: {success_rate:.2%}")
    
    env.close()
    
    if success_rate > 0:
        print(f"✅ Motion planning successful! Generated {total_success} valid trajectories.")
    else:
        print("❌ No successful trajectories generated. Check environment setup.")


if __name__ == "__main__":
    main()
