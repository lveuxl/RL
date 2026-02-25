#!/usr/bin/env python3

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
from mani_skill.utils.structs import Pose
from transforms3d.euler import euler2quat


def get_objects_sorted_by_height(env, planner, debug=False, excluded_indices=None):
    """
    获取物体列表
    基于all_objects的实时pose信息，而不是静态的object_info
    
    Args:
        env: 环境对象
        planner: 规划器对象
        debug: 是否输出调试信息
        excluded_indices: 已被抓取的物体索引列表，将被排除
    """
    height_sorted_objects = []
    excluded_indices = excluded_indices or []
    env_idx = 0  # 假设我们处理第一个环境
    
    if hasattr(env, 'all_objects') and env.all_objects:
        for i, obj in enumerate(env.all_objects):
            # 跳过已被抓取的物体
            if i in excluded_indices:
                continue
                
            # 检查物体是否属于当前环境
            if hasattr(obj, '_scene_idxs') and len(obj._scene_idxs) > 0:
                if obj._scene_idxs[0] != env_idx:
                    continue
                    
            try:
                # 获取物体的实时位置
                obj_pose = obj.pose.p
                if isinstance(obj_pose, torch.Tensor):
                    if len(obj_pose.shape) > 1:
                        position = obj_pose[0].cpu().numpy()  # 取第一个环境的位置
                    else:
                        position = obj_pose.cpu().numpy()
                else:
                    position = np.array(obj_pose)
                
                height = position[2]  # Z坐标即为高度
                
                # 获取对应的初始物体信息（用于类型等静态信息）
                obj_info = None
                if (hasattr(env, 'object_info') and env.object_info and 
                    len(env.object_info) > env_idx and i < len(env.object_info[env_idx])):
                    obj_info = env.object_info[env_idx][i]
                
                height_sorted_objects.append({
                    'index': i,
                    'position': position,
                    'height': height,
                    'obj_info': obj_info,
                    'actor': obj  # 保存物体引用以便后续使用
                })
                
            except Exception as e:
                if debug:
                    pass
                continue
        
        height_sorted_objects.sort(key=lambda x: x['height'], reverse=True)
        
        if debug:
            pass
    
    return height_sorted_objects


def remove_object_from_scene(env, target_obj_index, debug=False):
    """
    从场景中移除指定的物体
    基于官方代码实现，支持GPU和CPU仿真模式
    
    Args:
        env: 环境对象
        target_obj_index: 要移除的物体索引
        debug: 是否输出调试信息
        
    Returns:
        bool: 是否成功移除
    """
    if not hasattr(env, 'all_objects') or target_obj_index >= len(env.all_objects):
        if debug:
            print(f"   ⚠️ 无法找到索引{target_obj_index}的物体")
        return False
    
    try:
        target_obj = env.all_objects[target_obj_index]
        
        # 检查是否为GPU仿真模式 - 基于官方代码逻辑
        if hasattr(env, 'scene') and hasattr(env.scene, 'gpu_sim_enabled') and env.scene.gpu_sim_enabled:
            # GPU仿真：将物体移动到远处（模拟删除效果）
            target_obj.set_pose(Pose.create_from_pq(p=[100.0, 100.0, 100.0]))
            if debug:
                print(f"   ✅ GPU仿真模式 - 已将物体{target_obj_index}移动到远处")
        else:
            # CPU仿真：物理删除物体
            target_obj.remove_from_scene()
            if debug:
                print(f"   ✅ CPU仿真模式 - 已从场景中删除物体{target_obj_index}")
        
        return True
        
    except Exception as e:
        if debug:
            print(f"   ❌ 移除物体{target_obj_index}失败: {e}")
        return False


def is_object_grasped(env, target_obj_info, initial_pos, debug=False):
    """
    检测物体是否被成功抓取
    通过比较物体当前位置与初始位置的差异来判断
    
    Args:
        env: 环境对象
        target_obj_info: 目标物体信息
        initial_pos: 物体初始位置
        debug: 是否输出调试信息
        
    Returns:
        bool: 是否成功抓取
    """
    try:
        obj_index = target_obj_info['index']
        if hasattr(env, 'all_objects') and obj_index < len(env.all_objects):
            target_obj = env.all_objects[obj_index]
            
            # 获取当前位置
            current_pose = target_obj.pose.p
            if isinstance(current_pose, torch.Tensor):
                if len(current_pose.shape) > 1:
                    current_pos = current_pose[0].cpu().numpy()
                else:
                    current_pos = current_pose.cpu().numpy()
            else:
                current_pos = np.array(current_pose)
            
            # 计算位置变化
            pos_change = np.linalg.norm(current_pos - initial_pos)
            
            # 如果物体位置变化超过阈值(5cm)，认为抓取成功
            success = pos_change > 0.05
            
            if debug:
                print(f"   📍 物体位置变化: {pos_change:.3f}m")
                print(f"   🎯 抓取判断: {'成功' if success else '失败'}")
            
            return success
            
    except Exception as e:
        if debug:
            print(f"   ❌ 检测抓取状态失败: {e}")
        return False
    
    return False


def execute_grasp_sequence(env, planner, target_obj_info, debug=False):
    """
    执行完整的抓取序列：接近 -> 抓取 -> 提升 -> 运输 -> 放置 -> 释放
    基于官方motion planning方法实现，使用精确的抓取点计算
    """
    target_pos = target_obj_info['position'].copy()  # 保存初始位置
    obj_index = target_obj_info['index']
    
    if debug:
        pass
    
    # 获取目标物体的实际Actor对象
    target_actor = None
    
    # 优先使用传入的actor引用
    if 'actor' in target_obj_info and target_obj_info['actor'] is not None:
        target_actor = target_obj_info['actor']
        if debug:
            pass
    elif hasattr(env, 'all_objects') and obj_index < len(env.all_objects):
        target_actor = env.all_objects[obj_index]
        if debug:
            pass
    
    # 计算精确的抓取姿态
    FINGER_LENGTH = 0.025  # 夹爪长度
    
    if target_actor is not None:
        # 使用官方方法计算精确抓取信息
        try:
            obb = get_actor_obb(target_actor)
            approaching = np.array([0, 0, -1])  # 向下接近
            target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
            
            grasp_info = compute_grasp_info_by_obb(
                obb,
                approaching=approaching,
                target_closing=target_closing,
                depth=FINGER_LENGTH,
            )
            closing, center = grasp_info["closing"], grasp_info["center"]
            grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
            
            # 寻找有效的抓取姿态，参考官方代码
            angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
            angles = np.repeat(angles, 2)
            angles[1::2] *= -1
            
            valid_grasp_found = False
            for angle in angles:
                delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
                grasp_pose_test = grasp_pose * delta_pose
                res = planner.move_to_pose_with_screw(grasp_pose_test, dry_run=True)
                if res != -1:  # 找到可达的姿态
                    grasp_pose = grasp_pose_test
                    valid_grasp_found = True
                    if debug:
                        pass
                    break
            
            if not valid_grasp_found:
                if debug:
                    pass
            
            if debug:
                pass
        except Exception as e:
            if debug:
                pass
            # 备用方法：使用简单的向下抓取
            grasp_pose = sapien.Pose(
                p=target_pos,
                q=euler2quat(np.pi, 0, 0)  # 向下朝向
            )
    else:
        # 没有找到物体Actor，使用基本方法
        if debug:
            pass
        grasp_pose = sapien.Pose(
            p=target_pos,
            q=euler2quat(np.pi, 0, 0)  # 向下朝向
        )
    
    try:
        # === 阶段1: 移动到物体上方 ===
        approach_height = 0.1  # 减小接近高度，更贴近物体
        
        # 使用计算出的精确抓取姿态，但在更高的位置
        pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -approach_height])  # 在抓取点上方
        
        # 使用RRTConnect规划长距离移动
        result = planner.move_to_pose_with_RRTConnect(pre_grasp_pose)
        if result == -1:
            result = planner.move_to_pose_with_screw(pre_grasp_pose)
            if result == -1:
                return False
        
        # === 阶段2: 下降到抓取位置 ===
        # 直接使用计算出的精确抓取姿态
        result = planner.move_to_pose_with_screw(grasp_pose)
        if result == -1:
            return False
        
        # === 阶段3: 夹爪抓取 ===
        # 执行夹爪闭合指令
        planner.close_gripper()
        
        # === 阶段3.5: 等待夹爪完全闭合 ===
        # 分阶段等待，确保夹爪完全闭合
        time.sleep(0.8)  # 主要闭合等待时间
        
        # 检查夹爪闭合状态的额外等待时间
        max_wait_iterations = 5
        for i in range(max_wait_iterations):
            time.sleep(0.2)  # 每次等待0.2秒
            # 这里可以添加夹爪状态检查，目前使用时间等待
            
        # 最终稳定等待，确保物体被牢固抓住
        time.sleep(0.3)
        
        # === 阶段4: 提升物体 ===
        # 在抓取姿态基础上向上提升0.1米，参考官方代码
        lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
        
        result = planner.move_to_pose_with_screw(lift_pose)
        if result == -1:
            return False
        
        # 等待物体稳定后检测是否成功抓取
        time.sleep(0.5)
        grasp_success = is_object_grasped(env, target_obj_info, target_pos, debug)
        
        # === 阶段5: 运输到目标位置 ===
        if hasattr(env, 'goal_site') and env.goal_site is not None:
            goal_pos = env.goal_site.pose.sp.p
            
            # 先移动到目标位置上方，保持提升时的姿态
            transport_pose = sapien.Pose(
                p=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.2],  # 目标上方
                q=lift_pose.q  # 保持提升姿态
            )
            
            result = planner.move_to_pose_with_RRTConnect(transport_pose)
            if result == -1:
                result = planner.move_to_pose_with_screw(transport_pose)
                if result == -1:
                    return False
            
            # === 阶段6: 下降到放置位置 ===
            place_pose = sapien.Pose(
                p=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.05],
                q=lift_pose.q  # 保持提升姿态直到放置
            )
            
            result = planner.move_to_pose_with_screw(place_pose)
            if result == -1:
                return False
        
        # === 阶段7: 释放物体 ===
        # 执行夹爪打开指令
        planner.open_gripper()
        
        # === 阶段7.5: 等待夹爪完全打开 ===
        # 等待夹爪完全打开，确保物体完全释放
        time.sleep(0.6)  # 主要打开等待时间
        
        # 检查夹爪打开状态的额外等待时间
        max_wait_iterations = 3
        for i in range(max_wait_iterations):
            time.sleep(0.15)  # 每次等待0.15秒
            # 这里可以添加夹爪状态检查，目前使用时间等待
            
        # 等待物体完全稳定和分离
        time.sleep(0.4)
        
        # === 阶段8: 后退到安全位置 ===
        try:
            current_pose = env.agent.tcp.pose
            current_p = current_pose.p
            current_q = current_pose.q
            
            # 处理tensor格式
            if isinstance(current_p, torch.Tensor):
                if len(current_p.shape) > 1:
                    current_p = current_p[0]
                if len(current_q.shape) > 1:
                    current_q = current_q[0]
            
            retreat_pose = sapien.Pose(
                p=[current_p[0].item(), current_p[1].item(), current_p[2].item() + 0.10],
                q=current_q.cpu().numpy() if isinstance(current_q, torch.Tensor) else current_q
            )
            planner.move_to_pose_with_screw(retreat_pose)
        except Exception as e:
            pass
        
        # === 阶段9: 条件移除物体 ===
        if grasp_success:
            print("🔄 Phase 9: Removing successfully grasped object from scene...")
            removal_success = remove_object_from_scene(env, obj_index, debug)
            if removal_success:
                print(f"   ✅ 物体{obj_index}已从场景中移除")
                print(f"\n🎉 物体{target_obj_info['index']}抓取并移除成功完成！")
            else:
                print(f"   ⚠️ 物体{obj_index}移除失败，但抓取成功")
                print(f"\n🎉 物体{target_obj_info['index']}抓取成功完成！")
        else:
            print("⏭️ 物体未被成功抓取，保留在场景中")
            print(f"\n❌ 物体{target_obj_info['index']}抓取失败")
            
        print()  # 添加空行分隔
        
        return grasp_success
        
    except Exception as e:
        pass
        return False


def solve_env_clutter_height_sorted(env, seed=None, debug=False, vis=False):
    """
    EnvClutter环境排序抓取解决方案
    从最高的物体开始，依次抓取所有可达物体
    """
    env.reset(seed=seed)
    
    # 初始化运动规划器
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=debug,
        joint_vel_limits=0.8,
        joint_acc_limits=0.8,
    )
    
    success_count = 0
    total_objects = 0
    
    try:
        # 初始分析场景物体
        initial_objects = get_objects_sorted_by_height(env, planner, debug)
        
        if not initial_objects:
            planner.close()
            return {"success": False, "grasped_count": 0, "total_objects": 0}
        
        total_objects = len(initial_objects)
        
        # 追踪已抓取的物体索引
        grasped_objects = set()
        attempt_count = 0  # 抓取尝试计数
        
        # 动态抓取循环 - 每次都重新排序
        while success_count < total_objects:
            attempt_count += 1
            
            current_objects = get_objects_sorted_by_height(
                env, planner, debug, excluded_indices=grasped_objects
            )
            
            if not current_objects:
                break
            
            # 选择当前最高的物体
            target_obj = current_objects[0]  # 第一个就是最高的
            
            
            # 执行抓取
            success = execute_grasp_sequence(env, planner, target_obj, debug)
            
            if success:
                success_count += 1
                grasped_objects.add(target_obj['index'])
            else:
                # 抓取失败，将该物体标记为已处理，避免无限循环
                grasped_objects.add(target_obj['index'])
            
            # 短暂延迟，让环境稳定
            time.sleep(0.5)
            
            # 安全检查，避免无限循环
            if attempt_count > total_objects * 2:  # 最多尝试两倍的物体数量
                break
        
        planner.close()
        
        # 计算成功率
        success_rate = success_count / total_objects if total_objects > 0 else 0
        
        return {
            "success": success_count > 0,
            "grasped_count": success_count,
            "total_objects": total_objects,
            "success_rate": success_rate
        }
        
    except Exception as e:
        planner.close()
        return {"success": False, "grasped_count": success_count, "total_objects": total_objects}


def main():
    parser = argparse.ArgumentParser(description="EnvClutter环境排序抓取解决方案")
    parser.add_argument("-n", "--num-traj", type=int, default=1, 
                       help="生成轨迹数量")
    parser.add_argument("--vis", action="store_true", 
                       help="启用可视化")
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    parser.add_argument("--save-video", action="store_true",
                       help="保存视频录制")
    args = parser.parse_args()
    
    try:
        # 导入环境类
        from base_env.env_clutter import EnvClutterEnv
        
        # 实例化环境
        env = EnvClutterEnv(
            obs_mode="none",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sim_backend="auto",
            num_envs=1,
        )
        
        all_results = []
        pbar = tqdm(range(args.num_traj), desc="🔄 排序抓取轨迹处理")
        
        for i in pbar:
            try:
                result = solve_env_clutter_height_sorted(
                    env,
                    seed=i,
                    debug=args.debug,
                    vis=args.vis
                )
                
                all_results.append(result)
                
                success = result.get("success", False)
                grasped = result.get("grasped_count", 0)
                total = result.get("total_objects", 0)
                
                status = f"🎉 成功 ({grasped}/{total})" if success else f"❌ 失败 ({grasped}/{total})"
                
                # 更新进度条
                avg_success_rate = np.mean([r.get("success_rate", 0) for r in all_results])
                avg_grasped = np.mean([r.get("grasped_count", 0) for r in all_results])
                pbar.set_postfix({
                    'avg_success_rate': f"{avg_success_rate:.1%}",
                    'avg_grasped': f"{avg_grasped:.1f}"
                })
                
            except Exception as e:
                all_results.append({"success": False, "grasped_count": 0, "total_objects": 0, "success_rate": 0})
        
        # 输出最终结果
        total_trials = len(all_results)
        successful_trials = sum(1 for r in all_results if r.get("success", False))
        avg_grasped = np.mean([r.get("grasped_count", 0) for r in all_results])
        avg_total = np.mean([r.get("total_objects", 0) for r in all_results])
        avg_success_rate = np.mean([r.get("success_rate", 0) for r in all_results])
        
        print(f"\n{'='*60}")
        print(f"{'='*60}")
        print(f"🎯 总轨迹数: {total_trials}")
        print(f"🎉 成功轨迹数: {successful_trials}")
        print(f"📊 轨迹成功率: {successful_trials/total_trials:.1%}")
        print(f"🎯 平均物体数: {avg_total:.1f}")
        print(f"✅ 平均抓取数: {avg_grasped:.1f}")
        print(f"📈 平均抓取成功率: {avg_success_rate:.1%}")
        
        
        
        env.close()
        return 0 if avg_success_rate > 0.3 else 1
        
    except ImportError as e:
        print(f"❌ 环境导入失败: {e}")
        return 1
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
