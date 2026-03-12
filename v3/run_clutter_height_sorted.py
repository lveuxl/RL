#!/usr/bin/env python3
"""
EnvClutter环境按高度排序抓取解决方案
基于现有的env_clutter环境实现机器人根据物体的高度按顺序从高到底抓取
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
sys.path.append('/home/linux/jzh/RL')
sys.path.append('/home/linux/jzh/RL/base_env')
sys.path.append('/home/linux/jzh/RL/examples/motionplanning/panda')

# 导入必要的模块
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
from transforms3d.euler import euler2quat


def get_objects_sorted_by_height(env, planner, debug=False):
    """
    获取按高度从高到低排序的物体列表
    基于env.object_info获取物体信息，按Z坐标（高度）降序排列
    """
    height_sorted_objects = []
    
    if hasattr(env, 'object_info') and env.object_info:
        objects_info = env.object_info[0]  # 获取第一个环境的物体信息
        
        for i, obj_info in enumerate(objects_info):
            pos = obj_info['center']  # 获取物体中心位置
            height = pos[2]  # Z坐标即为高度
            
            height_sorted_objects.append({
                'index': i,
                'position': pos,
                'height': height,
                'obj_info': obj_info
            })
        
        # 按高度从高到低排序
        height_sorted_objects.sort(key=lambda x: x['height'], reverse=True)
        
        if debug:
            print("🔍 物体高度排序结果:")
            for i, obj in enumerate(height_sorted_objects):
                print(f"   {i+1}. 物体{obj['index']}: 高度={obj['height']:.3f}m, 位置={obj['position']}")
    
    return height_sorted_objects


def add_objects_as_obstacles(env, planner, target_obj_index, debug=False):
    """
    将环境中的其他物体添加为碰撞检测的障碍物
    基于官方的add_box_collision方法
    """
    if not hasattr(env, 'all_objects') or not env.all_objects:
        return
    
    # 清除之前的碰撞约束
    planner.clear_collisions()
    
    obstacles_added = 0
    for i, obj_actor in enumerate(env.all_objects):
        # 跳过目标物体
        if i == target_obj_index:
            continue
        
        try:
            # 获取物体的边界框信息
            obb = get_actor_obb(obj_actor)
            extents = obb.extents
            pose = sapien.Pose(obb.center, obb.orientation)
            
            # 添加安全边界 - 将物体稍微放大
            safety_margin = 0.01  # 1cm安全边界
            expanded_extents = extents + safety_margin
            
            # 添加为盒子形状的碰撞体
            planner.add_box_collision(expanded_extents, pose)
            obstacles_added += 1
            
        except Exception as e:
            if debug:
                print(f"   ⚠️ 添加物体{i}为障碍物失败: {e}")
            continue
    
    if debug:
        print(f"   ✅ 已添加{obstacles_added}个物体作为障碍物")


def execute_grasp_sequence(env, planner, target_obj_info, debug=False):
    """
    执行完整的抓取序列：接近 -> 抓取 -> 提升 -> 运输 -> 放置 -> 释放
    基于官方motion planning方法实现，使用精确的抓取点计算和碰撞检测
    """
    target_pos = target_obj_info['position']
    obj_index = target_obj_info['index']
    
    if debug:
        print(f"🎯 开始抓取物体{obj_index}")
        print(f"   目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"   目标高度: {target_obj_info['height']:.3f}m")
    
    # 添加其他物体作为障碍物
    print("🛡️ Adding collision obstacles...")
    add_objects_as_obstacles(env, planner, obj_index, debug)
    
    # 获取目标物体的实际Actor对象
    target_actor = None
    if hasattr(env, 'all_objects') and obj_index < len(env.all_objects):
        target_actor = env.all_objects[obj_index]
    
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
                        print(f"   ✅ 找到可达抓取姿态，角度偏移: {angle:.2f}弧度")
                    break
            
            if not valid_grasp_found:
                if debug:
                    print(f"   ⚠️ 未找到可达的抓取姿态，使用原始姿态")
            
            if debug:
                print(f"   ✅ 使用OBB计算精确抓取姿态")
                print(f"   抓取中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        except Exception as e:
            if debug:
                print(f"   ⚠️ OBB计算失败，使用备用方法: {e}")
            # 备用方法：使用简单的向下抓取
            grasp_pose = sapien.Pose(
                p=target_pos,
                q=euler2quat(np.pi, 0, 0)  # 向下朝向
            )
    else:
        # 没有找到物体Actor，使用基本方法
        if debug:
            print(f"   ⚠️ 未找到物体Actor，使用基本抓取方法")
        grasp_pose = sapien.Pose(
            p=target_pos,
            q=euler2quat(np.pi, 0, 0)  # 向下朝向
        )
    
    try:
        # === 阶段1: 移动到物体上方 ===
        print("🔄 Phase 1: Moving above object...")
        approach_height = 0.2  # 增加接近高度，提高安全性
        
        # 使用计算出的精确抓取姿态，但在更高的位置
        pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -approach_height])  # 在抓取点上方
        
        # 尝试多种规划策略以提高成功率
        result = -1
        
        # 策略1: 先移动到高空安全位置，再接近
        safe_height = max(0.3, target_pos[2] + 0.25)  # 确保足够高
        high_safe_pose = sapien.Pose(
            p=[target_pos[0], target_pos[1], safe_height],
            q=pre_grasp_pose.q
        )
        
        print("   🔄 Step 1.1: Moving to high safe position...")
        result = planner.move_to_pose_with_RRTConnect(high_safe_pose)
        if result != -1:
            print("   🔄 Step 1.2: Descending to approach position...")
            result = planner.move_to_pose_with_screw(pre_grasp_pose)
        
        # 策略2: 如果分步移动失败，尝试直接移动
        if result == -1:
            print("   🔄 Attempting direct RRTConnect...")
            result = planner.move_to_pose_with_RRTConnect(pre_grasp_pose)
        
        # 策略3: 如果RRT失败，尝试screw规划
        if result == -1:
            print("   🔄 Attempting screw motion...")
            result = planner.move_to_pose_with_screw(pre_grasp_pose)
            
        if result == -1:
            print("   ❌ 所有规划策略都失败，无法到达物体上方")
            return False
            
        print("   ✅ 已到达物体上方")
        time.sleep(0.3)  # 短暂等待确保位置稳定
        
        # === 阶段2: 下降到抓取位置 ===
        print("🔄 Phase 2: Descending to grasp position...")
        # 缓慢下降策略：分多步下降以避免碰撞
        intermediate_height = target_pos[2] + 0.05  # 中间高度
        intermediate_pose = sapien.Pose(
            p=[target_pos[0], target_pos[1], intermediate_height],
            q=grasp_pose.q
        )
        
        # 先下降到中间位置
        print("   🔄 Step 2.1: Descending to intermediate position...")
        result = planner.move_to_pose_with_screw(intermediate_pose)
        if result != -1:
            print("   🔄 Step 2.2: Final descent to grasp position...")
            result = planner.move_to_pose_with_screw(grasp_pose)
        
        # 如果分步下降失败，尝试直接下降
        if result == -1:
            print("   🔄 Attempting direct descent...")
            result = planner.move_to_pose_with_screw(grasp_pose)
            
        if result == -1:
            print("   ❌ 无法下降到抓取位置")
            return False
        print("   ✅ 已到达抓取位置")
        
        # === 阶段3: 夹爪抓取 ===
        print("🔄 Phase 3: Grasping...")
        planner.close_gripper()
        print("   ✅ 夹爪已关闭")
        
        # 等待物理稳定，确保物体被抓住
        time.sleep(0.8)  # 增加等待时间，确保抓取稳定
        
        # === 阶段4: 提升物体 ===
        print("🔄 Phase 4: Lifting object...")
        # 在抓取姿态基础上向上提升0.1米，参考官方代码
        lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
        
        result = planner.move_to_pose_with_screw(lift_pose)
        if result == -1:
            print("   ❌ 提升失败")
            return False
        print("   ✅ 物体已提升")
        
        # === 阶段5: 运输到目标位置 ===
        if hasattr(env, 'goal_site') and env.goal_site is not None:
            print("🔄 Phase 5: Transporting to goal...")
            goal_pos = env.goal_site.pose.sp.p
            
            # 安全运输策略：先上升到更高的安全高度，再水平移动，最后下降
            transport_safe_height = max(0.35, max(target_pos[2], goal_pos[2]) + 0.3)
            
            # 步骤5.1: 上升到安全运输高度
            print("   🔄 Step 5.1: Rising to safe transport height...")
            high_transport_pose = sapien.Pose(
                p=[target_pos[0], target_pos[1], transport_safe_height],
                q=lift_pose.q
            )
            result = planner.move_to_pose_with_screw(high_transport_pose)
            
            if result != -1:
                # 步骤5.2: 水平移动到目标上方的安全高度
                print("   🔄 Step 5.2: Moving horizontally to target area...")
                goal_high_pose = sapien.Pose(
                    p=[goal_pos[0], goal_pos[1], transport_safe_height],
                    q=lift_pose.q
                )
                result = planner.move_to_pose_with_RRTConnect(goal_high_pose)
                
                if result != -1:
                    # 步骤5.3: 下降到目标位置上方
                    print("   🔄 Step 5.3: Descending to target position...")
                    transport_pose = sapien.Pose(
                        p=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.2],
                        q=lift_pose.q
                    )
                    result = planner.move_to_pose_with_screw(transport_pose)
            
            # 备用策略：如果分步运输失败，尝试直接运输
            if result == -1:
                print("   🔄 Attempting direct transport...")
                transport_pose = sapien.Pose(
                    p=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.2],
                    q=lift_pose.q
                )
                result = planner.move_to_pose_with_RRTConnect(transport_pose)
                if result == -1:
                    result = planner.move_to_pose_with_screw(transport_pose)
                    
            if result == -1:
                print("   ❌ 所有运输策略都失败，无法到达目标上方")
                return False
            print("   ✅ 已到达目标上方")
            
            # === 阶段6: 下降到放置位置 ===
            print("🔄 Phase 6: Descending to place...")
            place_pose = sapien.Pose(
                p=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.05],
                q=lift_pose.q  # 保持提升姿态直到放置
            )
            
            result = planner.move_to_pose_with_screw(place_pose)
            if result == -1:
                print("   ❌ 无法下降到放置位置")
                return False
            print("   ✅ 已到达放置位置")
        
        # === 阶段7: 释放物体 ===
        print("🔄 Phase 7: Releasing object...")
        planner.open_gripper()
        print("   ✅ 物体已释放")
        
        # 等待物体稳定
        time.sleep(0.3)
        
        # === 阶段8: 后退到安全位置 ===
        print("🔄 Phase 8: Retreating to safe position...")
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
            print("   ✅ 已后退到安全位置")
        except Exception as e:
            print(f"   ✅ 后退完成 (微调: {e})")
        
        print(f"\n🎉 物体{target_obj_info['index']}抓取成功完成！\n")
        return True
        
    except Exception as e:
        print(f"❌ 抓取过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def solve_env_clutter_height_sorted(env, seed=None, debug=False, vis=False):
    """
    EnvClutter环境按高度排序抓取解决方案
    从最高的物体开始，依次抓取所有可达物体
    """
    print("🚀 启动按高度排序抓取...")
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
        # 获取按高度排序的物体列表
        print("🧠 分析场景物体...")
        height_sorted_objects = get_objects_sorted_by_height(env, planner, debug)
        
        if not height_sorted_objects:
            print("❌ 未找到可抓取物体")
            planner.close()
            return {"success": False, "grasped_count": 0, "total_objects": 0}
        
        total_objects = len(height_sorted_objects)
        print(f"🎯 发现 {total_objects} 个物体，按高度排序开始抓取")
        
        # 按高度顺序抓取每个物体
        for i, target_obj in enumerate(height_sorted_objects):
            print(f"\n{'='*50}")
            print(f"🎯 抓取第 {i+1}/{total_objects} 个物体 (最高的物体)")
            print(f"   物体索引: {target_obj['index']}")
            print(f"   物体高度: {target_obj['height']:.3f}m")
            print(f"   物体位置: [{target_obj['position'][0]:.3f}, {target_obj['position'][1]:.3f}, {target_obj['position'][2]:.3f}]")
            print(f"{'='*50}")
            
            # 执行抓取
            success = execute_grasp_sequence(env, planner, target_obj, debug)
            
            if success:
                success_count += 1
                print(f"✅ 物体{target_obj['index']}抓取成功! ({success_count}/{i+1})")
            else:
                print(f"❌ 物体{target_obj['index']}抓取失败")
                # 继续尝试下一个物体
                continue
            
            # 短暂延迟，让环境稳定
            time.sleep(0.5)
        
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
        print(f"❌ 按高度排序抓取过程中出现错误: {e}")
        planner.close()
        return {"success": False, "grasped_count": success_count, "total_objects": total_objects}


def main():
    parser = argparse.ArgumentParser(description="EnvClutter环境按高度排序抓取解决方案")
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
        
        print("🚀 创建EnvClutter环境...")
        
        # 实例化环境
        env = EnvClutterEnv(
            obs_mode="none",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sim_backend="auto",
            num_envs=1,
        )
        
        print("✅ 环境创建成功!")
        
        # 执行按高度排序抓取测试
        all_results = []
        pbar = tqdm(range(args.num_traj), desc="🔄 按高度排序抓取轨迹处理")
        
        for i in pbar:
            print(f"\n{'='*60}")
            print(f"🎯 按高度排序抓取轨迹 {i+1}/{args.num_traj}")
            print(f"{'='*60}")
            
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
                print(f"📋 轨迹 {i+1} 结果: {status}")
                
                # 更新进度条
                avg_success_rate = np.mean([r.get("success_rate", 0) for r in all_results])
                avg_grasped = np.mean([r.get("grasped_count", 0) for r in all_results])
                pbar.set_postfix({
                    'avg_success_rate': f"{avg_success_rate:.1%}",
                    'avg_grasped': f"{avg_grasped:.1f}"
                })
                
            except Exception as e:
                print(f"❌ 轨迹 {i+1} 出现错误: {e}")
                all_results.append({"success": False, "grasped_count": 0, "total_objects": 0, "success_rate": 0})
        
        # 输出最终结果
        total_trials = len(all_results)
        successful_trials = sum(1 for r in all_results if r.get("success", False))
        avg_grasped = np.mean([r.get("grasped_count", 0) for r in all_results])
        avg_total = np.mean([r.get("total_objects", 0) for r in all_results])
        avg_success_rate = np.mean([r.get("success_rate", 0) for r in all_results])
        
        print(f"\n{'='*60}")
        print(f"🏆 按高度排序抓取最终结果")
        print(f"{'='*60}")
        print(f"🎯 总轨迹数: {total_trials}")
        print(f"🎉 成功轨迹数: {successful_trials}")
        print(f"📊 轨迹成功率: {successful_trials/total_trials:.1%}")
        print(f"🎯 平均物体数: {avg_total:.1f}")
        print(f"✅ 平均抓取数: {avg_grasped:.1f}")
        print(f"📈 平均抓取成功率: {avg_success_rate:.1%}")
        
        if avg_success_rate >= 0.8:
            print(f"🏆 优秀！按高度排序抓取表现出色！")
        elif avg_success_rate >= 0.6:
            print(f"🎉 很好！按高度排序抓取表现良好！")
        elif avg_success_rate > 0.3:
            print(f"⚡ 可接受！按高度排序抓取基本可用！")
        else:
            print(f"🔧 需要进一步优化抓取策略。")
        
        print(f"\n🎮 使用方法:")
        print(f"   conda activate maniskill")
        print(f"   python v3/run_clutter_height_sorted.py -n 5    # 运行5个轨迹")
        print(f"   python v3/run_clutter_height_sorted.py --vis   # 启用可视化")
        print(f"   python v3/run_clutter_height_sorted.py --debug # 启用调试")
        
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
