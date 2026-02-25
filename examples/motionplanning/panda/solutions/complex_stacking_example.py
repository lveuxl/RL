"""
复杂堆叠环境Motion Planning使用示例

演示如何在各种复杂场景中使用ComplexStackingMotionPlanner：
1. 多层金字塔堆叠
2. 不规则物体堆叠  
3. 受限空间内的精确堆叠
4. 动态障碍物环境下的堆叠
"""

import numpy as np
import sapien
import gymnasium as gym
from typing import List

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.solutions.complex_stacking_solver import (
    ComplexStackingMotionPlanner, 
    StackingTarget,
    create_complex_stacking_plan
)

def solve_complex_pyramid_stacking(env, seed=None, debug=False, vis=False):
    """
    解决复杂金字塔堆叠任务
    
    场景：4个物体组成的双层金字塔
    底层：3个物体呈三角形排列
    顶层：1个物体放置在底层中心
    """
    env.reset(seed=seed)
    
    # 初始化基础motion planner
    base_planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False
    )
    
    # 创建复杂堆叠求解器
    complex_planner = ComplexStackingMotionPlanner(base_planner)
    
    # 定义堆叠序列（假设环境中有4个cube对象）
    objects = [env.unwrapped.cubeA, env.unwrapped.cubeB, 
               env.unwrapped.cubeC, env.unwrapped.cubeD]
    
    # 创建四层金字塔堆叠计划
    stacking_sequence = [
        # 第一步：B靠近A（底层第一条边）
        StackingTarget(
            source_obj=objects[1], 
            target_obj=objects[0],
            stack_height=0.0,
            approach_angles=np.linspace(0, np.pi*2, 8)  # 8个候选角度
        ),
        
        # 第二步：C与A,B形成三角形（底层完成）
        StackingTarget(
            source_obj=objects[2],
            target_obj=objects[0],  # 以A为参考点
            stack_height=0.0,
            approach_angles=np.linspace(np.pi/3, np.pi*4/3, 6)
        ),
        
        # 第三步：D堆叠在ABC中心上方（顶层）
        StackingTarget(
            source_obj=objects[3],
            target_obj=objects[0],  # 以A为参考计算中心位置
            stack_height=0.08,  # 显著高度差
            approach_angles=np.linspace(0, np.pi*2, 12)  # 更多尝试角度
        )
    ]
    
    # 执行复杂堆叠
    success = complex_planner.solve_complex_stacking(stacking_sequence)
    
    if success:
        print("✅ 复杂金字塔堆叠成功完成!")
        # 添加最终验证步骤
        _verify_stacking_stability(env, objects)
    else:
        print("❌ 复杂金字塔堆叠失败")
    
    base_planner.close()
    return success

def solve_constrained_space_stacking(env, seed=None, debug=False, vis=False):
    """
    受限空间内的精确堆叠
    
    场景：在狭窄容器内堆叠物体，需要精确的路径规划
    """
    env.reset(seed=seed)
    
    base_planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        print_env_info=False
    )
    
    complex_planner = ComplexStackingMotionPlanner(base_planner)
    
    # 添加容器壁作为碰撞约束
    container_walls = [
        # 左壁
        (np.array([0.02, 0.2, 0.2]), sapien.Pose([-0.15, 0, 0.1])),
        # 右壁  
        (np.array([0.02, 0.2, 0.2]), sapien.Pose([0.15, 0, 0.1])),
        # 后壁
        (np.array([0.3, 0.02, 0.2]), sapien.Pose([0, 0.15, 0.1])),
        # 前壁
        (np.array([0.3, 0.02, 0.2]), sapien.Pose([0, -0.15, 0.1]))
    ]
    
    # 注册容器壁为碰撞体
    for extents, pose in container_walls:
        base_planner.add_box_collision(extents, pose)
    
    # 在受限空间内执行堆叠
    objects = [env.unwrapped.cubeA, env.unwrapped.cubeB]
    
    stacking_sequence = [
        StackingTarget(
            source_obj=objects[1],
            target_obj=objects[0], 
            stack_height=0.02,
            approach_angles=[0, np.pi/2, np.pi, 3*np.pi/2]  # 4个主要方向
        )
    ]
    
    success = complex_planner.solve_complex_stacking(stacking_sequence)
    
    print("🏗️ 受限空间堆叠:", "成功" if success else "失败")
    
    base_planner.close()
    return success

def solve_dynamic_obstacle_stacking(env, seed=None, debug=False, vis=False):
    """
    动态障碍物环境下的堆叠
    
    场景：环境中存在移动障碍物，需要实时更新碰撞约束
    """
    env.reset(seed=seed)
    
    base_planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        print_env_info=False
    )
    
    class DynamicObstacleComplexPlanner(ComplexStackingMotionPlanner):
        """支持动态障碍物的复杂堆叠规划器"""
        
        def __init__(self, base_planner):
            super().__init__(base_planner)
            self.obstacle_positions = []
        
        def update_dynamic_obstacles(self, obstacle_poses: List[sapien.Pose]):
            """更新动态障碍物位置"""
            # 清除旧的障碍物约束
            self.planner.clear_collisions()
            
            # 添加新的障碍物约束
            for pose in obstacle_poses:
                obstacle_size = np.array([0.05, 0.05, 0.15])  # 细长障碍物
                self.planner.add_box_collision(obstacle_size, pose)
            
            self.obstacle_positions = obstacle_poses
        
        def _execute_pick_and_stack(self, grasp_pose, target):
            """重写执行函数，在每个关键步骤前更新障碍物"""
            
            # 模拟动态障碍物移动
            new_obstacle_poses = [
                sapien.Pose([0.1 + 0.05*np.sin(self.planner.elapsed_steps*0.1), 
                           0.1, 0.075]),  # 摆动障碍物
                sapien.Pose([-0.1, 
                           0.1*np.cos(self.planner.elapsed_steps*0.08), 
                           0.075])   # 旋转障碍物
            ]
            
            self.update_dynamic_obstacles(new_obstacle_poses)
            
            # 调用父类的执行逻辑
            return super()._execute_pick_and_stack(grasp_pose, target)
    
    # 使用动态障碍物规划器
    dynamic_planner = DynamicObstacleComplexPlanner(base_planner)
    
    # 初始化动态障碍物
    initial_obstacles = [
        sapien.Pose([0.1, 0.1, 0.075]),
        sapien.Pose([-0.1, 0.1, 0.075])
    ]
    dynamic_planner.update_dynamic_obstacles(initial_obstacles)
    
    # 执行堆叠任务
    objects = [env.unwrapped.cubeA, env.unwrapped.cubeB]
    stacking_sequence = [
        StackingTarget(
            source_obj=objects[1],
            target_obj=objects[0],
            stack_height=0.02
        )
    ]
    
    success = dynamic_planner.solve_complex_stacking(stacking_sequence)
    
    print("🔄 动态障碍物堆叠:", "成功" if success else "失败")
    
    base_planner.close()
    return success

def _verify_stacking_stability(env, objects: List):
    """验证堆叠结构的稳定性"""
    print("🔍 验证堆叠稳定性...")
    
    # 等待物理稳定
    for _ in range(60):  # 1秒的物理模拟
        env.step(np.zeros(env.action_space.shape))
    
    # 检查物体是否保持堆叠状态
    base_height = objects[0].pose.sp.p[2]
    
    for i, obj in enumerate(objects[1:], 1):
        current_height = obj.pose.sp.p[2] 
        expected_min_height = base_height + i * 0.04  # 最小预期高度
        
        if current_height < expected_min_height:
            print(f"⚠️ 物体{i}高度不足，可能倒塌")
            return False
    
    print("✅ 堆叠结构稳定")
    return True

def benchmark_complex_stacking_algorithms():
    """
    对比不同复杂堆叠算法的性能
    """
    algorithms = [
        ("基础螺旋运动", "screw_only"),
        ("RRTConnect", "rrt_only"), 
        ("混合策略", "hybrid"),
        ("多角度重试", "multi_angle")
    ]
    
    results = {}
    
    for algo_name, algo_type in algorithms:
        print(f"\n🧪 测试算法: {algo_name}")
        
        success_count = 0
        total_time = 0
        
        for trial in range(10):
            start_time = time.time()
            # 这里应该运行具体的算法测试
            # success = run_algorithm_test(algo_type)
            success = True  # 占位符
            end_time = time.time()
            
            if success:
                success_count += 1
            total_time += (end_time - start_time)
        
        results[algo_name] = {
            "success_rate": success_count / 10,
            "avg_time": total_time / 10
        }
    
    # 输出性能对比
    print("\n📊 算法性能对比:")
    print(f"{'算法':<15} {'成功率':<10} {'平均时间(s)':<12}")
    print("-" * 40)
    
    for algo_name, metrics in results.items():
        print(f"{algo_name:<15} {metrics['success_rate']:<10.2%} {metrics['avg_time']:<12.2f}")

if __name__ == "__main__":
    # 运行复杂堆叠示例
    import time
    
    print("🚀 开始复杂堆叠Motion Planning演示")
    
    # 可以替换为你的具体环境
    # env = gym.make("StackPyramid-v1", ...)
    
    # solve_complex_pyramid_stacking(env, vis=True)
    # solve_constrained_space_stacking(env, vis=True)  
    # solve_dynamic_obstacle_stacking(env, vis=True)
    
    # benchmark_complex_stacking_algorithms()
    
    print("✨ 演示完成!")


