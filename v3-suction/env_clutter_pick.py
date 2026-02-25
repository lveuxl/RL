import os
from typing import Any, Dict, List, Union, Tuple
import numpy as np
import sapien
import torch
import cv2
import random

import mani_skill.envs.utils.randomization as randomization
from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# 新增：IK和控制器相关导入
# from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController

# 新增：导入SAPIEN约束相关模块
import sapien.physx as physx


@register_env(
    "EnvClutter-v1",
    asset_download_ids=["ycb"],
    max_episode_steps=200,
)
class EnvClutterEnv(BaseEnv):
    """
    **任务描述:**
    复杂堆叠抓取环境，包含各种形状的YCB物体堆积在托盘中。
    机械臂需要挑选最适合抓取的物体，并将其放到指定位置。
    
    **随机化:**
    - 物体在托盘内随机生成
    - 物体初始姿态随机化
    - 目标位置固定在托盘右侧
    
    **成功条件:**
    - 目标物体被成功抓取并放置到目标位置
    - 机器人静止
    """
    
    SUPPORTED_REWARD_MODES = ["dense", "sparse"]
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    # YCB物体
    BOX_OBJECTS = [
        #"003_cracker_box",          # 饼干盒
        "004_sugar_box",            # 糖盒
        "006_mustard_bottle",       # 芥末瓶
        "008_pudding_box",      # 布丁盒
        #"009_gelatin_box",          # 明胶盒
        #"010_potted_meat_can",      # 罐装肉罐头
    ]
    
    goal_thresh = 0.03  # 成功阈值
    # 托盘参数 (基于traybox.urdf的尺寸)
    tray_size = [0.6, 0.6, 0.15]  # 托盘内部尺寸 (长x宽x高)
    tray_spawn_area = [0.23, 0.23]  # 托盘内物体生成区域 (考虑边界墙和安全边距)
    num_objects_per_type = 5  # 每种类型的物体数量
    
    # 新增：离散动作相关常量
    MAX_N = len(BOX_OBJECTS) * num_objects_per_type  # 最大物体数量
    MAX_EPISODE_STEPS = 15  # 最大episode步数
    
    # 新增：吸盘约束相关常量
    SUCTION_DISTANCE_THRESHOLD = 0.1  # 吸盘激活距离阈值 (3cm)
    SUCTION_STIFFNESS = 1e6  # 吸盘约束刚度
    SUCTION_DAMPING = 1e4    # 吸盘约束阻尼
    
    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        use_discrete_action=False,  # 新增：是否使用离散动作
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_discrete_action = use_discrete_action
        
        # 设置类属性到实例属性
        self.MAX_N = len(self.BOX_OBJECTS) * self.num_objects_per_type  # 最大物体数量
        self.MAX_EPISODE_STEPS = 15  # 最大episode步数
        
        # 初始化离散动作相关变量 - 修改为多环境支持
        self.remaining_indices = []  # 每个环境的剩余可抓取物体索引 [[env0_indices], [env1_indices], ...]
        self.step_count = []  # 每个环境的当前步数 [env0_steps, env1_steps, ...]
        self.grasped_objects = []  # 每个环境已抓取的物体 [[env0_grasped], [env1_grasped], ...]
        
        # 新增：初始化吸盘约束相关变量
        self.suction_constraints = {}  # 存储约束对象的字典 {object_name: constraint}
        self.is_suction_active = False  # 吸盘是否激活
        self.current_suction_object = None  # 当前吸附的物体
        
        # 确保所有参数正确传递给父类
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21,
                max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0.0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.2, 0.35])
        return CameraConfig(
            "render_camera", 
            pose=pose, 
            width=512, 
            height=512, 
            fov=1, 
            near=0.01, 
            far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))
        
        
    
    def _move_to_position(self, target_pos: np.ndarray, steps: int = 200) -> bool:
        """
        逆运动学控制，使用pd_ee_delta_pose控制模式移动到目标位置
        
        Args:
            target_pos: 目标位置 [x, y, z]
            steps: 执行步数
            
        Returns:
            success: 是否成功到达目标位置
        """
        try:
            #print(f"开始移动到位置: {target_pos}, 步数: {steps}")
            
            # 转换为torch tensor
            if isinstance(target_pos, np.ndarray):
                target_pos = torch.tensor(target_pos, device=self.device, dtype=torch.float32)
            elif not isinstance(target_pos, torch.Tensor):
                target_pos = torch.tensor(target_pos, device=self.device, dtype=torch.float32)
            
            # 记录上一次的误差，用于检测收敛
            prev_distance = float('inf')
            stuck_count = 0  # 连续卡住的次数
            min_distance_achieved = float('inf')  # 记录达到的最小距离
            
            # 执行多步控制以到达目标位置
            for step in range(steps):
                # 获取当前末端执行器位置
                current_pos = self.agent.tcp.pose.p
                if current_pos.dim() > 1:
                    current_pos = current_pos[0]  # 取第一个环境
                
                # 计算位置误差
                pos_error = target_pos - current_pos
                current_distance = torch.linalg.norm(pos_error).item()
                
                # 更新最小距离记录
                if current_distance < min_distance_achieved:
                    min_distance_achieved = current_distance
                
                # 更严格的成功条件：误差小于2cm
                if current_distance < 0.02:  
                    #print(f"✅ 成功到达目标位置，误差: {current_distance:.4f}m，当前位置: {self.agent.tcp.pose.p}, 用时: {step}步")
                    return True
                
                # 改进的卡住检测
                distance_change = abs(current_distance - prev_distance)
                if distance_change < 0.001:  # 误差变化小于1mm
                    stuck_count += 1
                    if stuck_count > 15:  # 减少卡住阈值：从20->15
                        #print(f"⚠️ 检测到卡住，当前误差: {current_distance:.4f}m，最小误差: {min_distance_achieved:.4f}m")
                        # 如果卡住时误差小于12cm，仍然认为成功
                        if current_distance < 0.12:  # 进一步放宽卡住时的成功条件
                            #print(f"✅ 卡住但误差可接受，认为成功")
                            return True
                        else:
                            #print(f"❌ 卡住且误差过大，尝试最后几步大步长移动")
                            # 尝试最后几步大步长移动
                            for rescue_step in range(5):
                                action = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float32)
                                # 使用最大步长直接朝目标移动
                                action[:, :3] = (pos_error / torch.linalg.norm(pos_error)).unsqueeze(0).repeat(self.num_envs, 1) * 0.1
                                action[:, 3:6] = 0.0
                                action[:, 6] = 0.00
                                super().step(action)
                                
                                # 重新检查位置
                                current_pos = self.agent.tcp.pose.p
                                if current_pos.dim() > 1:
                                    current_pos = current_pos[0]
                                pos_error = target_pos - current_pos
                                current_distance = torch.linalg.norm(pos_error).item()
                                
                                if current_distance < 0.12:
                                    #print(f"✅ 救援移动成功，最终误差: {current_distance:.4f}m")
                                    return True
                            
                            #print(f"❌ 救援移动失败，最终误差: {current_distance:.4f}m")
                            return False
                else:
                    stuck_count = 0  # 重置卡住计数
                
                prev_distance = current_distance
                
                # 构建动作向量 [dx, dy, dz, drx, dry, drz, gripper]
                action = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float32)
                
                # 位置控制：充分利用控制器的0.1m最大增量能力
                max_controller_step = 0.1  # 控制器支持的最大增量：10cm
                
                # 更激进的步长策略：优先快速接近
                if current_distance > 0.15:
                    # 距离较远时，使用最大步长快速接近
                    scale_factor = 1.0  # 使用100%的控制器能力
                elif current_distance > 0.10:
                    # 中等距离时，仍然使用较大步长
                    scale_factor = 0.95  # 使用95%的控制器能力
                elif current_distance > 0.05:
                    # 接近目标时，使用中等步长
                    scale_factor = 0.8
                else:
                    # 非常接近时，使用精细控制
                    scale_factor = 0.5
                
                # 计算实际步长
                actual_max_step = max_controller_step * scale_factor
                
                # 归一化位置误差到控制器的最大增量范围
                pos_error_norm = torch.linalg.norm(pos_error)
                if pos_error_norm > actual_max_step:
                    # 如果距离大于最大步长，则按最大步长移动
                    pos_command = (pos_error / pos_error_norm) * actual_max_step
                else:
                    # 如果距离小于最大步长，则直接移动到目标位置
                    pos_command = pos_error
                
                # 为所有环境设置相同的动作
                action[:, :3] = pos_command.unsqueeze(0).repeat(self.num_envs, 1)
                
                # 姿态控制：保持垂直向下
                action[:, 3:6] = 0.0  # 不改变姿态
                
                # 夹爪控制
                action[:, 6] = 0.00  # 保持夹爪状态
                
                # 执行动作
                super().step(action)
                
                # 打印进度（减少输出频率）
                #if step % 20 == 0 or step < 5:  # 前5步和每30步输出一次
                    #print(f"步骤 {step}: 误差 {current_distance:.4f}m, 步长因子 {scale_factor:.2f}, 实际步长 {actual_max_step:.3f}m, 卡住计数 {stuck_count}")
            
            # 检查最终误差
            final_pos = self.agent.tcp.pose.p
            if final_pos.dim() > 1:
                final_pos = final_pos[0]
            final_error = torch.linalg.norm(target_pos - final_pos).item()
            #print(f"移动完成，最终误差: {final_error:.4f}m，最小误差: {min_distance_achieved:.4f}m")
            
            # 放宽成功条件：误差小于12cm
            success = final_error < 0.12  # 进一步放宽：从10cm->12cm
            if success:
                #print(f"✅ 移动成功，误差在可接受范围内: {final_error:.4f}m")
                pass
            else:
                #print(f"❌ 移动失败，误差过大: {final_error:.4f}m")
                pass
            
            return success
            
        except Exception as e:
            #print(f"❌ 移动到位置失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 新增：吸盘约束系统实现
    def _create_suction_constraint(self, target_object: Actor) -> bool:
        """
        创建吸盘约束
        
        Args:
            target_object: 目标物体
            
        Returns:
            bool: 是否成功创建约束
        """
        if self.is_suction_active:
            #print("吸盘已经激活，无法创建新约束")
            return False
            
        # 检查是否与物体接触
        if not self._is_contacting_object(target_object, self.SUCTION_DISTANCE_THRESHOLD):
            #print(f"物体距离过远，无法激活吸盘")
            return False
        
        try:
            # 导入Drive类
            from mani_skill.utils.structs.drive import Drive
            
            #print(f"创建吸盘约束: TCP链接 -> 物体 {target_object.name}")
            #print(f"TCP类型: {type(self.agent.tcp)}, 目标物体类型: {type(target_object)}")
            
            # 使用Drive.create_from_actors_or_links方法创建约束
            # 这个方法会正确处理Link和Actor对象的entity获取
            constraint = Drive.create_from_actors_or_links(
                scene=self.scene,
                entities0=self.agent.tcp,     # TCP是Link对象
                pose0=sapien.Pose(),          # 父体本地姿态
                entities1=target_object,      # 目标物体是Actor对象
                pose1=sapien.Pose(),          # 子体本地姿态
                scene_idxs=torch.tensor([0], device=self.device)  # 场景索引
            )
            
            # 设置约束参数使其表现为固定约束（类似PyBullet的JOINT_FIXED）
            # 线性约束（X, Y, Z方向）
            constraint.set_drive_property_x(stiffness=self.SUCTION_STIFFNESS, damping=self.SUCTION_DAMPING)
            constraint.set_drive_property_y(stiffness=self.SUCTION_STIFFNESS, damping=self.SUCTION_DAMPING)
            constraint.set_drive_property_z(stiffness=self.SUCTION_STIFFNESS, damping=self.SUCTION_DAMPING)
            
            # 角度约束（绕X, Y, Z轴的旋转）
            # 注意：Drive对象可能没有角度约束方法，先尝试设置限制
            try:
                # 设置限制来模拟固定约束
                constraint.set_limit_x(0, 0)  # 不允许X方向移动
                constraint.set_limit_y(0, 0)  # 不允许Y方向移动
                constraint.set_limit_z(0, 0)  # 不允许Z方向移动
                #print("✅ 已设置位置限制")
            except Exception as limit_error:
                pass
                #print(f"⚠️ 设置限制失败: {limit_error}")
                # 继续执行，仅使用驱动属性
            
            # 存储约束
            self.suction_constraints[target_object.name] = constraint
            self.is_suction_active = True
            self.current_suction_object = target_object
            
            #print(f"✅ 吸盘约束创建成功: {target_object.name}")
            return True
            
        except Exception as e:
            #print(f"❌ 创建吸盘约束失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _remove_suction_constraint(self) -> bool:
        """
        移除吸盘约束
        
        Returns:
            bool: 是否成功移除约束
        """
        if not self.is_suction_active or self.current_suction_object is None:
            #print("没有激活的吸盘约束需要移除")
            return False
        
        try:
            # 获取约束对象
            constraint_name = self.current_suction_object.name
            if constraint_name in self.suction_constraints:
                constraint = self.suction_constraints[constraint_name]
                
                #print(f"正在禁用约束: {constraint_name}")
                
                # 方法1: 通过设置刚度为0来禁用约束（最有效）
                try:
                    #print("设置约束刚度为0...")
                    constraint.set_drive_property_x(stiffness=0.0, damping=0.0)
                    constraint.set_drive_property_y(stiffness=0.0, damping=0.0)
                    constraint.set_drive_property_z(stiffness=0.0, damping=0.0)
                    #print("✅ 成功禁用约束驱动属性")
                except Exception as disable_error:
                    #print(f"❌ 禁用约束驱动属性失败: {disable_error}")
                    return False
                
                # 方法2: 重置约束限制（辅助方法）
                try:
                    #print("重置约束限制...")
                    # 设置非常大的限制范围，相当于取消限制
                    constraint.set_limit_x(-1000, 1000)
                    constraint.set_limit_y(-1000, 1000)
                    constraint.set_limit_z(-1000, 1000)
                    #print("✅ 成功重置约束限制")
                except Exception as limit_error:
                    #print(f"⚠️ 重置约束限制失败: {limit_error}")
                    # 限制重置失败不影响主要功能，继续执行
                    pass
                
                # 清理约束引用
                del self.suction_constraints[constraint_name]
                #print(f"✅ 约束引用已清理: {constraint_name}")
            else:
                print(f"⚠️ 未找到约束对象: {constraint_name}")
            
            # 重置吸盘状态
            self.is_suction_active = False
            self.current_suction_object = None
            
            #print("✅ 吸盘状态已重置")
            return True
            
        except Exception as e:
            #print(f"❌ 移除吸盘约束失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 即使移除失败，也要重置状态
            self.is_suction_active = False
            self.current_suction_object = None
            return False

    def _is_contacting_object(self, target_object: Actor, threshold: float = 0.05) -> bool:
        """
        检测TCP是否与物体接触
        
        Args:
            target_object: 目标物体
            threshold: 距离阈值
            
        Returns:
            bool: 是否接触
        """
        try:
            # 计算TCP到物体的距离
            tcp_pos = self.agent.tcp.pose.p
            if tcp_pos.dim() > 1:
                tcp_pos = tcp_pos[0]
                
            obj_pos = target_object.pose.p
            if obj_pos.dim() > 1:
                obj_pos = obj_pos[0]
            
            # 计算距离
            distance = torch.linalg.norm(tcp_pos - obj_pos).item() -0.05
            
            #print(f"TCP到物体距离: {distance:.4f}m, 阈值: {threshold:.4f}m")
            
            # 检查是否在接触阈值内
            return distance <= threshold
            
        except Exception as e:
            #print(f"检测接触失败: {e}")
            return False

    def _check_suction_grasp_success(self, target_object: Actor) -> bool:
        """
        检查吸盘抓取是否成功
        
        Args:
            target_object: 目标物体
            
        Returns:
            bool: 抓取是否成功
        """
        try:
            # 方法1：检查吸盘状态
            if (self.is_suction_active and 
                self.current_suction_object is not None and 
                self.current_suction_object.name == target_object.name):
                
                # 方法2：检查物体是否仍在TCP附近
                tcp_pos = self.agent.tcp.pose.p
                if tcp_pos.dim() > 1:
                    tcp_pos = tcp_pos[0]
                
                obj_pos = target_object.pose.p
                if obj_pos.dim() > 1:
                    obj_pos = obj_pos[0]
                
                distance = torch.linalg.norm(tcp_pos - obj_pos).item()-0.10
                
                # 距离小于5cm认为抓取成功
                success = distance < 0.05
                
                if success:
                    #print(f"✅ 吸盘抓取成功验证: 距离={distance:.4f}m")
                    pass
                else:
                    #print(f"❌ 吸盘抓取失败: 距离过远={distance:.4f}m")
                    pass
                
                return success
            else:
                #print("❌ 吸盘未激活或物体不匹配")
                return False
                
        except Exception as e:
            #print(f"检查吸盘抓取成功失败: {e}")
            return False
    
    def _pick_object_8_states(self, obj_idx: int, env_idx: int = 0) -> Tuple[bool, float]:
        """
        8状态抓取流程
        
        Args:
            obj_idx: 物体索引
            env_idx: 环境索引（多环境支持）
            
        Returns:
            success: 抓取是否成功
            displacement: 其他物体的位移
        """
        if obj_idx >= len(self.all_objects):
            #print(f"物体索引{obj_idx}超出范围")
            return False, 0.0
        
        # 获取目标物体
        target_obj = self.all_objects[obj_idx]
        
        # 记录抓取前其他物体的位置
        other_objects_pos_before = []
        for i, obj in enumerate(self.all_objects):
            # 修复：使用多环境的grasped_objects
            grasped_list = self.grasped_objects[env_idx] if env_idx < len(self.grasped_objects) else []
            if i != obj_idx and i not in grasped_list:
                obj_pos = obj.pose.p
                if obj_pos.dim() > 1:
                    obj_pos = obj_pos[env_idx] if env_idx < obj_pos.shape[0] else obj_pos[0]
                other_objects_pos_before.append(obj_pos.clone())
        
        try:
            # 获取目标物体位置
            obj_pos = target_obj.pose.p
            if obj_pos.dim() > 1:
                obj_pos = obj_pos[env_idx] if env_idx < obj_pos.shape[0] else obj_pos[0]
            obj_pos = obj_pos.cpu().numpy()
            
             # 等待物体稳定落下
            for _ in range(5):  # 等待30步让物体稳定
                action = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float32)
                super().step(action)

            #print(f"开始8状态抓取流程，目标物体{obj_idx}，位置: {obj_pos}")
            
            # 检查是否在机械臂范围内
            robot_base = np.array([-0.615, 0, 0])  # 机械臂基座位置
            distance = np.linalg.norm(obj_pos[:2] - robot_base[:2])
            if distance > 0.6:  # 最大抓取范围60cm
                #print(f"物体{obj_idx}超出机械臂范围，距离: {distance:.3f}m")
                return False, 0.0
            
            # # 检查是否被遮挡（简化版射线检测）
            # if self._is_object_blocked(target_obj):
            #     print(f"物体{obj_idx}被遮挡")
            #     return False, 0.0
            
            # === 状态0: 机械臂上升到物体上方 ===
            #print("状态0: 机械臂上升到物体上方")
            approach_pos = obj_pos.copy()
            approach_pos[2] += 0.15  # 减少高度：从20cm改为15cm
            
            if not self._move_to_position(approach_pos, steps=150):  # 增加步数：100->150
                #print("状态0失败：无法移动到物体上方")
                return False, 0.0
            
            # === 状态1: 机械臂下降到物体上方准备抓取 ===
            #print("状态1: 机械臂下降到物体上方")
            descend_pos = obj_pos.copy()
            descend_pos[2] += 0.03  # 减少高度：从5cm改为3cm，更接近物体
            
            if not self._move_to_position(descend_pos, steps=80):  # 增加步数：50->80
                #print("状态1失败：无法下降到物体上方")
                return False, 0.0
            
            # === 状态2: 吸取/抓取物体 ===
            #print("状态2: 抓取物体")
            grasp_pos = obj_pos.copy()
            grasp_pos[2] += 0.01  # 减少高度：从2cm改为1cm，更贴近物体
            
            if not self._move_to_position(grasp_pos, steps=80):  # 增加步数：50->80
                #print("状态2失败：无法抓取物体")
                return False, 0.0

            # 使用吸盘约束替代夹爪控制
            #  print("状态2: 激活吸盘约束")
            suction_success = self._create_suction_constraint(target_obj)
            if not suction_success:
                #print("状态2失败：无法创建吸盘约束")
                return False, 0.0

            # 检查抓取是否成功
            if not self._check_suction_grasp_success(target_obj):
                #print("状态2失败：吸盘抓取不成功")
                self._remove_suction_constraint()  # 清理失败的约束
                return False, 0.0
            
            # === 状态3: 物体上升 ===
            #print("状态3: 物体上升")
            lift_pos = grasp_pos.copy()
            lift_pos[2] += 0.2  # 减少高度：从30cm改为20cm
            
            if not self._move_to_position(lift_pos, steps=100):  # 增加步数：80->100
                #print("状态3失败：无法提升物体")
                self._remove_suction_constraint()  # 清理约束
                return False, 0.0
            
            # === 状态4: 移动到放置位置 ===
            #print("状态4: 移动到放置位置")
            transport_pos = np.array([-0.4, 0.4, lift_pos[2]])  # 目标区域上方
            
            if not self._move_to_position(transport_pos, steps=180):  # 增加步数：100->120
                #print("状态4失败：无法移动到放置位置")
                self._remove_suction_constraint()  # 清理约束
                return False, 0.0
            
            # === 状态5: 下降到放置位置 ===
            #print("状态5: 下降到放置位置")
            lower_pos = transport_pos.copy()
            lower_pos[2] -= 0.2  # 减少高度：从30cm改为20cm
            
            if not self._move_to_position(lower_pos, steps=100):  # 增加步数：80->100
                #print("状态5失败：无法下降到放置位置")
                self._remove_suction_constraint()  # 清理约束
                return False, 0.0
            
            # === 状态6: 放下物体 ===
            #print("状态6: 放下物体")
            release_pos = lower_pos.copy()
            
            if not self._move_to_position(release_pos, steps=20):  # 保持较少步数：30->20
                #print("状态6失败：无法放下物体")
                self._remove_suction_constraint()  # 清理约束
                return False, 0.0
            
            # 移除吸盘约束替代打开夹爪
            #print("状态6: 移除吸盘约束")
            if not self._remove_suction_constraint():
                #print("⚠️ 移除吸盘约束失败，但继续执行")
                pass
            
            # 等待物体稳定落下
            for _ in range(5):  # 等待30步让物体稳定
                action = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float32)
                super().step(action)
            
            # === 状态7: 回到初始位置 ===
            #print("状态7: 回到初始位置")
            home_pos = np.array([-0.6, 0.4, 0.4])  # 回到安全位置
            
            if not self._move_to_position(home_pos, steps=100):  # 增加步数：50->100
                #print("状态7失败：无法回到初始位置")
                # 这里不返回失败，因为物体已经成功放置
                pass
            
            # 修复：标记抓取成功 - 使用多环境格式
            if env_idx < len(self.grasped_objects):
                self.grasped_objects[env_idx].append(obj_idx)
            #print(f"8状态抓取流程完成，成功抓取物体{obj_idx}")
            
            # 计算其他物体的位移
            displacement = 0.0
            grasped_list = self.grasped_objects[env_idx] if env_idx < len(self.grasped_objects) else []
            for i, obj in enumerate(self.all_objects):
                if i != obj_idx and i not in grasped_list:
                    obj_pos_after = obj.pose.p
                    if obj_pos_after.dim() > 1:
                        obj_pos_after = obj_pos_after[env_idx] if env_idx < obj_pos_after.shape[0] else obj_pos_after[0]
                    
                    # 找到对应的初始位置
                    if len(other_objects_pos_before) > 0:
                        # 简化处理：使用第一个可用的初始位置
                        pos_before = other_objects_pos_before[0] if other_objects_pos_before else obj_pos_after
                        displacement += torch.linalg.norm(obj_pos_after - pos_before).item()
            
            return True, displacement
            
        except Exception as e:
            #print(f"8状态抓取流程出错: {e}")
            return False, 0.0
    
    def _low_level_step(self, delta_pose: torch.Tensor):
        """单步执行delta pose，只推进仿真，不走离散逻辑"""
        # 调用父类的step方法执行连续动作
        super().step(delta_pose)
    
    
    def _is_object_blocked(self, target_obj) -> bool:
        """
        简化的遮挡检测，对应PyBullet的射线检测
        检查物体上方是否有其他物体
        """
        try:
            target_pos = target_obj.pose.p
            if target_pos.dim() > 1:
                target_pos = target_pos[0]
            
            # 检查是否有其他物体在目标物体上方
            for obj in self.all_objects:
                if obj == target_obj:
                    continue
                
                obj_pos = obj.pose.p
                if obj_pos.dim() > 1:
                    obj_pos = obj_pos[0]
                
                # 检查是否在目标物体上方（xy平面距离小于5cm，z高度大于目标物体）
                xy_distance = torch.linalg.norm(obj_pos[:2] - target_pos[:2])
                if xy_distance < 0.05 and obj_pos[2] > target_pos[2]:
                    return True
            
            return False
            
        except Exception as e:
            #print(f"遮挡检测失败: {e}")
            return False


    def _load_scene(self, options: dict):
        # 构建桌面场景
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        
        # 加载托盘
        self._load_tray()
        
        # 创建物体列表
        self.all_objects = []
        self.selectable_objects = []
        self.object_info = []  # 存储物体信息
        
        # 为每个环境创建物体
        for env_idx in range(self.num_envs):
            env_objects = []
            env_selectable = []
            env_info = []
            
            # 创建每种类型的物体
            for obj_type in self.BOX_OBJECTS:
                for i in range(self.num_objects_per_type):
                    # 创建物体
                    builder = actors.get_actor_builder(self.scene, id=f"ycb:{obj_type}")
                    
                    # 在托盘内随机生成位置
                    x, y, z = self._generate_object_position_in_tray(i)
                    
                    # 随机姿态
                    quat = randomization.random_quaternions(1)[0]
                    initial_pose = sapien.Pose(p=[x, y, z], q=quat.cpu().numpy())
                    
                    builder.initial_pose = initial_pose
                    builder.set_scene_idxs([env_idx])
                    
                    obj_name = f"env_{env_idx}_{obj_type}_{i}"
                    obj = builder.build(name=obj_name)
                    
                    env_objects.append(obj)
                    env_selectable.append(obj)
                    
                    # 存储物体信息
                    obj_info = {
                        'type': obj_type,
                        'size': self._get_object_size(obj_type),
                        'initial_pose': initial_pose,
                        'center': [x, y, z],
                        'exposed_area': 1.0,  # 初始暴露面积，后续会计算
                    }
                    env_info.append(obj_info)
            
            self.all_objects.extend(env_objects)
            self.selectable_objects.append(env_selectable)
            self.object_info.append(env_info)
        
        # 合并所有物体
        if self.all_objects:
            self.merged_objects = Actor.merge(self.all_objects, name="all_objects")
        
        # 创建目标位置标记
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)
        
        # 初始化目标物体相关变量
        self.target_object = None
        self.target_object_indices = []

    def _load_tray(self):
        """加载托盘URDF文件"""
        # 获取托盘URDF文件路径
        tray_urdf_path = "/home2/jzh/RL_RobotArm-main/assets/tray/traybox.urdf"
        
        if not os.path.exists(tray_urdf_path):
            raise FileNotFoundError(f"托盘URDF文件未找到: {tray_urdf_path}")
        
        # 创建URDF加载器
        loader = self.scene.create_urdf_loader()
        
        # 设置托盘的物理属性
        loader.set_material(static_friction=0.8, dynamic_friction=0.6, restitution=0.1)
        loader.fix_root_link = True  # 固定托盘不动
        loader.scale = 1.0  # 保持原始尺寸
        
        # 解析URDF文件
        parsed_result = loader.parse(tray_urdf_path)
        
        # 只使用 actor_builders 方式
        actor_builders = parsed_result.get("actor_builders", [])
        
        if not actor_builders:
            raise ValueError("托盘URDF文件中没有找到actor_builders")
        
        self.trays = []
        
        # 使用 actor_builders 加载托盘
        for env_idx in range(self.num_envs):
            builder = actor_builders[0]
            # 设置托盘位置 (放在桌面上，机器人前方)
            tray_position = [-0.2, 0.0, 0.02]  # 桌面高度加上托盘底部厚度
            builder.initial_pose = sapien.Pose(p=tray_position)
            builder.set_scene_idxs([env_idx])
            
            # 使用 build_static 创建静态托盘，确保不会移动
            tray = builder.build_static(name=f"tray_{env_idx}")
            self.trays.append(tray)
        
        # 合并所有托盘
        if self.trays:
            self.merged_trays = Actor.merge(self.trays, name="all_trays")
        
        #print(f"成功加载托盘，共 {len(self.trays)} 个")

    def _generate_object_position_in_tray(self, stack_level=0):
        """在托盘内生成物体位置"""
        # 托盘中心位置
        tray_center_x = -0.2
        tray_center_y = 0.0
        tray_bottom_z = 0.02 + 0.04  # 托盘底部 + 小偏移
        
        # 托盘边界计算（基于URDF文件中的边界墙位置）
        # 边界墙在托盘中心的±0.25米处，考虑边界墙厚度0.02米
        # 实际可用空间：从中心向两边各0.23米（留出安全边距）
        safe_spawn_area_x = 0.23
        safe_spawn_area_y = 0.23
        
        # 在托盘内随机生成xy位置
        x = tray_center_x + random.uniform(-safe_spawn_area_x, safe_spawn_area_x)
        y = tray_center_y + random.uniform(-safe_spawn_area_y, safe_spawn_area_y)
        
        # 堆叠高度
        z = tray_bottom_z + stack_level * 0.02  # 每层高度
        
        return x, y, z

    def _get_object_size(self, obj_type):
        """获取物体的大小信息"""
        # 基于YCB数据集的实际物体尺寸（单位：米）
        sizes = {
            #"003_cracker_box": [0.16, 0.21, 0.07],         # 饼干盒: 16cm x 21cm x 7cm
            "004_sugar_box": [0.09, 0.175, 0.044],         # 糖盒: 9cm x 17.5cm x 4.4cm
            "006_mustard_bottle": [0.095, 0.095, 0.177],   # 芥末瓶: 9.5cm x 9.5cm x 17.7cm
            "008_pudding_box": [0.078, 0.109, 0.032],      # 布丁盒: 7.8cm x 10.9cm x 3.2cm
            #"009_gelatin_box": [0.028, 0.085, 0.114],      # 明胶盒: 2.8cm x 8.5cm x 11.4cm  
            #"010_potted_meat_can": [0.101, 0.051, 0.051],  # 罐装肉罐头: 10.1cm x 5.1cm x 5.1cm
           
        }
        return sizes.get(obj_type, [0.05, 0.05, 0.05])

    def _sample_target_objects(self):
        """随机选择目标物体"""
        target_objects = []
        self.target_object_indices = []
        
        for env_idx in range(self.num_envs):
            if env_idx < len(self.selectable_objects) and self.selectable_objects[env_idx]:
                # 随机选择一个可选择的物体
                target_idx = random.randint(0, len(self.selectable_objects[env_idx]) - 1)
                target_obj = self.selectable_objects[env_idx][target_idx]
                target_objects.append(target_obj)
                self.target_object_indices.append(target_idx)
        
        if target_objects:
            self.target_object = Actor.merge(target_objects, name="target_object")

    def _calculate_exposed_area(self, env_idx):
        """计算物体的暴露面积"""
        # 这里是简化的暴露面积计算
        # 实际应用中可能需要更复杂的几何计算
        if env_idx < len(self.object_info):
            for i, obj_info in enumerate(self.object_info[env_idx]):
                # 基于物体高度和周围物体数量的简单估算
                exposed_area = max(0.1, 1.0 - i * 0.1)  # 越高的物体暴露面积越大
                obj_info['exposed_area'] = exposed_area

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化每个episode"""
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            
            # 重置托盘位置
            if hasattr(self, 'merged_trays'):
                # 在GPU仿真中，静态对象不能改变位姿，所以跳过
                if not self.scene.gpu_sim_enabled:
                    if b == self.num_envs:
                        self.merged_trays.pose = self.merged_trays.initial_pose
                    else:
                        mask = torch.isin(self.merged_trays._scene_idxs, env_idx)
                        self.merged_trays.pose = self.merged_trays.initial_pose[mask]
                else:
                    #print("GPU仿真模式下跳过静态托盘位姿重置")
                    pass
            
            # 重置物体到初始位置
            if hasattr(self, 'merged_objects'):
                if b == self.num_envs:
                    self.merged_objects.pose = self.merged_objects.initial_pose
                else:
                    mask = torch.isin(self.merged_objects._scene_idxs, env_idx)
                    self.merged_objects.pose = self.merged_objects.initial_pose[mask]
            
            # 设置目标位置 - 固定在托盘右侧
            goal_pos = torch.zeros((b, 3), device=self.device)
            
            # 托盘中心位置：[-0.2, 0.0, 0.02]
            # 托盘尺寸：长0.6m，宽0.6m
            # 目标位置设定在托盘右侧外10cm处，避免与托盘边界冲突
            goal_pos[:, 0] = -0.4  # 托盘右侧的固定位置
            goal_pos[:, 1] = 0.4  
            goal_pos[:, 2] = 0.05  # 桌面高度5cm，确保物体稳定放置
            
            self.goal_pos = goal_pos
            self.goal_site.set_pose(Pose.create_from_pq(self.goal_pos))
            
            # 记录初始物体位置（用于计算位移）
            self.initial_object_positions = []
            for i in range(b):
                env_positions = []
                for obj in self.all_objects:
                    if hasattr(obj, '_scene_idxs') and len(obj._scene_idxs) > 0:
                        if obj._scene_idxs[0] == env_idx[i]:
                            env_positions.append(obj.pose.p.clone())
                self.initial_object_positions.append(env_positions)
            
            # 计算暴露面积
            for i in range(b):
                self._calculate_exposed_area(env_idx[i])
            
            # 重新选择目标物体
            self._sample_target_objects()
            
            # 新增：初始化离散动作相关变量
            if self.use_discrete_action:
                # 为每个环境初始化状态
                if len(self.remaining_indices) != self.num_envs:
                    self.remaining_indices = [list(range(self.MAX_N)) for _ in range(self.num_envs)]
                    self.step_count = [0 for _ in range(self.num_envs)]
                    self.grasped_objects = [[] for _ in range(self.num_envs)]
                else:
                    # 重置指定环境的状态
                    for i, env_id in enumerate(env_idx):
                        env_id_int = env_id.item() if hasattr(env_id, 'item') else int(env_id)
                        self.remaining_indices[env_id_int] = list(range(self.MAX_N))
                        self.step_count[env_id_int] = 0
                        self.grasped_objects[env_id_int] = []
            
            # 新增：重置吸盘约束状态
            self.suction_constraints = {}
            self.is_suction_active = False
            self.current_suction_object = None
            
            
            # 使用默认重置
            self.agent.reset()

    def _get_obs_extra(self, info: Dict):
        """获取额外观测信息"""
        # 获取批次大小
        batch_size = self.num_envs
        
        # 基础观测信息
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        
        if "state" in self.obs_mode:
            if hasattr(self, 'target_object') and self.target_object is not None:
                obs.update(
                    target_obj_pose=self.target_object.pose.raw_pose,
                    tcp_to_obj_pos=self.target_object.pose.p - self.agent.tcp.pose.p,
                    obj_to_goal_pos=self.goal_site.pose.p - self.target_object.pose.p,
                )
            else:
                # 如果没有目标物体，提供零张量以保持维度一致
                zero_pose = torch.zeros((batch_size, 7), device=self.device)
                zero_pos = torch.zeros((batch_size, 3), device=self.device)
                obs.update(
                    target_obj_pose=zero_pose,
                    tcp_to_obj_pos=zero_pos,
                    obj_to_goal_pos=zero_pos,
                )
            
            # 添加标量观测信息，确保维度一致
            obs.update(
                num_objects=torch.tensor([len(self.all_objects)], device=self.device).repeat(batch_size),
            )
        
        # 新增：离散动作相关观测
        if self.use_discrete_action:
            # 创建掩码：未抓取=1，已抓取=0
            mask = torch.ones(batch_size, self.MAX_N, device=self.device)
            
            # 为每个环境设置掩码
            for env_idx in range(batch_size):
                if env_idx < len(self.grasped_objects):
                    for grasped_idx in self.grasped_objects[env_idx]:
                        if grasped_idx < self.MAX_N:
                            mask[env_idx, grasped_idx] = 0
            
            # 物体特征：中心坐标、尺寸、暴露面积
            object_features = torch.zeros(batch_size, self.MAX_N, 7, device=self.device)  # 3+3+1=7维特征
            
            for env_idx in range(batch_size):
                for obj_idx, obj in enumerate(self.all_objects):
                    if obj_idx < self.MAX_N:
                        # 检查是否已被抓取
                        is_grasped = (env_idx < len(self.grasped_objects) and 
                                    obj_idx in self.grasped_objects[env_idx])
                        if not is_grasped:
                            # 获取物体信息 - 修复索引越界问题
                            obj_pose_p = obj.pose.p
                            if len(obj_pose_p.shape) > 1 and obj_pose_p.shape[0] > env_idx:
                                # 多环境情况，且当前环境索引有效
                                pos = obj_pose_p[env_idx]
                            elif len(obj_pose_p.shape) > 1 and obj_pose_p.shape[0] == 1:
                                # 多环境格式但只有一个环境
                                pos = obj_pose_p[0]
                            else:
                                # 单环境情况
                                pos = obj_pose_p
                            
                            # 获取物体类型和尺寸
                            obj_type_idx = obj_idx // self.num_objects_per_type
                            if obj_type_idx < len(self.BOX_OBJECTS):
                                obj_type = self.BOX_OBJECTS[obj_type_idx]
                                size = self._get_object_size(obj_type)
                            else:
                                size = [0.05, 0.05, 0.05]  # 默认尺寸
                            
                            # 获取暴露面积
                            exposed_area = 1.0  # 简化处理
                            if hasattr(self, 'object_info') and env_idx < len(self.object_info):
                                if obj_idx < len(self.object_info[env_idx]):
                                    exposed_area = self.object_info[env_idx][obj_idx].get('exposed_area', 1.0)
                            
                            # 组合特征：[x, y, z, w, h, d, exposed_area]
                            object_features[env_idx, obj_idx] = torch.tensor([
                                pos[0].item() if hasattr(pos[0], 'item') else pos[0],
                                pos[1].item() if hasattr(pos[1], 'item') else pos[1],
                                pos[2].item() if hasattr(pos[2], 'item') else pos[2],
                                size[0], size[1], size[2],
                                exposed_area
                            ], device=self.device)
            
            # 将离散动作相关的观测展平为1D向量，避免维度不匹配问题
            # 将action_mask展平为1D
            action_mask_flat = mask.flatten(start_dim=1)  # [batch_size, MAX_N]
            
            # 将object_features展平为1D
            object_features_flat = object_features.flatten(start_dim=1)  # [batch_size, MAX_N*7]
            
            # 将step_count扩展为与batch_size匹配的1D向量
            step_counts = []
            for env_idx in range(batch_size):
                if env_idx < len(self.step_count):
                    step_counts.append(self.step_count[env_idx])
                else:
                    step_counts.append(0)
            step_count_expanded = torch.tensor(step_counts, device=self.device).unsqueeze(1)  # [batch_size, 1]
            
            # 将所有离散动作相关的观测合并为一个统一的张量
            discrete_obs = torch.cat([
                action_mask_flat,
                object_features_flat,
                step_count_expanded
            ], dim=1)  # [batch_size, MAX_N + MAX_N*7 + 1]
            
            obs.update(
                discrete_action_obs=discrete_obs,
            )
            
            # 在离散动作模式下，直接返回discrete_action_obs作为主要观测
            # 这样可以避免维度不匹配的问题
            return discrete_obs  # 直接返回 [batch_size, feature_dim] 格式
        
        return obs

    
    def _close_gripper(self):
        """闭合夹爪"""
        # 构建7维动作向量 [dx, dy, dz, drx, dry, drz, gripper]
        action = torch.zeros(self.num_envs, 7, device=self.device)
        action[:, 6] = 0.00  # 闭合夹爪
        
        # 执行多步以确保夹爪闭合
        for _ in range(5):
            self._low_level_step(action)
    
    def _open_gripper(self):
        """打开夹爪"""
        # 构建7维动作向量 [dx, dy, dz, drx, dry, drz, gripper]
        action = torch.zeros(self.num_envs, 7, device=self.device)
        action[:, 6] = 0.04  # 打开夹爪
        
        # 执行多步以确保夹爪打开
        for _ in range(5):
            self._low_level_step(action)

    def step(self, action):
        """
        覆盖step方法以支持离散动作选择
        
        Args:
            action: 如果use_discrete_action=True，则为物体索引；否则为连续动作
        """
        if self.use_discrete_action:
            return self._discrete_step(action)
        else:
            # 调用父类的连续动作step
            return super().step(action)
    
    def _discrete_step(self, action):
        """
        处理离散动作的step方法 - 多环境版本
        
        Args:
            action: 要抓取的物体索引，形状为(num_envs,)或标量
        """
        # 确保action是正确的形状
        if isinstance(action, (int, np.integer)):
            # 单个动作，复制到所有环境
            action = np.full(self.num_envs, action)
        elif isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, np.ndarray):
            if action.shape == ():  # 标量数组
                action = np.full(self.num_envs, action.item())
        
        # 确保action是正确长度的数组
        if len(action) != self.num_envs:
            print(f"警告：动作长度{len(action)}与环境数量{self.num_envs}不匹配")
            action = np.full(self.num_envs, action[0] if len(action) > 0 else 0)
        
        # 为每个环境处理动作
        rewards = []
        terminated = []
        truncated = []
        infos = []
        
        for env_idx in range(self.num_envs):
            env_action = int(action[env_idx])
            
            # 检查动作合法性
            if env_action < 0 or env_action >= len(self.remaining_indices[env_idx]):
                # 非法动作，给予惩罚
                reward = -1.0
                terminated_env = False
                info = {
                    'success': False,
                    'displacement': 0.0,
                    'remaining_objects': len(self.remaining_indices[env_idx]),
                    'grasped_objects': len(self.grasped_objects[env_idx]),
                }
            else:
                # 获取实际的物体索引
                target_idx = self.remaining_indices[env_idx][env_action]
                
                # 执行8状态抓取流程（简化版，避免阻塞）
                success, displacement = self._pick_object_8_states(target_idx, env_idx)
                
                # 从剩余列表中移除已尝试的物体
                self.remaining_indices[env_idx].pop(env_action)
                
                # 更新步数
                self.step_count[env_idx] += 1
                
                if success:
                    self.grasped_objects[env_idx].append(target_idx)
                
                # 计算奖励
                reward = self.compute_select_reward(success, displacement)
                
                # 检查终止条件
                terminated_env = (self.step_count[env_idx] >= self.MAX_EPISODE_STEPS or 
                                len(self.remaining_indices[env_idx]) == 0)
                
                info = {
                    'success': success,
                    'displacement': displacement,
                    'remaining_objects': len(self.remaining_indices[env_idx]),
                    'grasped_objects': len(self.grasped_objects[env_idx]),
                }
            
            rewards.append(reward)
            terminated.append(terminated_env)
            truncated.append(False)
            infos.append(info)
        
        # 转换为torch张量
        reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminated, device=self.device, dtype=torch.bool)
        truncated_tensor = torch.tensor(truncated, device=self.device, dtype=torch.bool)
        
        # 获取新的观测
        eval_info = self.evaluate()
        eval_info.update({
            'success': torch.tensor([info['success'] for info in infos], device=self.device, dtype=torch.bool),
            'displacement': sum(info['displacement'] for info in infos),
            'remaining_objects': sum(info['remaining_objects'] for info in infos),
            'grasped_objects': sum(info['grasped_objects'] for info in infos),
        })
        
        obs = self._get_obs_extra(eval_info)
        
        return obs, reward_tensor, terminated_tensor, truncated_tensor, eval_info
    
    def _get_failed_step_result(self):
        """获取失败步骤的结果"""
        # 惩罚性奖励 - 转换为torch.Tensor
        reward = torch.tensor([-1.0], device=self.device, dtype=torch.float32)
        
        # 不终止，让智能体学习 - 转换为torch.Tensor
        terminated = torch.tensor([False], device=self.device, dtype=torch.bool)
        truncated = torch.tensor([False], device=self.device, dtype=torch.bool)
        
        # 获取当前观测
        info = self.evaluate()
        info.update({
            'success': False,
            'displacement': 0.0,
            'remaining_objects': sum(len(env_remaining) for env_remaining in self.remaining_indices),
            'grasped_objects': sum(len(env_grasped) for env_grasped in self.grasped_objects),
        })
        
        obs = self._get_obs_extra(info)
        
        return obs, reward, terminated, truncated, info

    def compute_select_reward(self, grasp_success: bool, displacement: float) -> float:
        """
        计算离散动作选择的奖励
        
        Args:
            grasp_success: 抓取是否成功
            displacement: 其他物体的位移
            
        Returns:
            reward: 奖励值
        """
        # 基础奖励：成功抓取
        reward = 1.0 if grasp_success else 0.0
        
        # 位移惩罚
        reward -= displacement * 1.5  # disp_coeff
        
        # 时间惩罚
        reward -= 0.01  # time_coeff
        
        return reward

    @property
    def discrete_action_space(self):
        """获取离散动作空间"""
        if self.use_discrete_action:
            import gymnasium as gym
            return gym.spaces.Discrete(self.MAX_N)
        else:
            return None

    def evaluate(self):
        """评估任务完成情况"""
        success = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_grasped = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_robot_static = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_obj_placed = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        
        if hasattr(self, 'target_object') and self.target_object is not None:
            # 检查物体是否放置到目标位置
            obj_to_goal_dist = torch.linalg.norm(
                self.goal_site.pose.p - self.target_object.pose.p, axis=1
            )
            is_obj_placed = obj_to_goal_dist <= self.goal_thresh
            
            # 检查是否抓取
            is_grasped = self.agent.is_grasping(self.target_object)
            
            # 检查机器人是否静止
            is_robot_static = self.agent.is_static(0.2)
            
            # 成功条件：物体放置到位且机器人静止
            success = is_obj_placed & is_robot_static
        
        return {
            "success": success,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def _calculate_other_objects_displacement(self):
        """计算其他物体的位移距离"""
        total_displacement = torch.zeros(self.num_envs, device=self.device)
        
        for env_idx in range(self.num_envs):
            displacement = 0.0
            obj_count = 0
            
            for i, obj in enumerate(self.all_objects):
                if hasattr(obj, '_scene_idxs') and len(obj._scene_idxs) > 0:
                    if obj._scene_idxs[0] == env_idx:
                        # 跳过目标物体
                        if hasattr(self, 'target_object_indices') and env_idx < len(self.target_object_indices):
                            if i == self.target_object_indices[env_idx]:
                                continue
                        
                        # 计算位移
                        if hasattr(self, 'initial_object_positions') and env_idx < len(self.initial_object_positions):
                            if obj_count < len(self.initial_object_positions[env_idx]):
                                initial_pos = self.initial_object_positions[env_idx][obj_count]
                                current_pos = obj.pose.p
                                displacement += torch.linalg.norm(current_pos - initial_pos)
                                obj_count += 1
            
            total_displacement[env_idx] = displacement
        
        return total_displacement

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """计算密集奖励"""
        reward = torch.zeros(self.num_envs, device=self.device)
        
        if not hasattr(self, 'target_object') or self.target_object is None:
            return reward
        
        # 1. 接近奖励（优先级最高）
        tcp_to_obj_dist = torch.linalg.norm(
            self.target_object.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward * 2.0  # 权重2.0
        
        # 2. 抓取奖励
        is_grasped = info["is_grasped"]
        reward += is_grasped * 3.0  # 权重3.0
        
        # 3. 放置奖励
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.target_object.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped * 2.0  # 只有抓取时才给放置奖励
        
        # 4. 其他物体位移惩罚（优先级第二）
        other_displacement = self._calculate_other_objects_displacement()
        displacement_penalty = torch.tanh(other_displacement)
        reward -= displacement_penalty * 1.5  # 权重1.5
        
        # 5. 时间惩罚（优先级第三）
        time_penalty = 0.01  # 每步小惩罚
        reward -= time_penalty
        
        # 6. 静止奖励
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"] * 1.0
        
        # 7. 成功奖励
        reward[info["success"]] = 10.0
        
        return reward

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """计算稀疏奖励"""
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # 只有成功时才给奖励
        reward[info["success"]] = 1.0
        
        # 其他物体位移惩罚
        other_displacement = self._calculate_other_objects_displacement()
        displacement_penalty = other_displacement * 0.1
        reward -= displacement_penalty
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """计算归一化密集奖励"""
        # 检查是否有 reward_mode 属性，如果没有则默认使用 dense 模式
        if hasattr(self, 'reward_mode') and self.reward_mode == "sparse":
            return self.compute_sparse_reward(obs=obs, action=action, info=info)
        else:
            return self.compute_dense_reward(obs=obs, action=action, info=info) / 10.0 

    def reset(self, seed=None, options=None):
        """重置环境，确保返回一致的观测结构"""
        # 调用父类的reset方法
        obs, info = super().reset(seed=seed, options=options)
        
        # 在离散动作模式下，确保返回一致的观测结构
        if self.use_discrete_action:
            # 获取评估信息
            eval_info = self.evaluate()
            
            # 使用我们的观测处理方法
            obs = self._get_obs_extra(eval_info)
        
        return obs, info 