import os
from typing import Any, Dict, List, Union, Tuple, Optional
import numpy as np
import sapien
import torch
import cv2
import random
import sys

# 导入配置
try:
    from .config import Config, get_config
except ImportError:
    # 处理直接运行时的相对导入问题
    from config import Config, get_config

# 导入AnyGrasp相关模块
try:
    # 添加AnyGrasp路径到系统路径
    anygrasp_path = "/home/linux/jzh/RL_Robot/anygrasp_sdk/grasp_detection"
    if anygrasp_path not in sys.path:
        sys.path.insert(0, anygrasp_path)
    from gsnet import AnyGrasp
    from graspnetAPI import GraspGroup
    ANYGRASP_AVAILABLE = True
except ImportError as e:
    print(f"警告: AnyGrasp未能导入 - {e}")
    print("将跳过抓取点检测功能")
    ANYGRASP_AVAILABLE = False

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
    - 目标物体被成功抓取
    """
    
    SUPPORTED_REWARD_MODES = ["dense", "sparse"]
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    # 托盘参数 (基于traybox.urdf的尺寸)
    tray_size = [0.6, 0.6, 0.15]  # 托盘内部尺寸 (长x宽x高)
    tray_spawn_area = [0.23, 0.23]  # 托盘内物体生成区域 (考虑边界墙和安全边距)
    
    # 注意：物体相关参数现在从config中动态获取
    # BOX_OBJECTS, num_objects_per_type, MAX_N, MAX_EPISODE_STEPS 等
    # 都在 __init__ 方法中从配置文件初始化
    
    # 新增：吸盘约束相关常量
    SUCTION_DISTANCE_THRESHOLD = 0.1  # 吸盘激活距离阈值 
    SUCTION_STIFFNESS = 1e6  # 吸盘约束刚度
    SUCTION_DAMPING = 1e4    # 吸盘约束阻尼
    
    # AnyGrasp相关配置 - 🔧 修复：使用官方demo的高质量参数
    ANYGRASP_CHECKPOINT = "/home/linux/jzh/RL_Robot/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar"  # 模型权重路径
    ANYGRASP_MAX_GRIPPER_WIDTH = 0.1   # 🔧 修复：增加到10cm，与官方demo一致
    ANYGRASP_GRIPPER_HEIGHT = 0.03     # 🔧 修复：减少到3cm，与官方demo一致
    ANYGRASP_TOP_DOWN_GRASP = True     # 是否优先顶部抓取
    
    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        use_discrete_action=False,  # 新增：是否使用离散动作
        use_ideal_oracle=True,      # 新增：是否使用理想化神谕抓取
        config_preset="default",    # 新增：配置预设名称
        custom_config=None,         # 新增：自定义配置对象
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_discrete_action = use_discrete_action
        self.use_ideal_oracle = use_ideal_oracle
        
        # 初始化配置
        if custom_config is not None:
            self.config = custom_config
        else:
            self.config = get_config(config_preset)
        
        # 从配置中获取物体相关参数
        self.BOX_OBJECTS = self.config.env.box_objects
        self.num_objects_per_type = self.config.env.num_objects_per_type
        self.num_object_types = self.config.env.num_object_types
        self.total_objects_per_env = self.config.env.total_objects_per_env
        self.goal_thresh = self.config.env.goal_thresh  # 成功阈值
        
        # 设置动态计算的属性
        self.MAX_N = self.total_objects_per_env  # 最大物体数量
        self.MAX_EPISODE_STEPS = self.config.env.max_episode_steps_discrete  # 最大episode步数
        
        # 初始化离散动作相关变量 - 修改为多环境支持
        self.remaining_indices = []  # 每个环境的剩余可抓取物体索引 [[env0_indices], [env1_indices], ...]
        self.step_count = []  # 每个环境的当前步数 [env0_steps, env1_steps, ...]
        self.grasped_objects = []  # 每个环境已抓取的物体 [[env0_grasped], [env1_grasped], ...]
        
        # 新增：并行有限状态机变量
        self.env_stage = None      # [num_envs] 当前所处状态 0~7
        self.env_target = None     # [num_envs] 每个环境正在处理的物体索引
        self.env_busy = None       # [num_envs] True=流程进行中，False=本回合已结束或等待新指令
        self.stage_tick = None     # [num_envs] 在某状态中已经走了多少微步
        self.stage_positions = None # [num_envs, 3] 每个环境当前状态的目标位置
        
        # 新增：初始化吸盘约束相关变量
        self.suction_constraints = {}  # 存储约束对象的字典 {object_name: constraint}
        self.is_suction_active = [False] * num_envs  # 每个环境的吸盘激活状态
        self.current_suction_object = [None] * num_envs  # 每个环境当前吸附的物体
        
        # 新增：理想化神谕抓取相关变量
        self.oracle_attached_object = [None] * num_envs  # 每个环境当前理想化抓取的物体
        # 理想抓取姿态字典：物体类别 -> 局部坐标系下的理想抓取姿态
        self.ideal_grasp_poses_local = {
            # YCB常见物体的理想抓取姿态（局部坐标系）
            '003_cracker_box': sapien.Pose(p=[0, 0, 0.05], q=[1, 0, 0, 0]),  # 顶抓，上方5cm
            '004_sugar_box': sapien.Pose(p=[0, 0, 0.06], q=[1, 0, 0, 0]),   # 顶抓，上方6cm  
            '005_tomato_soup_can': sapien.Pose(p=[0, 0, 0.04], q=[1, 0, 0, 0]),  # 顶抓，上方4cm
            '006_mustard_bottle': sapien.Pose(p=[0, 0, 0.05], q=[1, 0, 0, 0]),   # 顶抓，上方5cm
            '009_gelatin_box': sapien.Pose(p=[0, 0, 0.04], q=[1, 0, 0, 0]),      # 顶抓，上方4cm
            'default': sapien.Pose(p=[0, 0, 0.05], q=[1, 0, 0, 0])  # 默认：顶抓，朝下，上方5cm
        }
        
        # 初始化AnyGrasp（只初始化一次）
        self.anygrasp_model = None
        self.anygrasp_enabled = ANYGRASP_AVAILABLE
        if self.anygrasp_enabled:
            self._init_anygrasp()
        
        # 确保所有参数正确传递给父类
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            **kwargs,
        )
        
        # 在父类初始化后初始化FSM状态张量
        if self.use_discrete_action:
            self._init_fsm_states()

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
        # 🔧 平衡相机位置：既要获得合理的深度值，又要保证AnyGrasp能检测到抓取
        # 托盘位置在 [-0.2, 0.0, 0.006]，物体在z=0.04-0.1左右
        # 使用适中的相机高度和距离，斜向俯视角度有利于抓取检测
        pose = sapien_utils.look_at(eye=[0.0, 0, 0.35], target=[-0.2, 0.0, 0.05])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=640,   # 🔧 修复：提高分辨率到640x480，接近官方demo
                height=480,
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
        
    # 吸盘约束系统实现
    def _create_suction_constraint(self, target_object: Actor, env_idx: int = 0) -> bool:
        """
        创建吸盘约束
        
        Args:
            target_object: 目标物体
            env_idx: 环境索引（多环境支持）
            
        Returns:
            bool: 是否成功创建约束
        """
        if self.is_suction_active[env_idx]:
            print(f"环境{env_idx}: 吸盘已经激活，无法创建新约束")
            return False
            
        # 检查是否与物体接触
        if not self._is_contacting_object(target_object, self.SUCTION_DISTANCE_THRESHOLD, env_idx):
            print(f"环境{env_idx}: 物体距离过远，无法激活吸盘")
            return False
        
        try:
            # 导入Drive类
            from mani_skill.utils.structs.drive import Drive
            
            print(f"环境{env_idx}: 创建吸盘约束: TCP链接 -> 物体 {target_object.name}")
            
            # 🔧 修复：安全的多环境对象选择
            # 1. 验证目标物体的环境归属
            target_scene_idxs = target_object._scene_idxs
            if len(target_scene_idxs) == 0:
                print(f"环境{env_idx}: 目标物体没有场景索引")
                return False
            
            target_env_idx = target_scene_idxs[0].item()
            print(f"环境{env_idx}: 目标物体实际属于环境{target_env_idx}")
            
            # 验证环境索引一致性
            if target_env_idx != env_idx:
                print(f"环境{env_idx}: 环境索引不匹配，目标物体属于环境{target_env_idx}")
                return False
            
            # 2. 🔧 修复：通过scene_idxs安全获取TCP实体
            tcp_objs = self.agent.tcp._objs
            tcp_scene_idxs = self.agent.tcp._scene_idxs
            
            # 找到属于target_env_idx环境的TCP对象
            tcp_mask = (tcp_scene_idxs == target_env_idx)
            if not tcp_mask.any():
                print(f"环境{env_idx}: 找不到对应环境的TCP对象")
                return False
            
            tcp_indices = torch.where(tcp_mask)[0]
            if len(tcp_indices) == 0:
                print(f"环境{env_idx}: TCP索引列表为空")
                return False
                
            tcp_idx = tcp_indices[0].item()  # 获取第一个匹配的索引
            tcp_entity = tcp_objs[tcp_idx].entity
            print(f"环境{env_idx}: 找到TCP对象，索引={tcp_idx}，环境={target_env_idx}")
            
            # 3. 🔧 修复：安全获取目标物体实体
            if len(target_object._objs) == 0:
                print(f"环境{env_idx}: 目标物体没有实体对象")
                return False
            
            # 在当前设计中，每个物体通常只有一个实体
            target_entity = target_object._objs[0]
            print(f"环境{env_idx}: 目标物体实体数量={len(target_object._objs)}")
            
            print(f"环境{env_idx}: 使用TCP实体[索引{tcp_idx}]和目标实体创建约束")
            
            # 关键修复：直接使用SAPIEN的create_drive方法，绕过Drive包装器的批量处理
            # 这样可以避免scene_idxs和bodies索引不匹配的问题
            sub_scene = self.scene.sub_scenes[target_env_idx]
            physx_drive = sub_scene.create_drive(
                tcp_entity,           # TCP实体
                sapien.Pose(),        # 父体本地姿态 - 修复：直接使用sapien.Pose()
                target_entity,        # 目标物体实体
                sapien.Pose()         # 子体本地姿态 - 修复：直接使用sapien.Pose()
            )

            # 手动创建Drive包装器以便后续管理
            constraint = Drive(
                _objs=[physx_drive],
                _scene_idxs=torch.tensor([target_env_idx], device=self.device),
                pose_in_child=sapien.Pose(),
                pose_in_parent=sapien.Pose(),
                scene=self.scene
            )
            
            # 设置约束参数使其表现为固定约束（类似PyBullet的JOINT_FIXED）
            # 直接调用底层PhysxDriveComponent的方法（这些方法没有@before_gpu_init限制）
            try:
                # 线性约束（X, Y, Z方向）
                physx_drive.set_drive_property_x(stiffness=self.SUCTION_STIFFNESS, damping=self.SUCTION_DAMPING)
                physx_drive.set_drive_property_y(stiffness=self.SUCTION_STIFFNESS, damping=self.SUCTION_DAMPING)
                physx_drive.set_drive_property_z(stiffness=self.SUCTION_STIFFNESS, damping=self.SUCTION_DAMPING)
                print(f"环境{env_idx}: ✅ 已设置驱动属性")
            except Exception as drive_error:
                print(f"环境{env_idx}: ❌ 设置驱动属性失败: {drive_error}")
                return False
            
            # 设置位置限制来模拟固定约束
            try:
                physx_drive.set_limit_x(0, 0)  # 不允许X方向移动
                physx_drive.set_limit_y(0, 0)  # 不允许Y方向移动
                physx_drive.set_limit_z(0, 0)  # 不允许Z方向移动
                print(f"环境{env_idx}: ✅ 已设置位置限制")
            except Exception as limit_error:
                print(f"环境{env_idx}: ⚠️ 设置限制失败: {limit_error}")
                # 继续执行，仅使用驱动属性
            
            # 存储约束 - 使用环境特定的键
            constraint_key = f"{target_object.name}_env_{env_idx}"
            self.suction_constraints[constraint_key] = constraint
            self.is_suction_active[env_idx] = True
            self.current_suction_object[env_idx] = target_object
            
            print(f"环境{env_idx}: ✅ 吸盘约束创建成功: {constraint_key}")
            return True
            
        except Exception as e:
            print(f"环境{env_idx}: ❌ 创建吸盘约束失败: {e}")
            import traceback
            print(f"环境{env_idx}: 详细错误信息:")
            traceback.print_exc()
            return False

    def _remove_suction_constraint(self, env_idx: int = 0) -> bool:
        """
        移除吸盘约束
        
        Args:
            env_idx: 环境索引（多环境支持）
        
        Returns:
            bool: 是否成功移除约束
        """
        if not self.is_suction_active[env_idx] or self.current_suction_object[env_idx] is None:
            #print("没有激活的吸盘约束需要移除")
            return False
        
        try:
            # 获取约束对象 - 使用环境特定的键
            constraint_key = f"{self.current_suction_object[env_idx].name}_env_{env_idx}"
            if constraint_key in self.suction_constraints:
                constraint = self.suction_constraints[constraint_key]
                
                print(f"环境{env_idx}: 正在移除吸盘约束: {constraint_key}")
                
                # 关键修复：直接操作底层PhysxDriveComponent对象
                physx_drive = constraint._objs[0]  # 获取底层的PhysxDriveComponent对象
                
                # 方法1: 通过设置刚度为0来禁用约束（最有效）
                try:
                    print(f"环境{env_idx}: 设置约束刚度为0...")
                    physx_drive.set_drive_property_x(stiffness=0.0, damping=0.0)
                    physx_drive.set_drive_property_y(stiffness=0.0, damping=0.0)
                    physx_drive.set_drive_property_z(stiffness=0.0, damping=0.0)
                    print(f"环境{env_idx}: ✅ 成功禁用约束驱动属性")
                except Exception as disable_error:
                    print(f"环境{env_idx}: ❌ 禁用约束驱动属性失败: {disable_error}")
                    return False
                
                # 方法2: 重置约束限制（辅助方法）
                try:
                    print(f"环境{env_idx}: 重置约束限制...")
                    # 设置非常大的限制范围，相当于取消限制
                    physx_drive.set_limit_x(-1000, 1000)
                    physx_drive.set_limit_y(-1000, 1000)
                    physx_drive.set_limit_z(-1000, 1000)
                    print(f"环境{env_idx}: ✅ 成功重置约束限制")
                except Exception as limit_error:
                    print(f"环境{env_idx}: ⚠️ 重置约束限制失败: {limit_error}")
                    # 限制重置失败不影响主要功能，继续执行
                    pass
                
                # 清理约束引用
                del self.suction_constraints[constraint_key]
                print(f"环境{env_idx}: ✅ 约束引用已清理: {constraint_key}")
            else:
                print(f"环境{env_idx}: ⚠️ 未找到约束对象: {constraint_key}")
                pass
            
            # 删除被吸取的物体
            try:
                if self.current_suction_object[env_idx] is not None:
                    target_obj = self.current_suction_object[env_idx]
                    
                    # 检查是否为GPU仿真
                    if self.scene.gpu_sim_enabled:
                        # GPU仿真：将物体移动到远处（模拟删除效果）
                        target_obj.set_pose(Pose.create_from_pq(p=[100.0, 100.0, 100.0]))
                        print(f"环境{env_idx}: ✅ GPU仿真模式 - 将物体{target_obj.name}移动到远处")
                    else:
                        # CPU仿真：物理删除物体
                        target_obj.remove_from_scene()
                        print(f"环境{env_idx}: ✅ CPU仿真模式 - 已从场景中删除物体{target_obj.name}")
                        
            except Exception as remove_error:
                print(f"环境{env_idx}: ⚠️ 删除物体失败: {remove_error}")
                # 删除失败不影响主要流程，继续执行
                pass
            
            # 重置吸盘状态
            self.is_suction_active[env_idx] = False
            self.current_suction_object[env_idx] = None
            
            print(f"环境{env_idx}: ✅ 吸盘状态已重置")
            
            return True
            
        except Exception as e:
            print(f"环境{env_idx}: ❌ 移除吸盘约束失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 即使移除失败，也要重置状态
            self.is_suction_active[env_idx] = False
            self.current_suction_object[env_idx] = None
            return False

    def _is_contacting_object(self, target_object: Actor, threshold: float = 0.05, env_idx: int = 0) -> bool:
        """
        检测TCP是否与物体接触
        
        Args:
            target_object: 目标物体
            threshold: 距离阈值
            env_idx: 环境索引（多环境支持）
            
        Returns:
            bool: 是否接触
        """
        try:
            # 计算TCP到物体的距离 - 使用对应环境的TCP位置
            tcp_pos = self.agent.tcp.pose.p
            if tcp_pos.dim() > 1:
                if env_idx < tcp_pos.shape[0]:
                    tcp_pos = tcp_pos[env_idx]
                else:
                    tcp_pos = tcp_pos[0]
                    print(f"⚠️ 环境{env_idx}: TCP位置索引越界，使用环境0的位置")
            
            # 正确获取多环境下的物体位置
            obj_pos = target_object.pose.p
            obj_pos = obj_pos[0]
            
            # 计算距离
            raw_distance = torch.linalg.norm(tcp_pos - obj_pos).item()
            # 🔧 修复：使用更合理的半径估计值
            # TCP半径约2cm，物体平均半径约3cm，总计约5cm
            estimated_radius = 0.05  # 5cm的半径估计，与_check_suction_grasp_success保持一致
            distance = raw_distance - estimated_radius
            
            print(f"环境{env_idx}: TCP到物体距离检测: 原始距离={raw_distance:.4f}m, 调整后距离={distance:.4f}m, 阈值={threshold:.4f}m, 接触={'是' if distance <= threshold else '否'}")
            
            # 检查是否在接触阈值内
            return distance <= threshold
            
        except Exception as e:
            print(f"环境{env_idx}: 检测接触失败: {e}")
            return False

    def _check_suction_grasp_success(self, target_object: Actor, env_idx: int = 0) -> bool:
        """
        检查吸盘抓取是否成功
        
        Args:
            target_object: 目标物体
            env_idx: 环境索引（多环境支持）
            
        Returns:
            bool: 抓取是否成功
        """
        try:
            # 方法1：检查吸盘状态
            if (self.is_suction_active[env_idx] and 
                self.current_suction_object[env_idx] is not None and 
                self.current_suction_object[env_idx].name == target_object.name):
                
                # 方法2：检查物体是否仍在TCP附近
                tcp_pos = self.agent.tcp.pose.p
                if tcp_pos.dim() > 1:
                    if env_idx < tcp_pos.shape[0]:
                        tcp_pos = tcp_pos[env_idx]
                    else:
                        tcp_pos = tcp_pos[0]
                        print(f"⚠️ 环境{env_idx}: TCP位置索引越界，使用环境0的位置")
                
                obj_pos = target_object.pose.p
                obj_pos = obj_pos[0]
                
                raw_distance = torch.linalg.norm(tcp_pos - obj_pos).item()
                # 🔧 修复：使用与接触检测一致的半径估计
                estimated_radius = 0.05  # 5cm的半径估计，与_is_contacting_object保持一致
                distance = raw_distance - estimated_radius
                
                # 距离小于5cm认为抓取成功
                success_threshold = 0.05
                success = distance < success_threshold
                
                print(f"环境{env_idx}: 抓取成功检测 - 原始距离={raw_distance:.4f}m, 调整后距离={distance:.4f}m, 成功={'是' if success else '否'}")
                
                return success
            else:
                print(f"环境{env_idx}: 吸盘未激活或物体不匹配")
                return False
                
        except Exception as e:
            print(f"环境{env_idx}: 检查吸盘抓取成功失败: {e}")
            return False
    
    
    
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

    def _is_path_occluded_by_geometry(self, target_obj) -> bool:
        """
        几何遮挡检查：检查预抓取路径是否被阻挡
        优先使用现有的 _is_object_blocked 作为兜底
        
        Args:
            target_obj: 目标物体
            
        Returns:
            bool: True表示被遮挡，False表示路径畅通
        """
        try:
            # 优先使用现有的简化遮挡检测
            return self._is_object_blocked(target_obj)
            
        except Exception as e:
            print(f"遮挡检查失败: {e}")
            return True  # 出错时保守返回被遮挡

    def _is_supporting_others(self, target_obj) -> bool:
        """
        支撑检查：检查目标物体是否正在支撑其他物体
        使用ManiSkill物理查询检测接触力和位置关系
        
        Args:
            target_obj: 目标物体
            
        Returns:
            bool: True表示正在支撑其他物体，False表示没有支撑关系
        """
        try:
            target_pos = target_obj.pose.p
            if target_pos.dim() > 1:
                target_pos = target_pos[0]
            
            # 检查与所有其他物体的接触关系
            for other_obj in self.all_objects:
                if other_obj == target_obj:
                    continue
                
                other_pos = other_obj.pose.p
                if other_pos.dim() > 1:
                    other_pos = other_pos[0]
                
                # 首先检查位置关系：其他物体是否在目标物体上方
                if other_pos[2] <= target_pos[2]:
                    continue  # 其他物体不在上方，跳过
                
                try:
                    # 使用ManiSkill的接触力查询
                    contact_forces = self.scene.get_pairwise_contact_forces(target_obj, other_obj)
                    
                    if contact_forces is not None:
                        # 检查是否有足够的接触力
                        force_magnitude = torch.linalg.norm(contact_forces, dim=-1)
                        
                        # 如果有有效的接触力（阈值设为0.1N）
                        if torch.any(force_magnitude > 0.1):
                            # 检查力的方向：向下的力表明目标物体在支撑其他物体
                            if contact_forces.dim() > 1:
                                # 取第一个环境的力向量
                                force_vec = contact_forces[0]
                            else:
                                force_vec = contact_forces
                            
                            # 检查z方向的力：如果力向下（负z方向），说明目标物体在支撑
                            if len(force_vec) >= 3 and force_vec[2] < -0.05:  # 阈值-0.05N
                                return True
                                
                except Exception as contact_error:
                    # 接触查询失败，使用几何位置作为兜底判据
                    # 如果其他物体在目标物体正上方很近的距离（<3cm），认为有支撑关系
                    xy_distance = torch.linalg.norm(other_pos[:2] - target_pos[:2])
                    z_distance = other_pos[2] - target_pos[2]
                    if xy_distance < 0.08 and z_distance < 0.03:  # xy距离<8cm, z距离<3cm
                        return True
            
            return False
            
        except Exception as e:
            print(f"支撑检查失败: {e}")
            return True  # 出错时保守返回正在支撑

    def _get_ideal_world_grasp_pose(self, target_obj):
        """
        计算理想的世界坐标系抓取位姿
        从局部坐标系的理想抓取姿态转换到世界坐标系
        
        Args:
            target_obj: 目标物体
            
        Returns:
            tuple: (grasp_pose, pre_grasp_pose) - 抓取位姿和预抓取位姿
        """
        try:
            # 获取物体名称，尝试匹配已知的理想抓取姿态
            obj_name = target_obj.name
            local_grasp_pose = None
            
            # 尝试从已知物体类型中查找
            for key in self.ideal_grasp_poses_local:
                if key in obj_name:
                    local_grasp_pose = self.ideal_grasp_poses_local[key]
                    break
            
            # 如果未找到，使用默认抓取姿态
            if local_grasp_pose is None:
                local_grasp_pose = self.ideal_grasp_poses_local['default']
            
            # 获取目标物体的世界位姿
            target_world_pose = target_obj.pose
            if target_world_pose.p.dim() > 1:
                # 多环境情况，取第一个环境的姿态
                target_world_pose = sapien.Pose(
                    p=target_world_pose.p[0].cpu().numpy(),
                    q=target_world_pose.q[0].cpu().numpy()
                )
            
            # 🔧 关键修复：检查物体位置是否合理（避免瞬移后的异常坐标）
            obj_pos = target_world_pose.p
            pos_magnitude = (obj_pos[0]**2 + obj_pos[1]**2 + obj_pos[2]**2)**0.5
            
            # 如果物体距离原点超过3米，认为是已瞬移的物体，跳过抓取
            if pos_magnitude > 3.0:
                print(f"⚠️ 物体 {target_obj.name} 位置异常 ({obj_pos})，可能已被瞬移，跳过抓取")
                # 返回一个安全的工作区内位置
                safe_pose = sapien.Pose(p=[0.0, 0.0, 0.5], q=[1, 0, 0, 0])
                safe_pre_pose = sapien.Pose(p=[0.0, 0.0, 0.58], q=[1, 0, 0, 0])
                return safe_pose, safe_pre_pose
            
            # 计算世界坐标系下的抓取位姿：world_pose * local_pose
            world_grasp_pose = target_world_pose * local_grasp_pose
            
            # 计算预抓取位姿（在抓取位姿上方安全距离）
            grasp_p = world_grasp_pose.p
            grasp_q = world_grasp_pose.q
            # 确保转换为numpy数组
            if hasattr(grasp_p, 'cpu'):
                grasp_p = grasp_p.cpu().numpy()
            if hasattr(grasp_q, 'cpu'):
                grasp_q = grasp_q.cpu().numpy()
            
            pre_grasp_pose = sapien.Pose(
                p=[grasp_p[0], grasp_p[1], grasp_p[2] + 0.08],  # 上方8cm
                q=grasp_q
            )
            
            return world_grasp_pose, pre_grasp_pose
            
        except Exception as e:
            print(f"计算理想抓取位姿失败: {e}")
            # 失败时返回目标物体上方的简单位姿
            obj_pos = target_obj.pose.p
            if obj_pos.dim() > 1:
                obj_pos = obj_pos[0]
            
            # 简单的顶抓位姿（向下）
            # 确保obj_pos转换为numpy数组
            if hasattr(obj_pos, 'cpu'):
                obj_pos = obj_pos.cpu().numpy()
            
            grasp_pose = sapien.Pose(
                p=[obj_pos[0], obj_pos[1], obj_pos[2] + 0.05],
                q=[1, 0, 0, 0]  # 朝下
            )
            pre_grasp_pose = sapien.Pose(
                p=[obj_pos[0], obj_pos[1], obj_pos[2] + 0.13],
                q=[1, 0, 0, 0]  # 朝下
            )
            
            return grasp_pose, pre_grasp_pose


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
        tray_urdf_path = "/home/linux/jzh/RL_Robot/assets/tray/traybox.urdf"
        
        if not os.path.exists(tray_urdf_path):
            raise FileNotFoundError(f"托盘URDF文件未找到: {tray_urdf_path}")
        
        # 创建URDF加载器
        loader = self.scene.create_urdf_loader()
        
        # 设置托盘的物理属性
        loader.set_material(static_friction=0.8, dynamic_friction=0.6, restitution=0.05)
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
            tray_position = [-0.2, 0.0, 0.006]  # 桌面高度加上托盘底部厚度
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
        tray_bottom_z = 0.02 + 0.02  # 托盘底部 + 小偏移
        
        # 托盘边界计算（基于URDF文件中的边界墙位置）
        # 边界墙在托盘中心的±0.2米处
        # 实际可用空间：从中心向两边各0.18米（留出安全边距）
        safe_spawn_area_x = 0.18
        safe_spawn_area_y = 0.18
        
        # 在托盘内随机生成xy位置
        x = tray_center_x + random.uniform(-safe_spawn_area_x, safe_spawn_area_x)
        y = tray_center_y + random.uniform(-safe_spawn_area_y, safe_spawn_area_y)
        
        # 堆叠高度
        z = tray_bottom_z + stack_level * 0.04  # 每层高度
        
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
            
            # 重新选择目标物体 - 只在连续动作模式下使用
            if not self.use_discrete_action:
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
                
                # 新增：重置FSM状态
                if hasattr(self, 'env_stage') and self.env_stage is not None:
                    if b == self.num_envs:
                        # 重置所有环境
                        self.env_stage.fill_(0)
                        self.env_target.fill_(-1)
                        self.env_busy.fill_(False)
                        self.stage_tick.fill_(0)
                        self.stage_positions.fill_(0)
                    else:
                        # 重置指定环境
                        for i, env_id in enumerate(env_idx):
                            env_id_int = env_id.item() if hasattr(env_id, 'item') else int(env_id)
                            if env_id_int < self.num_envs:
                                self.env_stage[env_id_int] = 0
                                self.env_target[env_id_int] = -1
                                self.env_busy[env_id_int] = False
                                self.stage_tick[env_id_int] = 0
                                self.stage_positions[env_id_int].fill_(0)
            
            # 新增：重置吸盘约束状态
            self.suction_constraints = {}
            self.is_suction_active = [False] * self.num_envs  # 每个环境的吸盘激活状态
            self.current_suction_object = [None] * self.num_envs  # 每个环境当前吸附的物体
            
            # 新增：重置理想化神谕状态
            self.oracle_attached_object = [None] * self.num_envs  # 清空理想化抓取连接
            if hasattr(self, 'grasp_poses'):
                self.grasp_poses = [None] * self.num_envs  # 清空抓取位姿缓存
            
            
            # 使用指定的机器人初始姿态重置
            # 指定的关节位置：[-1.6137, 1.3258, 1.9346, -0.8884, -1.6172, 1.0867, -3.0494, 0.04, 0.04]
            #target_qpos = np.array([-0.5370, 1.3258, 1.9346, -0.8884, -1.6172, 1.0867, -3.0494, 0.04, 0.04])
            #target_qpos = np.array([-1.6137, 1.3258, 1.9346, -0.8884, -1.6172, 1.0867, -3.0494, 0.04, 0.04])
            target_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
            # qpos=np.array(
            #     [
            #         0.0,
            #         np.pi / 8,
            #         0,
            #         -np.pi * 5 / 8,
            #         0,
            #         np.pi * 3 / 4,
            #         np.pi / 4,
            #         0.04,
            #         0.04,
            #     ]
            # )
            # 重置机器人到指定姿态
            self.agent.reset(target_qpos)
            #self.agent.reset()
            
            # 🔧 关键修复：初始化TCP位置记录数组（稍后在动作执行时记录实际位置）
            if not hasattr(self, 'initial_tcp_positions'):
                self.initial_tcp_positions = [None] * self.num_envs
            if not hasattr(self, 'tcp_recorded'):
                self.tcp_recorded = [False] * self.num_envs
            
            # 重置记录状态
            for i in env_idx:
                if i < self.num_envs:
                    self.initial_tcp_positions[i] = None
                    self.tcp_recorded[i] = False

    def _get_obs_extra(self, info: Dict):
        """获取额外观测信息"""
        # 获取批次大小
        batch_size = self.num_envs
        
        if not self.use_discrete_action:
            # 连续动作模式：保持原有观测结构
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
                    zero_pose = torch.zeros((batch_size, 7), device=self.device)
                    zero_pos = torch.zeros((batch_size, 3), device=self.device)
                    obs.update(
                        target_obj_pose=zero_pose,
                        tcp_to_obj_pos=zero_pos,
                        obj_to_goal_pos=zero_pos,
                    )
                
                obs.update(
                    num_objects=torch.tensor([len(self.all_objects)], device=self.device).repeat(batch_size),
                )
            return obs
        
        # 离散动作模式：易收敛的baseline特征集
        
        # 1. 全局特征 (每环境 3 维)
        global_features = []
        for env_idx in range(batch_size):
            grasped_count = len(self.grasped_objects[env_idx])
            grasped_ratio = grasped_count / float(self.total_objects_per_env)  # 已抓数量/总数量
            
            # 使用抓取尝试次数的比例作为特征
            attempt_ratio = min(self.step_count[env_idx] / float(self.total_objects_per_env), 1.0)  # 抓取尝试次数/总数量，限制在[0,1]
            remaining_ratio = (self.total_objects_per_env - grasped_count) / float(self.total_objects_per_env)  # 剩余数量/总数量
            
            global_features.append([grasped_ratio, attempt_ratio, remaining_ratio])
        
        global_features = torch.tensor(global_features, device=self.device, dtype=torch.float32)  # [batch_size, 3]
        
        # 2. 每物体特征 
        # 获取工作空间范围用于归一化
        workspace_min = torch.tensor([-0.5, -0.5, 0.0], device=self.device)  # 工作空间最小值
        workspace_max = torch.tensor([0.5, 0.5, 0.3], device=self.device)     # 工作空间最大值
        workspace_size = workspace_max - workspace_min
        
        # 获取最大物体尺寸用于归一化
        max_size = 0.2  # 假设最大物体尺寸为0.2m
        
        object_features = []  # [batch_size, 8, 8]
        action_mask = []      # [batch_size, 8]
        
        for env_idx in range(batch_size):
            env_obj_features = []
            env_mask = []
            
            for obj_idx in range(self.total_objects_per_env):  # 动态物体数量
                # 🔧 修复：使用环境特定的物体列表而不是全局索引
                if (env_idx < len(self.selectable_objects) and 
                    obj_idx < len(self.selectable_objects[env_idx])):
                    
                    # 获取环境特定的物体
                    target_obj = self.selectable_objects[env_idx][obj_idx]
                    obj_pose_p = target_obj.pose.p
                    
                    # 处理多环境位置数据
                    if len(obj_pose_p.shape) > 1 and obj_pose_p.shape[0] > env_idx:
                        obj_pos = obj_pose_p[env_idx]  # [3]
                    elif len(obj_pose_p.shape) > 1 and obj_pose_p.shape[0] == 1:
                        obj_pos = obj_pose_p[0]
                    else:
                        obj_pos = obj_pose_p
                    
                    # 位置归一化到 [0, 1]
                    pos_normalized = (obj_pos - workspace_min) / workspace_size
                    pos_normalized = torch.clamp(pos_normalized, 0.0, 1.0)
                    
                    # 获取物体尺寸并归一化
                    obj_type_idx = obj_idx // self.num_objects_per_type
                    if obj_type_idx < len(self.BOX_OBJECTS):
                        obj_type = self.BOX_OBJECTS[obj_type_idx]
                        size = self._get_object_size(obj_type)
                        obj_size = torch.tensor(size, device=self.device)
                    else:
                        obj_size = torch.tensor([0.05, 0.05, 0.05], device=self.device)
                    
                    # 尺寸归一化到 [0, 1]
                    size_normalized = obj_size / max_size
                    size_normalized = torch.clamp(size_normalized, 0.0, 1.0)
                    
                    # 抓取标志
                    grabbed_flag = 1.0 if obj_idx in self.grasped_objects[env_idx] else 0.0
                    
                    # 高度特征 (可选，如果已包含在pos_normalized中可以去掉)
                    topness = pos_normalized[2]  # z坐标已经归一化
                    
                    # 组合特征: [size_x, size_y, size_z, pos_x, pos_y, pos_z, grabbed_flag, topness]
                    obj_feature = torch.cat([
                        size_normalized,      # [3] - 尺寸
                        pos_normalized,       # [3] - 位置
                        torch.tensor([grabbed_flag], device=self.device),  # [1] - 抓取标志
                        torch.tensor([topness], device=self.device)        # [1] - 高度特征
                    ])  # 总共8维
                    
                    # 动作掩码：未抓取=1(可选)，已抓取=0(不可选)
                    mask_value = 0.0 if grabbed_flag > 0.5 else 1.0
                    
                else:
                    # 填充零特征
                    obj_feature = torch.zeros(8, device=self.device)
                    mask_value = 0.0  # 不存在的物体不可选
                
                env_obj_features.append(obj_feature)
                env_mask.append(mask_value)
            
            object_features.append(torch.stack(env_obj_features))  # [8, 8]
            action_mask.append(torch.tensor(env_mask, device=self.device))  # [8]
        
        object_features = torch.stack(object_features)  # [batch_size, 8, 8]
        action_mask = torch.stack(action_mask)          # [batch_size, 8]
        
        # 3. 展平物体特征
        object_features_flat = object_features.view(batch_size, -1)  # [batch_size, total_objects_per_env * 8]
        
        # 4. 组合最终观测
        # obs = concat(obj_feats.flatten(), action_mask, global_feats)
        final_obs = torch.cat([
            object_features_flat,  # [batch_size, total_objects_per_env * 8] - 物体特征
            action_mask,          # [batch_size, total_objects_per_env]  - 动作掩码
            global_features       # [batch_size, 3]  - 全局特征
        ], dim=1)  # [batch_size, total_objects_per_env * 9 + 3]
        
        # 返回扁平化的观测向量，符合baseline训练要求
        return final_obs

    
    def _set_gripper_target(self, target_width: float, env_idx: int = None) -> torch.Tensor:
        """
        设置夹爪目标宽度，参考panda风格的夹爪控制
        
        Args:
            target_width: 夹爪目标宽度 (0.0=闭合, 0.04=完全打开)
            env_idx: 环境索引，None表示所有环境
            
        Returns:
            action: 7维动作向量，只设置夹爪部分
        """
        # 限制夹爪宽度在有效范围内（参考panda.py的配置）
        target_width = max(0.0, min(0.04, target_width))
        
        # 构建7维动作向量 [dx, dy, dz, drx, dry, drz, gripper]
        if env_idx is not None:
            # 单个环境
            action = torch.zeros(7, device=self.device, dtype=torch.float32)
            action[6] = target_width
        else:
            # 所有环境
            action = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float32)
            action[:, 6] = target_width
        
        return action

    def _is_grasping_object(self, target_obj, env_idx: int = 0, min_force: float = 0.5) -> bool:
        """
        检查是否成功抓取物体，参考panda.py的is_grasping方法
        在理想化模式中主要用于调试和验证
        
        Args:
            target_obj: 目标物体
            env_idx: 环境索引
            min_force: 最小接触力阈值
            
        Returns:
            bool: 是否成功抓取
        """
        try:
            # 获取机械臂的finger链接（假设与panda结构相同）
            finger_links = []
            for link in self.agent.robot.get_links():
                if 'finger' in link.name:
                    finger_links.append(link)
            
            if len(finger_links) < 2:
                # 如果找不到finger链接，退回到简单的距离检查
                tcp_pos = self.agent.tcp.pose.p
                if tcp_pos.dim() > 1:
                    tcp_pos = tcp_pos[env_idx]
                
                obj_pos = target_obj.pose.p
                if obj_pos.dim() > 1:
                    obj_pos = obj_pos[env_idx]
                
                distance = torch.linalg.norm(tcp_pos - obj_pos).item()
                return distance < 0.1  # 10cm以内认为抓取成功
            
            # 使用接触力检查（参考panda.py实现）
            total_force = 0.0
            for finger_link in finger_links:
                contact_forces = self.scene.get_pairwise_contact_forces(finger_link, target_obj)
                if contact_forces is not None:
                    force_magnitude = torch.linalg.norm(contact_forces, dim=-1)
                    if force_magnitude.dim() > 0:
                        total_force += force_magnitude[env_idx].item()
                    else:
                        total_force += force_magnitude.item()
            
            return total_force >= min_force
            
        except Exception as e:
            # 出错时使用距离检查作为兜底
            tcp_pos = self.agent.tcp.pose.p
            if tcp_pos.dim() > 1:
                tcp_pos = tcp_pos[env_idx]
            
            obj_pos = target_obj.pose.p
            if obj_pos.dim() > 1:
                obj_pos = obj_pos[env_idx]
            
            distance = torch.linalg.norm(tcp_pos - obj_pos).item()
            return distance < 0.1

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
        处理离散动作的step方法 - 使用while循环管理并行FSM
        
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
        
        # 1. 为空闲环境分配新任务
        for i in range(self.num_envs):
            if not self.env_busy[i]:
                pick = int(action[i])
                if pick >= 0 and pick < len(self.remaining_indices[i]):
                    # 获取实际的物体索引
                    target_idx = self.remaining_indices[i][pick]
                    self.env_target[i] = target_idx
                    self.remaining_indices[i].pop(pick)
                    self.env_stage[i] = 0
                    self.env_busy[i] = True
                    self.stage_tick[i] = 0
                    self.step_count[i] += 1
                    #print(f"环境{i}: 开始新任务 - 抓取物体索引{target_idx} (选择{pick})")
        
        # 2. while循环执行FSM，直到所有环境完成任务
        # 🔧 关键修复：基于阶段调整超时限制，避免复杂动作被过早截断
        max_fsm_steps = 1500  # 适度降低但仍足够完成完整流程
        fsm_step_count = 0
        consecutive_stuck_steps = 0  # 连续卡住步数计数器
        last_busy_count = torch.sum(self.env_busy).item()
        
        while torch.any(self.env_busy) and fsm_step_count < max_fsm_steps:
            # 为所有环境计算当前FSM状态对应的低级连续动作
            cmd = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float32)
            active_envs = 0
            
            for i in range(self.num_envs):
                if self.env_busy[i]:
                    cmd[i] = self._pick_object_step(i)
                    active_envs += 1
            
            # 批处理执行一步物理仿真 - 所有环境同步前进
            super().step(cmd)
            fsm_step_count += 1
            
            # 🔧 检测是否有进展（避免无限循环）
            current_busy_count = torch.sum(self.env_busy).item()
            if current_busy_count == last_busy_count:
                consecutive_stuck_steps += 1
            else:
                consecutive_stuck_steps = 0
                last_busy_count = current_busy_count
            
            # 如果连续很多步没有进展，提前结束避免卡死
            if consecutive_stuck_steps > 500:
                print(f"⚠️ 检测到FSM可能卡死（连续{consecutive_stuck_steps}步无进展），强制结束")
                break
            
            # 每300步输出一次进度信息（减少输出频率但提供足够反馈）
            if fsm_step_count % 300 == 0:
                print(f"FSM步骤 {fsm_step_count}: 仍有 {current_busy_count} 个环境在执行任务")
        
        # 检查是否因为超时而退出循环
        if fsm_step_count >= max_fsm_steps or consecutive_stuck_steps > 500:
            reason = "超时" if fsm_step_count >= max_fsm_steps else "无进展"
            print(f"⚠️ FSM执行因{reason}而终止 (步数: {fsm_step_count})，强制结束所有任务")
            # 🔧 优雅地强制结束所有任务并清理状态
            self._force_terminate_all_tasks()
        
        completed_envs = torch.sum(~self.env_busy).item()
        # 提供更详细的完成信息
        if completed_envs == self.num_envs:
            pass  # 成功完成时不输出，减少干扰
        else:
            active_envs = self.num_envs - completed_envs
            print(f"⚠️ FSM执行结束 - 总步数: {fsm_step_count}, 完成{completed_envs}个，剩余{active_envs}个")
        
        # 3. 清理所有环境状态，确保下一轮能正常开始
        self._cleanup_completed_tasks()
        
        # 4. 计算最终奖励和观测 - 基于完整动作的最终结果
        info = self.get_info()
        obs = self.get_obs(info)
        reward = self.get_reward(obs=obs, action=action, info=info)
        
        # 5. 检查终止条件
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 使用 info 中的 success 和 fail 状态
        if "success" in info and "fail" in info:
            terminated = torch.logical_or(info["success"], info["fail"])
        elif "success" in info:
            terminated = info["success"].clone()
        elif "fail" in info:
            terminated = info["fail"].clone()
        
        return obs, reward, terminated, truncated, info
    
    def _pick_object_step(self, env_idx: int) -> torch.Tensor:
        """
        理想化抓取状态机 - 每次调用只执行当前状态的一小步
        使用神谕逻辑：遮挡/支撑检查通过则100%成功，否则100%失败
        
        Args:
            env_idx: 环境索引
            
        Returns:
            action: 该环境的连续动作向量 [dx, dy, dz, drx, dry, drz, gripper]
        """
        stage = self.env_stage[env_idx].item()
        target_idx = self.env_target[env_idx].item()
        tick = self.stage_tick[env_idx].item()
        
        # 初始化动作向量
        action = torch.zeros(7, device=self.device, dtype=torch.float32)
        
        try:
            # 验证目标索引有效性
            if target_idx < 0 or env_idx >= len(self.selectable_objects) or target_idx >= len(self.selectable_objects[env_idx]):
                print(f"环境{env_idx}: 无效目标索引 target_idx={target_idx}")
                self.env_busy[env_idx] = False
                return action
            
            # 获取目标物体
            target_obj = self.selectable_objects[env_idx][target_idx]
            
            # 获取当前TCP位置
            tcp_pos = self.agent.tcp.pose.p
            if tcp_pos.dim() > 1:
                tcp_pos = tcp_pos[env_idx] if env_idx < tcp_pos.shape[0] else tcp_pos[0]

            # 🔧 重要修复：每次开始新的抓取任务时重新记录TCP初始位置
            if hasattr(self, 'tcp_recorded'):
                # 如果是新的抓取任务（stage=0, tick=0），或者从未记录过，则重新记录
                should_record = (stage == 0 and tick == 0) or not self.tcp_recorded[env_idx]
                if should_record:
                    self.initial_tcp_positions[env_idx] = tcp_pos.clone().detach()
                    self.tcp_recorded[env_idx] = True
                    tcp_pos_str = ', '.join([f'{x:.4f}' for x in self.initial_tcp_positions[env_idx].cpu().numpy()[:3]])
                    print(f"环境{env_idx}: 📍 记录初始TCP位置: [{tcp_pos_str}]")

            # 根据是否启用理想化神谕选择不同的FSM逻辑
            if self.use_ideal_oracle:
                return self._ideal_oracle_fsm(env_idx, stage, tick, target_obj, target_idx, tcp_pos, action)
            else:
                return self._legacy_suction_fsm(env_idx, stage, tick, target_obj, target_idx, tcp_pos, action)
                
        except Exception as e:
            print(f"状态机执行错误 env={env_idx}, stage={stage}: {e}")
            self.env_busy[env_idx] = False
            return action

    def _ideal_oracle_fsm(self, env_idx: int, stage: int, tick: int, target_obj, target_idx: int, tcp_pos, action: torch.Tensor) -> torch.Tensor:
        """
        理想化神谕FSM: 预测性的完美抓取流程
        Stage 0: 逻辑检查（遮挡+支撑）
        Stage 1: 移动到预抓取位姿
        Stage 2: 下探到抓取位姿并闭合夹爪
        Stage 3: 提升物体（理想化跟随）
        Stage 4: 瞬移物体到远处
        Stage 5: 回到初始位置并张开夹爪
        """
        if stage == 0:
            # Stage 0: 神谕逻辑检查
            if tick == 0:
                print(f"环境{env_idx}: 🔮 神谕检查 - 目标物体: {target_obj.name}")
                
                # 🔧 首先检查物体位置是否异常（是否已被瞬移）
                obj_pos = target_obj.pose.p
                if obj_pos.dim() > 1:
                    obj_pos = obj_pos[0]
                pos_magnitude = (obj_pos[0]**2 + obj_pos[1]**2 + obj_pos[2]**2)**0.5
                
                if pos_magnitude > 3.0:
                    obj_pos_str = ', '.join([f'{x:.3f}' for x in obj_pos.cpu().numpy()])
                    print(f"环境{env_idx}: ⚠️ 神谕判定：物体已被瞬移 ([{obj_pos_str}]) -> 跳过抓取")
                    self.env_busy[env_idx] = False
                    return action
                
                # 遮挡检查
                is_occluded = self._is_path_occluded_by_geometry(target_obj)
                if is_occluded:
                    print(f"环境{env_idx}: ❌ 神谕判定：路径被遮挡 -> 抓取必失败")
                    self.env_busy[env_idx] = False
                    return action
                
                # 支撑检查
                is_supporting = self._is_supporting_others(target_obj)
                if is_supporting:
                    print(f"环境{env_idx}: ❌ 神谕判定：物体正支撑其他物体 -> 抓取必失败")
                    self.env_busy[env_idx] = False
                    return action
                
                print(f"环境{env_idx}: ✅ 神谕判定：逻辑条件通过 -> 抓取必成功")
                
                # 计算理想抓取位姿
                grasp_pose, pre_grasp_pose = self._get_ideal_world_grasp_pose(target_obj)
                
                # 保存抓取相关信息
                self.stage_positions[env_idx] = torch.tensor(pre_grasp_pose.p, device=self.device)
                # 将抓取位姿保存到额外字段（如果需要）
                if not hasattr(self, 'grasp_poses'):
                    self.grasp_poses = [None] * self.num_envs
                self.grasp_poses[env_idx] = grasp_pose
                
                print(f"环境{env_idx}: 预抓取位置: {pre_grasp_pose.p}")
            
            # 直接进入下一阶段（逻辑检查只需1步）
            self.env_stage[env_idx] = 1
            self.stage_tick[env_idx] = 0
        
        elif stage == 1:
            # Stage 1: 移动到预抓取位姿
            if tick == 0:
                print(f"环境{env_idx}: Stage 1 - 移动到预抓取位置")
            
            target_pos = self.stage_positions[env_idx]
            action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=100)
            
            if reached or tick >= 100:
                print(f"环境{env_idx}: Stage 1完成 - 到达预抓取位置")
                # 设置真正的抓取位置
                if hasattr(self, 'grasp_poses') and self.grasp_poses[env_idx] is not None:
                    grasp_pos = self.grasp_poses[env_idx].p
                    self.stage_positions[env_idx] = torch.tensor(grasp_pos, device=self.device)
                
                self.env_stage[env_idx] = 2
                self.stage_tick[env_idx] = 0
            else:
                self.stage_tick[env_idx] += 1
            
            # 保持夹爪张开
            action[6] = 0.04
        
        elif stage == 2:
            # Stage 2: 下探到抓取位姿并闭合夹爪
            if tick == 0:
                print(f"环境{env_idx}: Stage 2 - 下探到抓取位置并闭合夹爪")
            
            target_pos = self.stage_positions[env_idx]
            action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=60)
            
            # 逐渐闭合夹爪
            if tick >= 20:  # 前20步用于移动，后面开始闭合夹爪
                action[6] = max(0.0, 0.04 - 0.04 * (tick - 20) / 20)  # 逐渐从0.04闭合到0.0
            else:
                action[6] = 0.04
            
            if reached or tick >= 60:
                # 建立理想化连接
                self.oracle_attached_object[env_idx] = target_obj
                print(f"环境{env_idx}: Stage 2完成 - 理想化抓取连接建立")
                
                # 设置提升目标位置
                current_pos = self.stage_positions[env_idx].clone()
                current_pos[2] += 0.20  # 上升20cm
                self.stage_positions[env_idx] = current_pos
                
                self.env_stage[env_idx] = 3
                self.stage_tick[env_idx] = 0
            else:
                self.stage_tick[env_idx] += 1
        
        elif stage == 3:
            # Stage 3: 提升物体（理想化跟随）
            if tick == 0:
                print(f"环境{env_idx}: Stage 3 - 提升物体")
            
            target_pos = self.stage_positions[env_idx]
            action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=80)
            
            # 保持夹爪闭合
            action[6] = 0.0
            
            # 理想化跟随：让物体跟随TCP移动
            if self.oracle_attached_object[env_idx] is not None:
                try:
                    # 获取TCP当前位置并设置物体位置
                    current_tcp_pos = tcp_pos.clone()
                    # 物体位置稍低于TCP（模拟被抓取的效果）
                    object_target_pos = current_tcp_pos.clone()
                    object_target_pos[2] -= 0.05  # 物体在TCP下方5cm
                    
                    # 直接设置物体位置（理想化跟随）
                    new_pose = sapien.Pose(p=object_target_pos.cpu().numpy())
                    self.oracle_attached_object[env_idx].set_pose(new_pose)
                except Exception as e:
                    print(f"环境{env_idx}: 理想化跟随失败: {e}")
            
            if reached or tick >= 80:
                # 提升完成后，等待一小会让物体稳定
                if tick >= 80 + 15:  # 提升完成后再等待15步（约0.25秒）
                    print(f"环境{env_idx}: Stage 3完成 - 物体提升完毕，等待结束")
                    self.env_stage[env_idx] = 4
                    self.stage_tick[env_idx] = 0
                elif tick == 80:  # 刚到达时的提示
                    print(f"环境{env_idx}: 物体提升到位，等待稳定中...")
                    self.stage_tick[env_idx] += 1
                else:
                    # 继续等待，保持物体跟随
                    self.stage_tick[env_idx] += 1
            else:
                self.stage_tick[env_idx] += 1
        
        elif stage == 4:
            # Stage 4: 瞬移物体到远处（不张开夹爪）
            if tick == 0:
                print(f"环境{env_idx}: Stage 4 - 瞬移物体到远处")
                
                if self.oracle_attached_object[env_idx] is not None:
                    try:
                        # 瞬移到远处（例如 (10, 10, 10)）
                        far_pose = sapien.Pose(p=[10.0, 10.0, 10.0])
                        self.oracle_attached_object[env_idx].set_pose(far_pose)
                        print(f"环境{env_idx}: ✅ 物体已瞬移到远处")
                        
                        # 断开理想化连接
                        self.oracle_attached_object[env_idx] = None
                    except Exception as e:
                        print(f"环境{env_idx}: 瞬移物体失败: {e}")
                
                # 🔧 关键修复：使用动态记录的真实初始TCP位置
                if hasattr(self, 'initial_tcp_positions') and self.initial_tcp_positions[env_idx] is not None:
                    initial_pos = self.initial_tcp_positions[env_idx]
                    initial_pos_str = ', '.join([f'{x:.4f}' for x in initial_pos.cpu().numpy()])
                    print(f"环境{env_idx}: 使用记录的初始TCP位置: [{initial_pos_str}]")
                else:
                    # 兜底：如果没有记录到初始位置，使用安全默认位置
                    initial_pos = torch.tensor([0.0, 0.0, 0.4], device=self.device)
                    print(f"环境{env_idx}: ⚠️ 未找到记录的初始位置，使用默认位置")
                
                self.stage_positions[env_idx] = initial_pos
            
            # 直接进入下一阶段（瞬移只需1步）
            self.env_stage[env_idx] = 5
            self.stage_tick[env_idx] = 0
            
            # 保持夹爪闭合
            action[6] = 0.0
        
        elif stage == 5:
            # Stage 5: 回到初始位置并张开夹爪
            if tick == 0:
                print(f"环境{env_idx}: Stage 5 - 回到初始位置")
            
            target_pos = self.stage_positions[env_idx]
            action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=100)
            
            # 到达后张开夹爪
            if reached or tick >= 80:
                action[6] = 0.04  # 张开夹爪
            else:
                action[6] = 0.0   # 保持闭合
            
            # 🔧 修复：更宽松的完成条件，避免卡死
            if (reached and tick >= 20) or tick >= 120:  # 到达后等待20步，或最多120步超时
                # 完成整个理想化流程
                self.env_busy[env_idx] = False
                self.grasped_objects[env_idx].append(target_idx)
                self.stage_tick[env_idx] = 0
                self.env_stage[env_idx] = 0  # 重置状态
                
                # 🔧 关键修复：清除TCP记录状态，为下一次抓取做准备
                if hasattr(self, 'tcp_recorded'):
                    self.tcp_recorded[env_idx] = False
                    self.initial_tcp_positions[env_idx] = None
                
                print(f"环境{env_idx}: ✅ 理想化抓取流程完成 - {target_obj.name}，已重置状态")
            else:
                self.stage_tick[env_idx] += 1
        
        else:
            # 未知状态，结束流程
            self.env_busy[env_idx] = False
        
        # 姿态控制：保持垂直向下
        action[3:6] = 0.0
        
        return action

    def _legacy_suction_fsm(self, env_idx: int, stage: int, tick: int, target_obj, target_idx: int, tcp_pos, action: torch.Tensor) -> torch.Tensor:
        """
        传统吸盘FSM（保留兼容性）
        """
        # 这里保留原有的吸盘逻辑作为fallback
        print(f"环境{env_idx}: 使用传统吸盘模式（理想化神谕未启用）")
        
        # 获取目标物体位置
        obj_pos = target_obj.pose.p[0].cpu().numpy()
        
        if stage == 0:
            # 移动到物体上方
            if tick == 0:
                target_pos = obj_pos.copy()
                target_pos[2] += 0.15  # 上方15cm
                self.stage_positions[env_idx] = torch.tensor(target_pos, device=self.device)
            
            target_pos = self.stage_positions[env_idx]
            action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=150)
            
            if reached or tick >= 150:
                self.env_stage[env_idx] = 1
                self.stage_tick[env_idx] = 0
            else:
                self.stage_tick[env_idx] += 1
        
        # 其他stage的传统逻辑...（简化版，如需完整版可继续添加）
        else:
            self.env_busy[env_idx] = False
        
        action[3:6] = 0.0  # 姿态控制
        action[6] = 0.0   # 夹爪闭合
        
        return action
    
    def _get_move_action(self, current_pos: torch.Tensor, target_pos: torch.Tensor, 
                        max_steps: int = 100) -> Tuple[torch.Tensor, bool]:
        """
        获取平滑移动动作和是否到达目标
        
        Args:
            current_pos: 当前位置
            target_pos: 目标位置
            max_steps: 最大步数（用于判断超时）
            
        Returns:
            action: 位置动作 [dx, dy, dz]
            reached: 是否到达目标
        """
        if isinstance(target_pos, np.ndarray):
            target_pos = torch.tensor(target_pos, device=self.device, dtype=torch.float32)
        
        # 计算位置误差
        pos_error = target_pos - current_pos
        current_distance = torch.linalg.norm(pos_error).item()
        
        # 判断是否到达 - 平衡精度和可达性的阈值
        reached = current_distance < 0.05  # 5cm精度，避免过于严格导致卡死
        
        if reached:
            return torch.zeros(3, device=self.device, dtype=torch.float32), True
        
        # 🔧 关键修复：使用更保守的步长策略，减少对其他物体的扰动
        max_controller_step = 0.05  # 降低最大增量到5cm，避免过大扰动
        
        # # 优化步长策略 - 提高收敛速度
        # if current_distance > 0.15:
        #     scale_factor = 1.0  # 使用100%的控制器能力
        # elif current_distance > 0.10:
        #     scale_factor = 0.95  # 稍微减速
        # elif current_distance > 0.05:
        #     scale_factor = 0.8  # 中等速度
        # 🚀 平滑步长策略 - 参考官方控制器的最佳实践
        if current_distance > 0.20:
            scale_factor = 0.8  # 长距离时适度减速
        elif current_distance > 0.15:
            scale_factor = 0.6  # 中等距离进一步减速
        elif current_distance > 0.08:
            scale_factor = 0.4  # 接近目标时显著减速
        elif current_distance > 0.04:
            scale_factor = 0.2  # 精细控制阶段
        else:
            scale_factor = 0.1  # 提高精细控制速度（从0.5提升到0.7）0.1
        
        actual_max_step = max_controller_step * scale_factor
        
        # 归一化位置误差到平滑步长
        pos_error_norm = torch.linalg.norm(pos_error)
        if pos_error_norm > actual_max_step:
            action = (pos_error / pos_error_norm) * actual_max_step
        else:
            action = pos_error
        
        # 🔧 额外平滑化：对快速变化进行限制
        # 🔧 额外平滑化：对每个维度单独限制
        action = torch.clamp(action, -actual_max_step, actual_max_step)
        
        return action, False
    
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
        fail = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_grasped = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_robot_static = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_obj_placed = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        
        if self.use_discrete_action:
            # 离散动作模式：基于抓取成功的物体数量比例评估
            for env_idx in range(self.num_envs):
                # 计算抓取成功的物体数量
                grasped_count = len(self.grasped_objects[env_idx])
                
                # 检查是否有物体被成功抓取
                is_grasped[env_idx] = grasped_count > 0
                
                # 计算成功率：抓取成功的物体数量比例
                success_ratio = grasped_count / self.total_objects_per_env
                success[env_idx] = success_ratio == 1.0   # 所有物体都被抓取认为成功
                
                # 失败条件：达到最大抓取尝试次数但未成功
                # 注意：step_count是抓取尝试次数，不是总仿真步数
                # MAX_EPISODE_STEPS=15 表示最多15次抓取尝试
                if hasattr(self, 'step_count'):
                    fail[env_idx] = self.step_count[env_idx] >= self.MAX_EPISODE_STEPS and not success[env_idx]
        else:
            # 连续动作模式：基于target_object评估
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
            "fail": fail,
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
        
        # 根据动作模式选择不同的奖励计算策略
        if self.use_discrete_action:
            # 离散动作模式：使用选择奖励逻辑
            return self._compute_discrete_action_reward(info)
        else:
            # 连续动作模式：使用原有的密集奖励逻辑
            return self._compute_continuous_action_reward(info)
    
    def _compute_discrete_action_reward(self, info: Dict):
        """计算离散动作模式的奖励 - 基于完整动作的最终结果
        
        注意：在新架构下，每个RL步骤对应一个完整的抓取动作，
        奖励基于该动作的最终结果计算，更简洁且易于收敛
        """
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # 奖励系数 - 针对完整动作优化
        R_success = 3.0      # 成功抓取一个物体的奖励
        R_complete = 15.0    # 完成所有物体的额外奖励
        R_failure = -0.5     # 抓取失败的惩罚
        w_disp = 0.8         # 位移惩罚权重
        
        for env_idx in range(self.num_envs):
            current_grasped = len(self.grasped_objects[env_idx])
            
            # 检查是否有新的成功抓取
            if not hasattr(self, '_prev_grasped_count'):
                self._prev_grasped_count = [0] * self.num_envs
            
            prev_grasped = self._prev_grasped_count[env_idx]
            
            if current_grasped > prev_grasped:
                # 成功抓取了新物体
                new_grasps = current_grasped - prev_grasped
                reward[env_idx] += R_success * new_grasps
                
                # 检查是否完成所有物体
                if current_grasped == self.total_objects_per_env:
                    reward[env_idx] += R_complete
                    
            # elif hasattr(self, 'step_count') and self.step_count[env_idx] > prev_grasped:
            #     # 有抓取尝试但没有成功（step_count增加但grasped_count没变）
            #     reward[env_idx] += R_failure
            
            # # 简化的位移惩罚
            # other_displacement = self._calculate_other_objects_displacement()
            # if env_idx < len(other_displacement):
            #     # 将位移惩罚限制在合理范围内
            #     displacement_penalty = torch.clamp(other_displacement[env_idx] * w_disp, 0, 2.0)
            #     reward[env_idx] -= displacement_penalty
        
        # 更新记录
        for env_idx in range(self.num_envs):
            self._prev_grasped_count[env_idx] = len(self.grasped_objects[env_idx])
        
        return reward
    
    def _compute_continuous_action_reward(self, info: Dict):
        """计算连续动作模式的奖励"""
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
        # 根据官方文档，normalized_dense_reward 应该是对 dense_reward 的归一化
        # 不应该根据 reward_mode 选择不同的奖励函数
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / 10.0 

 

    def _init_fsm_states(self):
        """初始化有限状态机状态张量"""
        self.env_stage = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        self.env_target = torch.full((self.num_envs,), -1, dtype=torch.int16, device=self.device)
        self.env_busy = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stage_tick = torch.zeros(self.num_envs, dtype=torch.int16, device=self.device)
        self.stage_positions = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
    
    def _force_terminate_all_tasks(self):
        """🔧 强制终止所有任务并清理状态"""
        self.env_busy.fill_(False)
        self.env_target.fill_(-1)
        self.env_stage.fill_(0)
        self.stage_tick.fill_(0)
        
        # 清理理想化附着状态
        if hasattr(self, 'oracle_attached_object'):
            for i in range(self.num_envs):
                self.oracle_attached_object[i] = None
                
        # 清理TCP状态
        if hasattr(self, 'tcp_recorded'):
            for i in range(self.num_envs):
                self.tcp_recorded[i] = False
                if hasattr(self, 'initial_tcp_positions'):
                    self.initial_tcp_positions[i] = None
        
        print("🔧 已强制终止所有FSM任务并清理状态")
    
    def _cleanup_completed_tasks(self):
        """清理已完成的任务状态"""
        for env_idx in range(self.num_envs):
            if not self.env_busy[env_idx] and self.env_target[env_idx] != -1:
                # 清理已完成的任务状态
                old_target = self.env_target[env_idx].item()
                self.env_target[env_idx] = -1
                self.env_stage[env_idx] = 0
                self.stage_tick[env_idx] = 0
                #print(f"环境{env_idx}: 任务完成，重置目标 {old_target}")
    
    def _init_anygrasp(self):
        """初始化AnyGrasp模型（只在第一次调用时加载）"""
        if not ANYGRASP_AVAILABLE:
            print("AnyGrasp不可用，跳过初始化")
            return
            
        if self.anygrasp_model is not None:
            return  # 已经初始化过了
        
        try:
            print("正在初始化AnyGrasp模型...")
            # 创建配置对象
            import argparse
            cfgs = argparse.Namespace()
            cfgs.checkpoint_path = self.ANYGRASP_CHECKPOINT
            cfgs.max_gripper_width = self.ANYGRASP_MAX_GRIPPER_WIDTH
            cfgs.gripper_height = self.ANYGRASP_GRIPPER_HEIGHT
            cfgs.top_down_grasp = self.ANYGRASP_TOP_DOWN_GRASP
            cfgs.debug = False  # 关闭调试可视化
            
            # 初始化AnyGrasp
            self.anygrasp_model = AnyGrasp(cfgs)
            self.anygrasp_model.load_net()
            print("✅ AnyGrasp模型初始化成功")
            
        except Exception as e:
            print(f"❌ AnyGrasp模型初始化失败: {e}")
            self.anygrasp_enabled = False
            self.anygrasp_model = None
    
    def _get_camera_observations(self, camera_name: str = "base_camera") -> Dict:
        """
        获取相机观测数据，包括RGB、深度和分割
        
        Args:
            camera_name: 相机名称
            
        Returns:
            包含sensor_data和sensor_param的字典
        """
        # 使用标准的ManiSkill方式获取sensor数据
        # 这会自动处理隐藏对象、更新渲染等
        for obj in self._hidden_objects:
            obj.hide_visual()
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.capture_sensor_data()
        
        # 获取sensor数据和参数
        sensor_data = {}
        sensor_params = {}
        
        if camera_name in self._sensors:
            camera = self._sensors[camera_name]
            sensor_data[camera_name] = camera.get_obs(
                rgb=True,
                depth=True,
                segmentation=True,
                position=True
            )
            sensor_params[camera_name] = camera.get_params()
        
        return {
            'sensor_data': sensor_data,
            'sensor_param': sensor_params
        }
    
    def _extract_target_pointcloud(self, target_obj: Actor, env_idx: int = 0, camera_name: str = "base_camera") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        使用实例分割提取目标物体的点云 - 使用深度图反投影方法
        
        Args:
            target_obj: 目标物体Actor对象
            env_idx: 环境索引
            camera_name: 使用的相机名称
            
        Returns:
            (points, colors): 相机坐标系下的点云和颜色，失败返回(None, None)
        """
        try:
            # 获取相机观测数据和参数
            camera_obs = self._get_camera_observations(camera_name)
            if camera_obs is None:
                return None, None
            
            sensor_data = camera_obs['sensor_data'][camera_name]
            sensor_params = camera_obs['sensor_param'][camera_name]
            
            # 提取各通道数据
            rgb = sensor_data["rgb"]  # [B, H, W, 3]
            depth = sensor_data["depth"]  # [B, H, W, 1]
            segmentation = sensor_data["segmentation"]  # [B, H, W, 1]
            
            print(f"环境{env_idx}: sensor_data包含的通道: {list(sensor_data.keys())}")
            
            # 获取当前环境的数据
            if env_idx >= rgb.shape[0]:
                print(f"环境索引{env_idx}超出范围")
                return None, None
            
            rgb_env = rgb[env_idx]  # [H, W, 3]
            depth_env = depth[env_idx]  # [H, W, 1]
            seg_env = segmentation[env_idx]  # [H, W, 1]
            
            # 获取目标物体的实例ID
            target_seg_ids = target_obj.per_scene_id  # torch.int32 tensor
            if target_seg_ids.numel() == 0:
                print(f"目标物体没有实例ID")
                return None, None
            
            # 确保在同一设备上
            if target_seg_ids.device != seg_env.device:
                target_seg_ids = target_seg_ids.to(seg_env.device)
            
            # 提取Actor层的分割（第0通道）
            actor_seg = seg_env[..., 0]  # [H, W]
            
            # 创建目标物体的基础mask
            if target_seg_ids.numel() == 1:
                base_mask = (actor_seg == target_seg_ids.item())
            else:
                # 多个ID的情况
                base_mask = torch.isin(actor_seg, target_seg_ids)
            
            # 检查基础mask是否有效
            base_pixels = base_mask.sum().item()
            if base_pixels == 0:
                print(f"环境{env_idx}: 目标物体在当前视角下不可见")
                return None, None
            
            # 🔧 修复：使用更多上下文的点云提取策略，类似官方demo
            
            # 获取托盘的segmentation ID（用于排除）
            tray_seg_ids = set()
            if hasattr(self, 'trays') and self.trays:
                for tray in self.trays:
                    if hasattr(tray, 'per_scene_id') and tray.per_scene_id.numel() > 0:
                        tray_seg_ids.add(tray.per_scene_id.item())
            
            # 🔧 策略改进：提供更多上下文信息给AnyGrasp，但智能过滤
            import torch.nn.functional as F
            
            # 1. 基础目标物体mask
            target_mask = base_mask
            
            # 2. 扩展区域包含周围物体（不包括托盘）
            # 使用更大的膨胀核来获取周围上下文，但会智能过滤
            kernel = torch.ones((1,1,7,7), device=base_mask.device)  # 使用7x7核获取更多上下文
            expanded_mask = F.conv2d(base_mask.float().unsqueeze(0).unsqueeze(0), kernel, padding=3).squeeze()
            expanded_mask = (expanded_mask > 0)
            
            # 3. 创建托盘排除mask  
            tray_exclusion_mask = torch.zeros_like(actor_seg, dtype=torch.bool)
            if tray_seg_ids:
                for tray_id in tray_seg_ids:
                    tray_exclusion_mask |= (actor_seg == tray_id)
            
            # 4. 深度有效性检查 - 🔧 修复：适应新的深度缩放
            depth_valid = (depth_env[..., 0] > 0) & (depth_env[..., 0] / 1000.0 < 1.0)
            
            # 5. 创建上下文mask：包含目标物体周围的其他物体，但排除托盘
            # 这样AnyGrasp可以看到目标物体及其周围环境，获得更好的抓取评分
            context_mask = expanded_mask & depth_valid & (actor_seg > 0) & (~tray_exclusion_mask)
            
            # 6. 最终mask：目标物体 + 周围上下文，但排除托盘
            mask = context_mask
            valid_pixels = mask.sum().item()
            
            print(f"环境{env_idx}: 原始目标像素: {base_pixels}, 增强后像素: {valid_pixels} (+{valid_pixels-base_pixels})")
            
            # 使用深度图反投影方法
            # 从sensor_params获取相机内参
            if 'intrinsic_cv' in sensor_params:
                # 使用OpenCV格式的内参矩阵
                intrinsic = sensor_params['intrinsic_cv'][env_idx].cpu().numpy()  # [3, 3]
                fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            else:
                # 备选方案：从其他参数计算
                print(f"警告：未找到intrinsic_cv，使用备选方法")
                # 检查可用的参数
                print(f"可用的sensor参数: {list(sensor_params.keys())}")
                # 使用默认值或从其他参数推导
                H, W = depth_env.shape[:2]
                fx = fy = W / 2.0  # 简单估算
                cx, cy = W / 2.0, H / 2.0
            
            print(f"环境{env_idx}: 相机内参 fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            
            # 获取mask内的像素坐标和深度值
            v, u = torch.where(mask)  # 注意：torch.where返回(row, col)即(y, x)
            z = depth_env[v, u, 0]  # 深度值
            
            # 🔧 修复深度数据格式和缩放 - 适应高分辨率相机
            if z.dtype == torch.int16:
                # ManiSkill的深度数据需要特殊的缩放因子
                # 🔧 修复：高分辨率相机需要调整缩放因子
                # 640x480相机的深度缩放因子需要重新校准
                scale_factor = 1000.0  # 更接近官方demo的scale=1000.0
                z = z.float() / scale_factor  
                print(f"环境{env_idx}: 深度数据从int16转换为米（缩放因子{scale_factor}）")
            elif z.dtype in [torch.float32, torch.float64]:
                # 已经是浮点数，假设单位是米
                z = z.float()
            
            # 过滤无效深度
            valid_depth = z > 0
            u_valid = u[valid_depth]
            v_valid = v[valid_depth]
            z_valid = z[valid_depth]
            
            if len(z_valid) == 0:
                print(f"环境{env_idx}: 没有有效的深度值")
                return None, None
            
            print(f"环境{env_idx}: 有效深度点数: {len(z_valid)}, 深度范围: [{z_valid.min():.3f}, {z_valid.max():.3f}]米")
            
            # 反投影到相机坐标系
            # 相机坐标系：X右，Y下，Z前（OpenGL风格）
            x = (u_valid.float() - cx) / fx * z_valid
            y = (v_valid.float() - cy) / fy * z_valid
            
            # 组合成点云 [N, 3]
            points_cam = torch.stack([x, y, z_valid], dim=-1).cpu().numpy().astype(np.float32)
            
            # 获取对应的颜色
            colors = rgb_env[v_valid, u_valid, :3].cpu().numpy().astype(np.float32)
            
            # 确保颜色在[0,1]范围内
            if colors.max() > 1.0:
                colors = colors / 255.0
            
            print(f"环境{env_idx}: 成功提取{len(points_cam)}个3D点")
            print(f"点云范围: X[{points_cam[:, 0].min():.3f}, {points_cam[:, 0].max():.3f}], Y[{points_cam[:, 1].min():.3f}, {points_cam[:, 1].max():.3f}], Z[{points_cam[:, 2].min():.3f}, {points_cam[:, 2].max():.3f}]")
            
            return points_cam, colors
                
        except Exception as e:
            print(f"环境{env_idx}: 提取目标点云失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _detect_grasps_for_target(self, target_obj: Actor, env_idx: int = 0, 
                                  top_k: int = 20, visualize: bool = False, 
                                  visualize_in_env: bool = False) -> Optional[List[Dict]]:
        """
        为目标物体检测抓取点
        
        Args:
            target_obj: 目标物体
            env_idx: 环境索引
            top_k: 返回前k个最佳抓取
            visualize: 是否使用Open3D/matplotlib可视化抓取结果
            visualize_in_env: 是否在仿真环境渲染图像中可视化抓取
            
        Returns:
            抓取候选列表，每个元素包含:
            - pose: 抓取位姿 (4x4 transformation matrix in world frame)
            - score: 抓取质量分数
            - width: 夹爪宽度
            失败返回None
        """
        if not self.anygrasp_enabled or self.anygrasp_model is None:
            print("AnyGrasp未启用或未初始化")
            return None
        
        try:
            # 1. 提取目标物体点云
            points, colors = self._extract_target_pointcloud(target_obj, env_idx)
            if points is None or len(points) == 0:
                print(f"环境{env_idx}: 无法提取目标物体点云")
                return None
            
            # 获取相机参数（用于后续坐标变换）
            camera_obs = self._get_camera_observations("base_camera")
            sensor_params = camera_obs['sensor_param']["base_camera"]
            
            # 2. 设置工作空间限制（相机坐标系）- 🔧 修复：更宽松的工作空间，类似官方demo
            if len(points) > 0:
                # 获取点云的整体范围
                point_min = points.min(axis=0)
                point_max = points.max(axis=0)
                obj_center = points.mean(axis=0)
                
                # 🔧 使用更宽松的工作空间设置，参考官方demo
                # 官方demo: X[-0.19, 0.12], Y[0.02, 0.15], Z[0.0, 1.0]
                # 在相机坐标系下，给予足够的空间让AnyGrasp检测
                
                # 基于点云范围，但给予充足的边距
                x_range = point_max[0] - point_min[0]
                y_range = point_max[1] - point_min[1] 
                z_range = point_max[2] - point_min[2]
                
                # 使用点云范围的1.5倍作为工作空间，但有最小值保证
                x_margin = max(x_range * 0.75, 0.12)  # 至少12cm边距
                y_margin = max(y_range * 0.75, 0.12)  # 至少12cm边距
                z_margin = max(z_range * 1.0, 0.15)   # 至少15cm深度边距
                
                xmin = obj_center[0] - x_margin
                xmax = obj_center[0] + x_margin
                ymin = obj_center[1] - y_margin
                ymax = obj_center[1] + y_margin
                zmin = max(0.01, point_min[2] - z_margin/2)
                zmax = point_max[2] + z_margin
                
                # 🔧 确保不会检测到托盘底部，但允许足够的空间
                # 托盘在相机坐标系中的大致深度是0.33+，我们设置上限为0.4
                zmax = min(zmax, 0.4)
                
                print(f"环境{env_idx}: 点云范围: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
                print(f"环境{env_idx}: 目标中心: [{obj_center[0]:.3f}, {obj_center[1]:.3f}, {obj_center[2]:.3f}]")
                print(f"环境{env_idx}: 工作空间边距: X±{x_margin:.3f}, Y±{y_margin:.3f}, Z+{z_margin:.3f}")
            else:
                # 使用类似官方demo的默认范围（相机坐标系）
                xmin, xmax = -0.2, 0.2
                ymin, ymax = -0.2, 0.2
                zmin, zmax = 0.01, 0.4
            
            lims = [xmin, xmax, ymin, ymax, zmin, zmax]
            print(f"环境{env_idx}: 工作空间限制: X[{xmin:.3f}, {xmax:.3f}], Y[{ymin:.3f}, {ymax:.3f}], Z[{zmin:.3f}, {zmax:.3f}]")
            
            # 3. 调用AnyGrasp检测抓取
            print(f"环境{env_idx}: 开始检测抓取点，输入点云大小: {points.shape}")
            print(f"环境{env_idx}: 点云数据类型: {points.dtype}, 颜色数据类型: {colors.dtype}")
            print(f"环境{env_idx}: 点云范围检查: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
            print(f"环境{env_idx}: 颜色范围检查: [{colors.min():.3f}, {colors.max():.3f}]")
            
            # 确保数据格式正确
            points = points.astype(np.float32)
            colors = colors.astype(np.float32)
            
            # 🔧 修复：优化点云密度检查和AnyGrasp参数
            min_points_threshold = 50  # 降低阈值，更现实
            if len(points) < min_points_threshold:
                print(f"环境{env_idx}: 点云稀疏({len(points)}个点，需要至少{min_points_threshold}个)，但继续尝试检测")
            
            # 🔧 修复：优化AnyGrasp参数，提高抓取质量
            gg, cloud = self.anygrasp_model.get_grasp(
                points, 
                colors, 
                lims=lims, 
                apply_object_mask=False,  # 我们已经提供了精确的目标点云
                dense_grasp=False,        # 🔧 修复：关闭密集检测，提高抓取质量
                collision_detection=True  # 🔧 修复：启用碰撞检测，过滤不合理的抓取
            )
            
            if len(gg) == 0:
                print(f"环境{env_idx}: 未检测到有效抓取")
                return None
            
            # 4. NMS和排序
            gg = gg.nms().sort_by_score()
            
            # 5. 选择Top-K抓取
            gg_top = gg[:min(top_k, len(gg))]
            
            # 6. 转换抓取表示
            grasps = []
            for i in range(len(gg_top)):
                grasp = gg_top[i]
                
                # ✅ 使用正确的方法构建4x4变换矩阵
                # 基于调试结果：grasp.grasp_array是17维向量，应该使用grasp.translation和grasp.rotation_matrix
                try:
                    if hasattr(grasp, 'translation') and hasattr(grasp, 'rotation_matrix'):
                        translation = grasp.translation  # [3] - 抓取中心位置
                        rotation = grasp.rotation_matrix  # [3, 3] - 旋转矩阵
                        
                        # 构建4x4变换矩阵（相机坐标系）
                        grasp_pose_cam = np.eye(4, dtype=np.float64)
                        grasp_pose_cam[:3, :3] = rotation
                        grasp_pose_cam[:3, 3] = translation
                        
                        # 只在调试模式下打印详细信息
                        if i == 0:  # 只为第一个抓取打印
                            print(f"环境{env_idx}: ✅ 成功构建抓取矩阵")
                            print(f"  translation: {translation}")
                            print(f"  rotation shape: {rotation.shape}")
                            print(f"  构建的4x4矩阵形状: {grasp_pose_cam.shape}")
                    else:
                        print(f"环境{env_idx}: ❌ 抓取对象缺少translation或rotation_matrix属性")
                        continue
                        
                except Exception as e:
                    print(f"环境{env_idx}: ❌ 构建抓取矩阵失败: {e}")
                    continue
                
                # 获取相机到世界的变换
                if 'cam2world_gl' in sensor_params:
                    # 使用cam2world变换矩阵
                    cam2world = sensor_params['cam2world_gl'][env_idx].cpu().numpy().astype(np.float64)
                    T_world_cam = cam2world
                else:
                    # 备选方案：使用相机配置中的固定位姿
                    camera_config_pose = self._default_sensor_configs[0].pose
                    cam_pos = np.array(camera_config_pose.p, dtype=np.float64)
                    cam_quat = np.array(camera_config_pose.q, dtype=np.float64)
                    
                    # 构建相机到世界的变换矩阵
                    T_world_cam = np.eye(4, dtype=np.float64)
                    from scipy.spatial.transform import Rotation
                    R = Rotation.from_quat(cam_quat)  # [x,y,z,w]格式
                    T_world_cam[:3, :3] = R.as_matrix()
                    T_world_cam[:3, 3] = cam_pos
                
                # 将抓取位姿转换到世界坐标系
                try:
                    grasp_pose_world = T_world_cam @ grasp_pose_cam
                    # 只为第一个抓取打印成功信息
                    if i == 0:
                        print(f"环境{env_idx}: ✅ 坐标变换成功，矩阵形状: {T_world_cam.shape} @ {grasp_pose_cam.shape} -> {grasp_pose_world.shape}")
                except Exception as matmul_error:
                    print(f"环境{env_idx}: ❌ 坐标变换失败: {matmul_error}")
                    # 暂时使用相机坐标系结果
                    grasp_pose_world = grasp_pose_cam
                    if i == 0:
                        print(f"环境{env_idx}: 使用相机坐标系结果")
                
                grasps.append({
                    'pose': grasp_pose_world,
                    'score': float(grasp.score),
                    'width': float(grasp.width),
                    'translation': grasp_pose_world[:3, 3],  # 抓取中心位置
                    'rotation': grasp_pose_world[:3, :3],    # 抓取姿态
                })
            
            # 🔧 修复：在世界坐标系中进一步过滤托盘区域的抓取
            filtered_grasps = []
            tray_center = np.array([-0.2, 0.0, 0.006])  # 托盘中心位置（世界坐标）
            tray_size = np.array([0.6, 0.6, 0.15])       # 托盘尺寸
            tray_safety_margin = 0.05  # 5cm安全边距
            
            for grasp in grasps:
                grasp_pos = grasp['translation']
                
                # 检查是否在托盘内部（考虑安全边距）
                relative_pos = np.abs(grasp_pos - tray_center)
                is_in_tray = (
                    relative_pos[0] <= (tray_size[0]/2 + tray_safety_margin) and
                    relative_pos[1] <= (tray_size[1]/2 + tray_safety_margin) and
                    grasp_pos[2] <= (tray_center[2] + tray_size[2] + tray_safety_margin)
                )
                
                if not is_in_tray:
                    filtered_grasps.append(grasp)
                else:
                    print(f"环境{env_idx}: 过滤掉托盘内的抓取点: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}], 分数={grasp['score']:.3f}")
            
            # 使用过滤后的抓取列表
            if not filtered_grasps:
                print(f"环境{env_idx}: 所有抓取都在托盘区域内，使用原始结果")
                filtered_grasps = grasps  # 如果全部被过滤，则使用原始结果
            else:
                grasps = filtered_grasps
                print(f"环境{env_idx}: 托盘过滤后剩余{len(grasps)}个抓取候选")
            
            print(f"环境{env_idx}: 最终检测到{len(grasps)}个抓取候选")
            if grasps:
                print(f"最佳抓取分数: {grasps[0]['score']:.3f}")
                print(f"最佳抓取位置: [{grasps[0]['translation'][0]:.3f}, {grasps[0]['translation'][1]:.3f}, {grasps[0]['translation'][2]:.3f}]")
            
            # 🎨 可视化功能
            if visualize:
                # 智能选择可视化方式
                try:
                    # 检测是否有图形界面环境
                    if os.environ.get('DISPLAY') is None and os.environ.get('WAYLAND_DISPLAY') is None:
                        print("🔍 检测到无图形界面环境，使用matplotlib可视化")
                        self._visualize_grasps_matplotlib(points, colors, gg_top, f"grasp_env_{env_idx}_{target_obj.name}.png")
                    else:
                        print("🔍 尝试Open3D 3D可视化...")
                        self._visualize_grasps(points, colors, gg_top, f"环境{env_idx}抓取检测结果")
                except Exception as viz_error:
                    print(f"Open3D可视化失败: {viz_error}")
                    print("🔄 自动切换到matplotlib可视化...")
                    self._visualize_grasps_matplotlib(points, colors, gg_top, f"grasp_env_{env_idx}_{target_obj.name}.png")
            
            # 🎬 在仿真环境中可视化抓取
            if visualize_in_env:
                try:
                    self._visualize_grasps_in_simulation(grasps, target_obj, env_idx)
                except Exception as env_viz_error:
                    print(f"环境可视化失败: {env_viz_error}")
            
            return grasps
            
        except Exception as e:
            print(f"环境{env_idx}: 抓取检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _visualize_grasps(self, points: np.ndarray, colors: np.ndarray, 
                          grasps, title: str = "抓取检测结果"):
        """
        使用Open3D可视化抓取结果
        
        Args:
            points: 点云坐标 [N, 3]
            colors: 点云颜色 [N, 3] 
            grasps: GraspGroup对象或抓取列表
            title: 可视化窗口标题
        """
        try:
            import open3d as o3d
            print(f"🎨 开始可视化: {title}")
            
            # 1. 创建点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            
            print(f"点云包含 {len(points)} 个点")
            
            # 2. 获取抓取几何对象
            if hasattr(grasps, 'to_open3d_geometry_list'):
                # GraspGroup对象
                grippers = grasps.to_open3d_geometry_list()
                print(f"生成了 {len(grippers)} 个抓取器几何对象")
            else:
                print("❌ 无法从抓取对象生成几何对象")
                return
            
            # 3. 坐标变换（与demo.py保持一致）
            # 翻转Z轴以适应可视化坐标系
            trans_mat = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0], 
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            
            pcd.transform(trans_mat)
            for gripper in grippers:
                gripper.transform(trans_mat)
            
            # 4. 可视化选项
            print("🎨 显示可视化窗口...")
            print("操作提示:")
            print("  - 鼠标左键拖拽: 旋转视角")
            print("  - 鼠标右键拖拽: 平移视角") 
            print("  - 滚轮: 缩放")
            print("  - 按Q或关闭窗口: 退出")
            
            # 尝试显示所有抓取
            print(f"显示点云 + 所有{len(grippers)}个抓取候选")
            try:
                success = o3d.visualization.draw_geometries(
                    [pcd] + grippers,
                    window_name=f"{title} - 所有抓取",
                    width=800,
                    height=600
                )
                
                # 显示最佳抓取（如果有的话）
                if len(grippers) > 0:
                    print("显示点云 + 最佳抓取")
                    o3d.visualization.draw_geometries(
                        [pcd, grippers[0]],
                        window_name=f"{title} - 最佳抓取",
                        width=800,
                        height=600
                    )
                
                print("✅ 可视化完成")
                
            except Exception as display_error:
                print(f"Open3D窗口显示失败: {display_error}")
                # 这里不抛出异常，因为可视化失败不应该影响主要功能
                print("⚠️ Open3D可视化失败，但检测功能正常")
            
        except ImportError:
            print("❌ 需要安装Open3D才能使用可视化功能: pip install open3d")
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_grasps_matplotlib(self, points: np.ndarray, colors: np.ndarray, 
                                   grasps, filename: str = "grasp_result.png"):
        """
        使用matplotlib生成2D抓取可视化（服务器友好）
        
        Args:
            points: 点云坐标 [N, 3]
            colors: 点云颜色 [N, 3] 
            grasps: GraspGroup对象或抓取列表
            filename: 输出文件名
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib
            matplotlib.use('Agg')  # 无界面后端
            
            print(f"🎨 使用matplotlib生成可视化: {filename}")
            
            # 提取抓取信息
            grasp_positions = []
            grasp_scores = []
            for i in range(len(grasps)):
                grasp = grasps[i]
                if hasattr(grasp, 'translation') and hasattr(grasp, 'score'):
                    pos = grasp.translation
                    grasp_positions.append(pos)
                    grasp_scores.append(grasp.score)
            
            if not grasp_positions:
                print("❌ 没有有效的抓取数据")
                return
            
            grasp_positions = np.array(grasp_positions)
            grasp_scores = np.array(grasp_scores)
            
            # 创建多子图
            fig = plt.figure(figsize=(12, 8))
            
            # 3D点云+抓取点
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       c=colors, s=2, alpha=0.6, label='Point Cloud')
            ax1.scatter(grasp_positions[:, 0], grasp_positions[:, 1], grasp_positions[:, 2],
                       c=grasp_scores, s=50, cmap='viridis', marker='^', alpha=0.8, label='Grasps')
            ax1.set_title('3D Point Cloud + Grasp Candidates')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.legend()
            
            # XY平面视图
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.scatter(points[:, 0], points[:, 1], c=colors[:, 0], s=5, alpha=0.6, label='Points')
            scatter = ax2.scatter(grasp_positions[:, 0], grasp_positions[:, 1], 
                                c=grasp_scores, s=80, cmap='viridis', marker='^', alpha=0.8)
            best_idx = np.argmax(grasp_scores)
            ax2.scatter(grasp_positions[best_idx, 0], grasp_positions[best_idx, 1], 
                       s=150, facecolors='none', edgecolors='red', linewidth=2, label='Best')
            plt.colorbar(scatter, ax=ax2, label='Score')
            ax2.set_title('XY Plane View')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 抓取分数柱状图
            ax3 = fig.add_subplot(2, 2, 3)
            bars = ax3.bar(range(len(grasp_scores)), grasp_scores, color='skyblue', alpha=0.7)
            bars[best_idx].set_color('red')  # 高亮最佳抓取
            ax3.axhline(y=np.mean(grasp_scores), color='orange', linestyle='--', 
                       label=f'Mean: {np.mean(grasp_scores):.4f}')
            ax3.set_title('Grasp Quality Distribution')
            ax3.set_xlabel('Grasp Index')
            ax3.set_ylabel('Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 统计信息文本
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis('off')
            
            stats_text = f"""
抓取检测统计:
━━━━━━━━━━━━━━
• 点云大小: {len(points)} 个点
• 检测抓取: {len(grasps)} 个
• 最佳分数: {np.max(grasp_scores):.4f}
• 平均分数: {np.mean(grasp_scores):.4f}
• 分数标准差: {np.std(grasp_scores):.4f}

最佳抓取位置:
• X: {grasp_positions[best_idx, 0]:.3f} m
• Y: {grasp_positions[best_idx, 1]:.3f} m  
• Z: {grasp_positions[best_idx, 2]:.3f} m

点云范围:
• X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]
• Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]
• Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]
"""
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ matplotlib可视化已保存: {filename}")
            
        except ImportError:
            print("❌ 需要安装matplotlib才能使用备用可视化功能")
        except Exception as e:
            print(f"❌ matplotlib可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_grasps_in_simulation(self, grasps: List[Dict], target_obj: Actor, env_idx: int = 0):
        """
        在仿真环境渲染图像中可视化抓取位姿
        
        Args:
            grasps: 抓取候选列表
            target_obj: 目标物体
            env_idx: 环境索引
        """
        try:
            import cv2
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            print(f"🎬 在仿真环境中可视化抓取...")
            
            # 1. 获取环境渲染图像
            env_image = self.render_rgb_array()  # 获取环境RGB图像
            if env_image is None:
                print("❌ 无法获取环境渲染图像")
                return
            
            # 如果是多环境，选择对应环境的图像
            if len(env_image.shape) == 4:  # [num_envs, H, W, C]
                if env_idx < env_image.shape[0]:
                    env_image = env_image[env_idx]
                else:
                    env_image = env_image[0]
                    print(f"⚠️ 环境索引{env_idx}越界，使用环境0的图像")
            
            # 转换为numpy数组和0-255范围的uint8格式
            if isinstance(env_image, torch.Tensor):
                env_image = env_image.cpu().numpy()
            
            if env_image.dtype == np.float32 or env_image.dtype == np.float64:
                if env_image.max() <= 1.0:
                    env_image = (env_image * 255).astype(np.uint8)
                else:
                    env_image = env_image.astype(np.uint8)
            
            print(f"环境图像形状: {env_image.shape}, 数据类型: {env_image.dtype}")
            
            # 2. 获取相机参数
            camera_obs = self._get_camera_observations("base_camera")
            if camera_obs is None or "base_camera" not in camera_obs['sensor_param']:
                print("❌ 无法获取相机参数")
                return
            
            sensor_params = camera_obs['sensor_param']["base_camera"]
            
            # 3. 在图像上绘制抓取
            annotated_image = self._draw_grasps_on_image(env_image.copy(), grasps, sensor_params, env_idx)
            
            # 4. 保存结果
            output_filename = f"grasp_simulation_env_{env_idx}_{target_obj.name}.png"
            
            # 使用matplotlib保存（更好的质量控制）
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 原始环境图像
            ax1.imshow(env_image)
            ax1.set_title('Original Environment View')
            ax1.axis('off')
            
            # 带抓取标注的图像
            ax2.imshow(annotated_image)
            ax2.set_title(f'Grasp Candidates ({len(grasps)} detected)')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 环境可视化已保存: {output_filename}")
            
            # 5. 保存抓取信息
            info_filename = output_filename.replace('.png', '_info.txt')
            with open(info_filename, 'w', encoding='utf-8') as f:
                f.write(f"环境抓取可视化报告\n")
                f.write(f"========================\n")
                f.write(f"目标物体: {target_obj.name}\n")
                f.write(f"环境索引: {env_idx}\n")
                f.write(f"图像尺寸: {env_image.shape}\n")
                f.write(f"检测到抓取数量: {len(grasps)}\n\n")
                
                for i, grasp in enumerate(grasps):
                    f.write(f"抓取 {i+1}:\n")
                    f.write(f"  分数: {grasp['score']:.4f}\n")
                    f.write(f"  位置: [{grasp['translation'][0]:.3f}, {grasp['translation'][1]:.3f}, {grasp['translation'][2]:.3f}]\n")
                    f.write(f"  宽度: {grasp['width']:.3f}m\n\n")
            
            print(f"📋 抓取信息已保存: {info_filename}")
            
        except ImportError as e:
            print(f"❌ 缺少依赖: {e}")
            print("需要安装: pip install opencv-python")
        except Exception as e:
            print(f"❌ 环境可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_grasps_on_image(self, image: np.ndarray, grasps: List[Dict], 
                             sensor_params: Dict, env_idx: int = 0) -> np.ndarray:
        """
        在图像上绘制抓取位姿
        
        Args:
            image: 环境图像 [H, W, 3]
            grasps: 抓取候选列表
            sensor_params: 相机参数
            env_idx: 环境索引
            
        Returns:
            带抓取标注的图像
        """
        try:
            import cv2
            
            # 获取相机内参
            if 'intrinsic_cv' in sensor_params:
                intrinsic = sensor_params['intrinsic_cv'][env_idx].cpu().numpy()
                fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            else:
                # 使用默认内参
                H, W = image.shape[:2]
                fx = fy = W / 2.0
                cx, cy = W / 2.0, H / 2.0
            
            # 获取相机位姿（世界到相机的变换）
            if 'cam2world_gl' in sensor_params:
                cam2world = sensor_params['cam2world_gl'][env_idx].cpu().numpy()
                world2cam = np.linalg.inv(cam2world)
            else:
                # 使用单位矩阵（假设相机在世界原点）
                world2cam = np.eye(4)
            
            annotated_image = image.copy()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # BGR格式
            
            for i, grasp in enumerate(grasps):
                try:
                    # 获取抓取位置（世界坐标系）
                    world_pos = np.array([grasp['translation'][0], grasp['translation'][1], grasp['translation'][2], 1.0])
                    
                    # 转换到相机坐标系
                    cam_pos = world2cam @ world_pos
                    x_cam, y_cam, z_cam = cam_pos[:3]
                    
                    # 投影到图像平面
                    if z_cam > 0:  # 确保在相机前方
                        u = int(fx * x_cam / z_cam + cx)
                        v = int(fy * y_cam / z_cam + cy)
                        
                        # 检查是否在图像范围内
                        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                            color = colors[i % len(colors)]
                            
                            # 绘制抓取点
                            radius = max(5, int(20 * grasp['score'] / 0.1))  # 根据分数调整大小
                            cv2.circle(annotated_image, (u, v), radius, color, -1)
                            
                            # 绘制抓取轮廓（简化的夹爪表示）
                            gripper_size = int(grasp['width'] * 1000)  # 转换为像素
                            gripper_size = max(10, min(50, gripper_size))  # 限制大小
                            
                            # 绘制夹爪轮廓
                            cv2.rectangle(annotated_image, 
                                        (u - gripper_size//2, v - gripper_size//2),
                                        (u + gripper_size//2, v + gripper_size//2),
                                        color, 2)
                            
                            # 添加文本标签
                            label = f"G{i+1}: {grasp['score']:.3f}"
                            cv2.putText(annotated_image, label, (u + 10, v - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
                            print(f"  抓取{i+1}: 世界({world_pos[:3]}) -> 相机({x_cam:.2f},{y_cam:.2f},{z_cam:.2f}) -> 图像({u},{v})")
                    else:
                        print(f"  抓取{i+1}: 在相机后方，跳过 (z_cam={z_cam:.2f})")
                        
                except Exception as draw_error:
                    print(f"绘制抓取{i+1}失败: {draw_error}")
                    continue
            
            # 添加图例
            legend_y = 30
            for i, grasp in enumerate(grasps[:5]):  # 最多显示5个
                color = colors[i % len(colors)]
                cv2.rectangle(annotated_image, (10, legend_y + i*25), (30, legend_y + i*25 + 15), color, -1)
                text = f"Grasp {i+1}: Score {grasp['score']:.3f}"
                cv2.putText(annotated_image, text, (35, legend_y + i*25 + 12),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return annotated_image
            
        except Exception as e:
            print(f"❌ 图像标注失败: {e}")
            return image