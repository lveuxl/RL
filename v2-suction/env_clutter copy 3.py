import os
from typing import Any, Dict, List, Union, Tuple
import numpy as np
import sapien
import torch
import cv2
import random

# 导入配置
try:
    from .config import Config, get_config
except ImportError:
    # 处理直接运行时的相对导入问题
    from config import Config, get_config

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
    
    # 托盘参数 (基于traybox.urdf的尺寸)
    tray_size = [0.6, 0.6, 0.15]  # 托盘内部尺寸 (长x宽x高)
    tray_spawn_area = [0.23, 0.23]  # 托盘内物体生成区域 (考虑边界墙和安全边距)
    
    # 注意：物体相关参数现在从config中动态获取
    # BOX_OBJECTS, num_objects_per_type, MAX_N, MAX_EPISODE_STEPS 等
    # 都在 __init__ 方法中从配置文件初始化
    
    # 新增：吸盘约束相关常量
    SUCTION_DISTANCE_THRESHOLD = 0.15  # 吸盘激活距离阈值 从10cm增加到15cm
    SUCTION_STIFFNESS = 1e6  # 吸盘约束刚度
    SUCTION_DAMPING = 1e4    # 吸盘约束阻尼
    
    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        use_discrete_action=False,  # 新增：是否使用离散动作
        config_preset="default",    # 新增：配置预设名称
        custom_config=None,         # 新增：自定义配置对象
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_discrete_action = use_discrete_action
        
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
                tcp_pos = tcp_pos[env_idx] 
            
            obj_pos = target_object.pose.p
            obj_pos = obj_pos[0]
            
            # 计算距离
            raw_distance = torch.linalg.norm(tcp_pos - obj_pos).item()
            # 🔧 修复：使用更合理的半径估计值
            # TCP半径约2cm，物体平均半径约3cm，总计约5cm
            estimated_radius = 0.1  # 10cm的半径估计，与_check_suction_grasp_success保持一致
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
                estimated_radius = 0.1  # 10cm的半径估计，与_is_contacting_object保持一致
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
            
            
            # 使用指定的机器人初始姿态重置
            # 指定的关节位置：[-1.6137, 1.3258, 1.9346, -0.8884, -1.6172, 1.0867, -3.0494, 0.04, 0.04]
            #target_qpos = np.array([-0.5370, 1.3258, 1.9346, -0.8884, -1.6172, 1.0867, -3.0494, 0.04, 0.04])

            # 重置机器人到指定姿态
            #self.agent.reset(target_qpos)
            self.agent.reset()

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

    
    def _close_gripper(self):
        """闭合夹爪"""
        # 构建7维动作向量 [dx, dy, dz, drx, dry, drz, gripper]
        action = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float32)
        action[:, 6] = 0.00  # 闭合夹爪
        
        # 执行多步以确保夹爪闭合
        for _ in range(5):
            self._low_level_step(action)
    
    def _open_gripper(self):
        """打开夹爪"""
        # 构建7维动作向量 [dx, dy, dz, drx, dry, drz, gripper]
        action = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float32)
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
        处理离散动作的step方法 - 并行状态机版本
        
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
        
        # 1. 把新指令分配给空闲环境
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
        
        # 2. 生成连续动作 - 为所有忙碌的环境执行一步FSM
        cmd = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float32)
        for i in range(self.num_envs):
            if self.env_busy[i]:
                cmd[i] = self._pick_object_step(i)
        
        # 3. 执行一步仿真
        super().step(cmd)
        
        # 4. 更新环境状态（重置刚完成抓取的目标）
        for env_idx in range(self.num_envs):
            if not self.env_busy[env_idx] and self.env_target[env_idx] != -1:
                # 重置目标
                self.env_target[env_idx] = -1
        
        # 5. 使用标准的奖励计算流程
        info = self.get_info()
        obs = self.get_obs(info)
        reward = self.get_reward(obs=obs, action=action, info=info)
        
        # 6. 检查终止条件 - 使用标准的 ManiSkill 逻辑
        # success 和 fail 都会导致 episode 提前结束
        # info 已经包含了 evaluate() 的结果，直接使用
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
        单步状态机执行 - 每次调用只执行当前状态的一小步
        
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
            # 关键修复：使用selectable_objects根据环境和相对索引获取正确的物体对象
            if target_idx < 0 or env_idx >= len(self.selectable_objects) or target_idx >= len(self.selectable_objects[env_idx]):
                # 无效目标，结束流程
                print(f"环境{env_idx}: 无效目标索引 target_idx={target_idx}, selectable_objects长度={len(self.selectable_objects[env_idx]) if env_idx < len(self.selectable_objects) else 0}")
                self.env_busy[env_idx] = False
                return action
            
            # 使用环境特定的物体列表获取目标物体
            target_obj = self.selectable_objects[env_idx][target_idx]
            #print(f"环境{env_idx}: 使用目标物体 {target_obj.name} (环境内索引={target_idx})")
            
            # 获取目标物体位置
            obj_pos = target_obj.pose.p
            obj_pos = obj_pos[0]
            obj_pos = obj_pos.cpu().numpy()
            
            # 获取当前TCP位置
            tcp_pos = self.agent.tcp.pose.p
            if tcp_pos.dim() > 1:
                if env_idx < tcp_pos.shape[0]:
                    tcp_pos = tcp_pos[env_idx]
                else:
                    tcp_pos = tcp_pos[0]
                    print(f"⚠️ 环境{env_idx}: TCP位置索引越界，使用环境0的位置")
            
            # 状态机逻辑
            if stage == 0:
                # 状态0: 移动到物体上方
                if tick == 0:
                    # 第一次进入此状态，设置目标位置
                    target_pos = obj_pos.copy()
                    target_pos[2] += 0.15  # 上方15cm
                    self.stage_positions[env_idx] = torch.tensor(target_pos, device=self.device)
                    print(f"环境{env_idx}: 状态0初始化 - 物体位置={obj_pos}, 目标位置={target_pos}")
                
                target_pos = self.stage_positions[env_idx]
                action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=150)
                
                # 添加调试信息
                current_distance = torch.linalg.norm(tcp_pos - target_pos).item()
                # 打印进度（减少输出频率）
                if tick % 10 == 0 :  # 前5步和每30步输出一次
                    #print(f"环境{env_idx}: 状态0 步{tick} - TCP位置={tcp_pos.cpu().numpy()}, 目标位置={target_pos.cpu().numpy()}, 距离={current_distance:.4f}m, 到达={'是' if reached else '否'}")
                    print(f"环境{env_idx}: 状态0 步{tick}, 距离={current_distance:.4f}m, 到达={'是' if reached else '否'}")
                
                if reached or tick >= 150:
                    print(f"环境{env_idx}: 状态0完成 - reached={reached}, tick={tick}")
                    self.env_stage[env_idx] = 1
                    self.stage_tick[env_idx] = 0
                else:
                    self.stage_tick[env_idx] += 1
            
            elif stage == 1:
                # 状态1: 下降到物体上方
                if tick == 0:
                    target_pos = obj_pos.copy()
                    target_pos[2] += 0.05  # 上方3cm
                    self.stage_positions[env_idx] = torch.tensor(target_pos, device=self.device)
                
                target_pos = self.stage_positions[env_idx]
                action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=80)
                
                if reached or tick >= 80:
                    self.env_stage[env_idx] = 2
                    self.stage_tick[env_idx] = 0
                else:
                    self.stage_tick[env_idx] += 1
            
            elif stage == 2:
                # 状态2: 抓取物体
                if tick == 0:
                    target_pos = obj_pos.copy()
                    target_pos[2] += 0.01  # 上方1cm
                    self.stage_positions[env_idx] = torch.tensor(target_pos, device=self.device)
                
                target_pos = self.stage_positions[env_idx]
                action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=80)
                
                if reached or tick >= 80:
                    # 尝试创建吸盘约束
                    suction_success = self._create_suction_constraint(target_obj, env_idx)
                    if suction_success and self._check_suction_grasp_success(target_obj, env_idx):
                        self.env_stage[env_idx] = 3
                        self.stage_tick[env_idx] = 0
                    else:
                        # 抓取失败，结束流程
                        print(f"环境{env_idx}: 抓取失败，结束流程")
                        self.env_busy[env_idx] = False
                else:
                    self.stage_tick[env_idx] += 1
            
            elif stage == 3:
                # 状态3: 物体上升
                if tick == 0:
                    current_pos = tcp_pos.cpu().numpy()
                    target_pos = current_pos.copy()
                    target_pos[2] += 0.2  # 上升20cm
                    self.stage_positions[env_idx] = torch.tensor(target_pos, device=self.device)
                
                target_pos = self.stage_positions[env_idx]
                action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=100)
                
                if reached or tick >= 100:
                    self.env_stage[env_idx] = 4
                    self.stage_tick[env_idx] = 0
                else:
                    self.stage_tick[env_idx] += 1
            
            elif stage == 4:
                # 状态4: 移动到放置位置
                if tick == 0:
                    current_z = tcp_pos[2].item()
                    target_pos = np.array([-0.4, 0.4, current_z])  # 保持当前高度
                    self.stage_positions[env_idx] = torch.tensor(target_pos, device=self.device)
                
                target_pos = self.stage_positions[env_idx]
                action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=180)
                
                if reached or tick >= 180:
                    self.env_stage[env_idx] = 5
                    self.stage_tick[env_idx] = 0
                else:
                    self.stage_tick[env_idx] += 1
            
            elif stage == 5:
                # 状态5: 下降到放置位置
                if tick == 0:
                    current_pos = self.stage_positions[env_idx].cpu().numpy()  # 使用状态4的目标位置
                    target_pos = current_pos.copy()
                    target_pos[2] = 0.05  # 下降到桌面高度5cm
                    self.stage_positions[env_idx] = torch.tensor(target_pos, device=self.device)
                
                target_pos = self.stage_positions[env_idx]
                action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=100)
                
                if reached or tick >= 100:
                    self.env_stage[env_idx] = 6
                    self.stage_tick[env_idx] = 0
                else:
                    self.stage_tick[env_idx] += 1
            
            elif stage == 6:
                # 状态6: 放下物体
                if tick == 0:
                    # 尝试移除吸盘约束（只在第一次尝试）
                    try:
                        if self.is_suction_active[env_idx] and self.current_suction_object[env_idx] is not None:
                            success = self._remove_suction_constraint(env_idx)
                            if success:
                                print(f"移除吸盘约束成功 env={env_idx}")
                            else:
                                print(f"移除吸盘约束失败 env={env_idx}")
                            
                    except Exception as e:
                        # 吸盘约束移除失败不影响状态转换
                        print(f"移除吸盘约束异常 env={env_idx}: {e}")
                        pass
                
                # 等待物体稳定
                if tick >= 10:  # 等待10步让物体稳定
                    self.env_stage[env_idx] = 7
                    self.stage_tick[env_idx] = 0
                else:
                    self.stage_tick[env_idx] += 1
            
            elif stage == 7:
                # 状态7: 回到初始位置
                if tick == 0:
                    target_pos = np.array([-0.6, 0.4, 0.4])  # 安全的初始位置
                    self.stage_positions[env_idx] = torch.tensor(target_pos, device=self.device)
                
                target_pos = self.stage_positions[env_idx]
                action[:3], reached = self._get_move_action(tcp_pos, target_pos, max_steps=100)
                
                if reached or tick >= 100:
                    # 完成整个流程
                    self.env_busy[env_idx] = False
                    # 修复：使用相对索引标记抓取成功
                    self.grasped_objects[env_idx].append(target_idx)
                    self.stage_tick[env_idx] = 0
                    print(f"环境{env_idx}完成抓取物体{target_obj.name} (相对索引={target_idx})")
                else:
                    self.stage_tick[env_idx] += 1
            
            else:
                # 未知状态，结束流程
                self.env_busy[env_idx] = False
            
            # 姿态控制：保持垂直向下
            action[3:6] = 0.0
            # 夹爪控制
            action[6] = 0.0
            
            return action
            
        except Exception as e:
            print(f"状态机执行错误 env={env_idx}, stage={stage}: {e}")
            # 出错时结束流程
            self.env_busy[env_idx] = False
            return action
    
    def _get_move_action(self, current_pos: torch.Tensor, target_pos: torch.Tensor, 
                        max_steps: int = 100) -> Tuple[torch.Tensor, bool]:
        """
        获取移动动作和是否到达目标
        
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
        
        # 判断是否到达 - 放宽阈值到5cm
        reached = current_distance < 0.05
        
        if reached:
            print(f"_get_move_action: 已到达目标，距离={current_distance:.4f}m")
            return torch.zeros(3, device=self.device, dtype=torch.float32), True
        
        # 计算移动动作
        max_controller_step = 0.1  # 控制器支持的最大增量：10cm
        
        # 优化步长策略 - 提高收敛速度
        if current_distance > 0.15:
            scale_factor = 1.0  # 使用100%的控制器能力
        elif current_distance > 0.10:
            scale_factor = 0.95  # 稍微减速
        elif current_distance > 0.05:
            scale_factor = 0.8  # 中等速度
        else:
            scale_factor = 0.7  # 提高精细控制速度（从0.5提升到0.7）
        
        actual_max_step = max_controller_step * scale_factor
        
        # 归一化位置误差
        pos_error_norm = torch.linalg.norm(pos_error)
        if pos_error_norm > actual_max_step:
            action = (pos_error / pos_error_norm) * actual_max_step
        else:
            action = pos_error
        
        #print(f"_get_move_action: 距离={current_distance:.4f}m, 动作={action.cpu().numpy()}, scale_factor={scale_factor}")
        
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
        """计算离散动作模式的奖励 - 易收敛的baseline版本
        
        注意：在离散动作模式下，每个动作代表选择一个物体进行抓取
        奖励在每个仿真步都会计算，但主要的奖励信号来自于抓取成功
        """
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # 奖励系数 - 易调参、易收敛
        w_success = 2.0      # 成功抓取奖励权重
        w_time = 0.01        # 时间惩罚权重（每个仿真步）
        w_disp = 0.5         # 位移惩罚权重（简化处理）
        R_complete = 10.0    # 全部完成大奖励
        disp_scale = 0.1     # 位移缩放因子
        
        for env_idx in range(self.num_envs):
            # 计算当前状态
            grasped_count = len(self.grasped_objects[env_idx])
            
            # 检查是否刚完成一次抓取动作（环境从忙碌变为空闲）
            if not self.env_busy[env_idx] and hasattr(self, '_prev_grasped_count'):
                # 检查是否有新的物体被抓取
                prev_count = getattr(self, '_prev_grasped_count', [0] * self.num_envs)[env_idx]
                if grasped_count > prev_count:
                    # 成功抓取了新物体
                    reward[env_idx] += w_success * 1.0  # 单次成功奖励
                    
                    # 简化的位移惩罚（假设其他物体的位移很小）
                    # 在实际实现中，可以计算其他物体位置的变化
                    other_displacement = 0.0  # 简化为0，避免复杂计算
                    reward[env_idx] -= w_disp * min(other_displacement / disp_scale, 1.0)
            
            # 时间惩罚 - 每步都有
            reward[env_idx] -= w_time
            
            # 全部完成大奖励
            if grasped_count == self.total_objects_per_env:
                reward[env_idx] += R_complete
        
        # 记录当前抓取数量，用于下次比较
        if not hasattr(self, '_prev_grasped_count'):
            self._prev_grasped_count = [0] * self.num_envs
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