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
    
    
    # AnyGrasp相关配置 - 🔧 修复：使用官方demo的高质量参数
    ANYGRASP_CHECKPOINT = "/home/linux/jzh/RL_Robot/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar"  # 模型权重路径
    ANYGRASP_MAX_GRIPPER_WIDTH = 0.1   # 🔧 修复：增加到10cm，与官方demo一致
    ANYGRASP_GRIPPER_HEIGHT = 0.025     
    ANYGRASP_TOP_DOWN_GRASP = False     # 是否优先顶部抓取
    
    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        parallel_in_single_scene=False,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # 基本物体配置 - 简化为固定配置
        self.BOX_OBJECTS = ["004_sugar_box", "009_gelatin_box", "008_pudding_box"]
        self.num_objects_per_type = 18  # 每种类型4个物体
        self.total_objects_per_env = 54  # 总共12个物体
        
        # 任务相关参数
        self.goal_thresh = 0.05  # 成功阈值
        self.MAX_EPISODE_STEPS = 500  # 最大步数
        
        # 跟踪变量
        self.grasped_objects_count = 0  # 已成功抓取的物体数量
        self.current_target_idx = 0     # 当前目标物体索引
        
        # 初始位置记录（用于计算位移）
        self.initial_positions = {}
        
        # 简化环境使用连续动作模式
        self.use_discrete_action = False
        
        # YCB数据集缓存
        self._ycb_dataset = None
        
        # 确保所有参数正确传递给父类
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            parallel_in_single_scene=parallel_in_single_scene,
            **kwargs,
        )
    
    def _load_ycb_dataset(self):
        """加载YCB数据集信息，基于官方实现"""
        if self._ycb_dataset is None:
            self._ycb_dataset = {
                "model_data": load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"),
            }
        return self._ycb_dataset
    
    def _create_scaled_ycb_builder(self, obj_id: str, scale: float = 1.0, add_collision: bool = True, add_visual: bool = True):
        """
        创建缩放的YCB对象构建器，基于官方get_ycb_builder函数实现
        
        Args:
            obj_id: YCB对象ID，如"004_sugar_box"
            scale: 缩放比例，如0.7表示缩小到原来的70%
            add_collision: 是否添加碰撞形状
            add_visual: 是否添加视觉形状
        """
        # 确保数据集已加载
        dataset = self._load_ycb_dataset()
        model_db = dataset["model_data"]
        
        # 创建actor builder
        builder = self.scene.create_actor_builder()
        
        # 获取模型元数据
        metadata = model_db[obj_id]
        density = metadata.get("density", 1000)
        
        # 使用自定义缩放而不是metadata中的scales
        custom_scale = scale
        physical_material = None
        
        # 构建模型路径
        model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / obj_id
        
        # 添加碰撞形状（如果需要）
        if add_collision:
            collision_file = str(model_dir / "collision.ply")
            builder.add_multiple_convex_collisions_from_file(
                filename=collision_file,
                scale=[custom_scale] * 3,  # 应用自定义缩放
                material=physical_material,
                density=density,
            )
        
        # 添加视觉形状（如果需要）
        if add_visual:
            visual_file = str(model_dir / "textured.obj")
            builder.add_visual_from_file(
                filename=visual_file, 
                scale=[custom_scale] * 3  # 应用自定义缩放
            )
        
        return builder
        

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
                    # 创建物体 - 针对004_sugar_box应用特殊缩放
                    if obj_type == "004_sugar_box":
                        builder = self._create_scaled_ycb_builder(obj_type, scale=0.7)
                    else:
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
            color=[0, 1, 0, 0],
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
        """获取物体的大小信息，考虑缩放效果"""
        # 基于YCB数据集的实际物体尺寸（单位：米）
        base_sizes = {
            #"003_cracker_box": [0.16, 0.21, 0.07],         # 饼干盒: 16cm x 21cm x 7cm
            "004_sugar_box": [0.09, 0.175, 0.044],         # 糖盒: 9cm x 17.5cm x 4.4cm
            "006_mustard_bottle": [0.095, 0.095, 0.177],   # 芥末瓶: 9.5cm x 9.5cm x 17.7cm
            "008_pudding_box": [0.078, 0.109, 0.032],      # 布丁盒: 7.8cm x 10.9cm x 3.2cm
            "009_gelatin_box": [0.028, 0.085, 0.114],      # 明胶盒: 2.8cm x 8.5cm x 11.4cm  
            #"010_potted_meat_can": [0.101, 0.051, 0.051],  # 罐装肉罐头: 10.1cm x 5.1cm x 5.1cm
        }
        
        base_size = base_sizes.get(obj_type, [0.05, 0.05, 0.05])
        
        # 对于004_sugar_box应用0.7倍缩放
        if obj_type == "004_sugar_box":
            return [dim * 0.7 for dim in base_size]  # 缩放到原来的70%
        else:
            return base_size

    def _sample_target_objects(self):
        """随机选择目标物体"""
        target_objects = []
        self.target_object_indices = []
        
        for env_idx in range(self.num_envs):
            if len(self.selectable_objects[env_idx]) > 0:
                # 随机选择一个尚未抓取的物体
                available_objects = [obj for i, obj in enumerate(self.selectable_objects[env_idx]) 
                                   if i not in getattr(self, 'completed_objects', set())]
                
                if available_objects:
                    target_obj = random.choice(available_objects)
                    target_idx = self.selectable_objects[env_idx].index(target_obj)
                    target_objects.append(target_obj)
                    self.target_object_indices.append(target_idx)
                else:
                    # 如果没有可选择的物体，使用第一个物体作为占位符
                    target_objects.append(self.selectable_objects[env_idx][0])
                    self.target_object_indices.append(0)
        
        if target_objects:
            self.target_object = Actor.merge(target_objects, name="target_object")
            
        # 初始化完成的物体集合
        if not hasattr(self, 'completed_objects'):
            self.completed_objects = set()

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
            
            
            # 重新选择目标物体 - 只在连续动作模式下使用
            if not self.use_discrete_action:
                self._sample_target_objects()
        
            
            
            
            # 重置任务相关变量
            self.grasped_objects_count = 0
            self.completed_objects = set()
            
            # 记录所有物体的初始位置（用于计算位移）
            self.initial_positions = {}
            for i, obj in enumerate(self.all_objects):
                if hasattr(obj, 'pose'):
                    self.initial_positions[i] = obj.pose.p.clone()
            
            # 选择新的目标物体
            self._sample_target_objects()
            
            # 初始姿态重置
            target_qpos = np.array([-1.6137, 1.3258, 1.9346, -0.8884, -1.6172, 1.0867, -3.0494, 0.04, 0.04])
            self.agent.reset(target_qpos)
            #self.agent.reset()

    def _get_obs_extra(self, info: Dict):
        """获取额外观测信息 - 简化版本"""
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            grasped_count=info["grasped_count"],
        )
        
        if "state" in self.obs_mode:
            if hasattr(self, 'target_object') and self.target_object is not None:
                obs.update(
                    target_obj_pose=self.target_object.pose.raw_pose,
                    tcp_to_obj_pos=self.target_object.pose.p - self.agent.tcp.pose.p,
                )
            else:
                # 提供零值作为占位符
                batch_size = self.num_envs
                zero_pose = torch.zeros((batch_size, 7), device=self.device)
                zero_pos = torch.zeros((batch_size, 3), device=self.device)
                obs.update(
                    target_obj_pose=zero_pose,
                    tcp_to_obj_pos=zero_pos,
                )
            
            # 添加任务进度信息
            obs.update(
                progress_ratio=torch.tensor([
                    self.grasped_objects_count / self.total_objects_per_env
                ] * self.num_envs, device=self.device),
            )
        
        return obs


    def evaluate(self):
        """评估任务完成情况"""
        success = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_grasped = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_robot_static = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        
        if hasattr(self, 'target_object') and self.target_object is not None:
            # 检查当前目标物体是否被抓取
            is_grasped = self.agent.is_grasping(self.target_object)
            
            # 检查机器人是否静止
            is_robot_static = self.agent.is_static(0.2)
        
        # 成功条件：所有物体都被成功抓取（简化版本）
        # 这里我们检查已抓取物体数量是否等于总数
        success = torch.tensor([
            self.grasped_objects_count >= self.total_objects_per_env
        ] * self.num_envs, device=self.device, dtype=bool)
        
        return {
            "success": success,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "grasped_count": torch.tensor([self.grasped_objects_count] * self.num_envs, device=self.device),
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
        """计算奖励 - 只在抓取结束时给奖励"""
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # 第一优先级：抓取奖励（稀疏奖励，只在成功时给予）
        grasp_reward = self._compute_grasp_reward(info)
        
        # 第二优先级：位移惩罚（减轻惩罚强度）
        displacement_penalty = self._compute_displacement_penalty()
        
        # 第三优先级：轻微时间惩罚（避免过度负奖励）
        time_penalty = 0.001  # 非常小的时间惩罚
        
        # 组合奖励
        reward = grasp_reward - displacement_penalty * 0.1 - time_penalty  # 减轻惩罚权重
        
        # 完成所有物体的大奖励
        if info.get("success", False).any():
            completion_bonus = 20.0  # 增加完成奖励
            reward[info["success"]] += completion_bonus
        
        return reward
    
    def _compute_grasp_reward(self, info: Dict):
        """计算抓取奖励 - 第一优先级，只在成功抓取时给奖励"""
        reward = torch.zeros(self.num_envs, device=self.device)
        
        if hasattr(self, 'target_object') and self.target_object is not None:
            # 检查抓取状态
            is_grasped = info.get("is_grasped", torch.zeros_like(reward, dtype=bool))
            
            # 检查是否有新的抓取（与上一步比较）
            if not hasattr(self, '_prev_grasped'):
                self._prev_grasped = torch.zeros_like(is_grasped)
            
            # 只有新抓取成功时才给奖励
            new_grasp = is_grasped & (~self._prev_grasped)
            if new_grasp.any():
                self.grasped_objects_count += new_grasp.sum().item()
                reward += new_grasp.float() * 10.0  # 成功抓取给10分大奖励
                print(f"🎉 成功抓取! 当前已抓取数量: {self.grasped_objects_count}")
                
                # 选择下一个目标物体
                self._update_target_object()
            
            self._prev_grasped = is_grasped
            
        return reward
    
    def _update_target_object(self):
        """更新目标物体 - 选择下一个未抓取的物体"""
        if not hasattr(self, 'completed_objects'):
            self.completed_objects = set()
        
        # 将当前目标物体标记为已完成
        if hasattr(self, 'target_object_indices') and self.target_object_indices:
            for env_idx, target_idx in enumerate(self.target_object_indices):
                if env_idx < len(self.target_object_indices):
                    self.completed_objects.add(target_idx)
        
        # 重新选择目标物体
        self._sample_target_objects()
    
    def _compute_displacement_penalty(self):
        """计算位移惩罚 - 第二优先级"""
        # 简化版本：计算所有非目标物体的位移
        total_displacement = self._calculate_other_objects_displacement()
        # 限制惩罚在合理范围内
        displacement_penalty = torch.clamp(total_displacement * 0.5, 0, 2.0)
        return displacement_penalty
    
    
   

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


    # AnyGrasp功能已移除 - 简化为基本抓取环境
    
    # 相机观测功能简化 - 专注于基本抓取任务
    
    # 以下所有AnyGrasp相关功能已移除，专注于基本抓取任务
    