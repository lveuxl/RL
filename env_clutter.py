import os
import sys
from typing import Any, Dict, List, Union
import numpy as np
import sapien
import torch
import cv2
import random
import gym

import mani_skill.envs.utils.randomization as randomization
from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda

# 直接导入PandaSuction类
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents', 'robots', 'panda'))
from panda_suction import PandaSuction

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

# 不再导入运动规划模块，改为直接控制
# from motion_planner import GraspPlanner, SimpleMotionPlanner, TrajectoryPlayer


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
    
    **控制模式:**
    - 使用pd_ee_pose控制模式，直接控制末端执行器位姿
    - 离散动作选择目标物体，连续动作控制机械臂移动
    """
    
    SUPPORTED_REWARD_MODES = ["dense", "sparse"]
    SUPPORTED_ROBOTS = ["panda", "fetch", "panda_stick", "panda_suction"]  # 添加panda_suction支持
    agent: Union[Panda, Fetch, PandaSuction]  # 添加类型注解
    
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
    
    # 固定目标位置 - 托盘右侧合适位置
    FIXED_GOAL_POSITION = [0.35, 0.0, 0.02]  # 托盘右侧，机器人可达范围内
    
    # 新增：状态机参数
    GRASP_HEIGHT_OFFSET = 0.08  # 增加到0.08，给更多抓取空间
    PLACE_HEIGHT_OFFSET = 0.10  # 增加到0.10
    SUCTION_DISTANCE_THRESHOLD = 0.10  # 增加到0.10，更宽松的激活条件
    POSITION_TOLERANCE = 0.08  # 增加到0.08，更宽松的位置控制
    
    # 调整工作空间限制，使其更适合新的机器人位置
    WORKSPACE_LIMITS = {
        'x_min': -0.15, 'x_max': 0.45,   # 扩大x范围，提高边界可达性
        'y_min': -0.25, 'y_max': 0.25,   # 稍微扩大y范围
        'z_min': 0.02, 'z_max': 0.45    # 稍微提高z上限
    }

    def __init__(
        self,
        *args,
        robot_uids="panda_suction",  # 默认使用panda_suction
        robot_init_qpos_noise=0.02,
        num_envs=1,
        use_discrete_action=True,  # 新增参数：是否使用离散动作空间
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_discrete_action = use_discrete_action
        
        # 初始化状态机变量
        self.fsm_state = "idle"  # 状态机状态
        self.current_target_object = None  # 当前目标物体
        self.target_position = None  # 当前目标位置
        self.suction_active = False  # 吸盘状态
        self.phase_step_count = 0  # 当前阶段步数计数
        self.max_phase_steps = 100  # 每个阶段的最大步数
        
        # 确保所有参数正确传递给父类
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            **kwargs,
        )

    def _get_discrete_action_space(self):
        """获取离散动作空间"""
        if self.use_discrete_action:
            # 离散动作空间：每个动作对应选择一个物体进行抓取
            num_objects = len(self.BOX_OBJECTS) * self.num_objects_per_type
            return gym.spaces.Discrete(num_objects)
        else:
            # 连续动作空间：使用原始的机械臂控制
            return None

    @property
    def discrete_action_space(self):
        """离散动作空间属性"""
        return self._get_discrete_action_space()

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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
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
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
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
        # 原来：[-0.615, 0, 0]
        # 托盘在 [0.1, 0.0, 0.02]，机器人应该在托盘后方稍近的位置
        super()._load_agent(options, sapien.Pose(p=[-0.15, 0, 0]))

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
                    
                    # 在托盘内随机生成位置 - 传递堆叠层级
                    x, y, z = self._generate_object_position_in_tray(i % 3)  # 最多3层堆叠
                    
                    # 确保位置在合理范围内
                    x = np.clip(x, 0.1 - 0.15, 0.1 + 0.15)  # 限制在托盘中心±0.15m
                    y = np.clip(y, -0.15, 0.15)
                    z = max(z, 0.03)  # 确保最小高度
                    
                    # 随机旋转
                    rotation = randomization.random_quaternions(1)[0]
                    
                    # 设置初始位姿（在build之前设置）
                    builder.initial_pose = sapien.Pose(p=[x, y, z], q=rotation)
                    builder.set_scene_idxs([env_idx])
                    
                    # 构建物体
                    obj = builder.build(name=f"env_{env_idx}_{obj_type}_{i}")
                    
                    env_objects.append(obj)
                    env_selectable.append(obj)
                    env_info.append({
                        'type': obj_type,
                        'index': i,
                        'env_idx': env_idx,
                        'name': f"env_{env_idx}_{obj_type}_{i}",
                        'initial_pos': [x, y, z]  # 记录初始位置
                    })
            
            self.all_objects.extend(env_objects)
            self.selectable_objects.append(env_selectable)
            self.object_info.extend(env_info)
        
        # 创建固定目标区域
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        for env_idx in range(self.num_envs):
            # 使用固定的目标位置
            self.goal_pos[env_idx] = torch.tensor(self.FIXED_GOAL_POSITION, device=self.device)
        
        # 创建目标站点（绿色球体标记）
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 0.8],  # 绿色半透明
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=self.FIXED_GOAL_POSITION)  # 使用固定位置
        )
        
        # 合并所有物体以便批量操作
        if self.all_objects:
            self.merged_objects = Actor.merge(self.all_objects, name="all_objects")

    def _load_tray(self):
        """加载托盘URDF文件"""
        # 获取托盘URDF文件路径
        tray_urdf_path = os.path.join(os.getcwd(), "assets", "tray", "traybox.urdf")
        
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
        
        # 使用 actor_builders 创建托盘
        actor_builders = parsed_result.get("actor_builders", [])
        
        self.trays = []
        
        # 使用 actor_builders 加载托盘
        print("使用actor builders加载托盘")
        for env_idx in range(self.num_envs):
            builder = actor_builders[0]
            # 设置托盘位置 (放在桌面上，机器人前方)
            tray_position = [0.1, 0.0, 0.02]  # 桌面高度加上托盘底部厚度
            builder.initial_pose = sapien.Pose(p=tray_position)
            builder.set_scene_idxs([env_idx])
            
            # 使用 build_static 创建静态托盘，确保不会移动
            tray = builder.build_static(name=f"tray_{env_idx}")
            self.trays.append(tray)
        
        # 合并所有托盘
        if self.trays:
            self.merged_trays = Actor.merge(self.trays, name="all_trays")
        
        print(f"成功加载托盘，共 {len(self.trays)} 个")

    def _generate_object_position_in_tray(self, stack_level=0):
        """在托盘内生成物体位置"""
        # 托盘中心位置
        tray_center_x = 0.1
        tray_center_y = 0.0
        tray_bottom_z = 0.02 + 0.01  # 托盘底部 + 小偏移
        
        # 减小生成区域，确保物体在托盘内
        safe_spawn_area_x = 0.15  # 从0.23减小到0.15
        safe_spawn_area_y = 0.15  # 从0.23减小到0.15
        
        # 在托盘内随机生成xy位置
        x = tray_center_x + random.uniform(-safe_spawn_area_x, safe_spawn_area_x)
        y = tray_center_y + random.uniform(-safe_spawn_area_y, safe_spawn_area_y)
        
        # 堆叠高度 - 确保z坐标为正值
        z = max(tray_bottom_z + stack_level * 0.03, 0.03)  # 确保最小高度为3cm
        
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
        
        # 遍历所有物体信息，找到属于指定环境的物体
        for i, obj_info in enumerate(self.object_info):
            if isinstance(obj_info, dict) and obj_info.get('env_idx') == env_idx:
                # 基于物体高度和周围物体数量的简单估算
                exposed_area = max(0.1, 1.0 - (i % 5) * 0.1)  # 越高的物体暴露面积越大
                obj_info['exposed_area'] = exposed_area

    def step(self, action):
        """重写step方法，使用简化的控制逻辑"""
        if self.use_discrete_action and isinstance(action, (int, np.integer)):
            # 离散动作模式：简单的目标设置
            return self._step_discrete_simple(action)
        else:
            # 连续动作模式：直接控制
            return super().step(action)

    def _step_discrete_simple(self, discrete_action: int):
        """简化的离散动作处理 - 不使用状态机"""
        try:
            # 获取目标物体
            env_idx = 0  # 单环境
            target_object = self._get_object_by_action(discrete_action, env_idx)
            
            if target_object is None:
                # 如果没有找到目标物体，保持当前位置
                tcp_pos = self.agent.tcp.pose.p[0].cpu().numpy()
                low_level_action = np.concatenate([tcp_pos, np.array([0, 0, 0])])
            else:
                # 直接计算目标位置（物体上方）
                obj_pos = target_object.pose.p[0].cpu().numpy()
                
                # 检查物体位置是否合理
                if (obj_pos[0] < -1.0 or obj_pos[0] > 1.0 or 
                    obj_pos[1] < -1.0 or obj_pos[1] > 1.0 or 
                    obj_pos[2] < -1.0 or obj_pos[2] > 1.0):
                    print(f"物体位置异常: {obj_pos}，使用默认位置")
                    obj_pos = np.array([0.1, 0.0, 0.1])  # 使用安全的默认位置
                
                # 获取当前TCP位置
                tcp_pos = self.agent.tcp.pose.p[0].cpu().numpy()
                distance_to_obj = np.linalg.norm(tcp_pos - obj_pos)
                
                # 改进的目标位置计算
                if distance_to_obj > 0.15:
                    # 距离较远时，移动到物体上方
                    target_pos = obj_pos + np.array([0, 0, self.GRASP_HEIGHT_OFFSET])
                else:
                    # 距离较近时，直接移动到物体位置
                    target_pos = obj_pos + np.array([0, 0, 0.02])  # 接近物体表面
                
                target_pos = self._clamp_to_workspace(target_pos)
                
                # 构造6维动作 (x, y, z, rx, ry, rz)
                # 使用更合理的旋转角度：垂直向下
                low_level_action = np.concatenate([target_pos, np.array([0, np.pi, 0])])
                
                # 改进的吸盘控制逻辑
                if distance_to_obj < self.SUCTION_DISTANCE_THRESHOLD:
                    if not self.suction_active:
                        success = self._activate_suction_simple(target_object)
                        if success:
                            print(f"吸盘激活成功，距离: {distance_to_obj:.3f}m")
                        else:
                            print(f"吸盘激活失败，距离: {distance_to_obj:.3f}m")
                elif distance_to_obj > 0.20 and self.suction_active:
                    self._deactivate_suction_simple()
                    print(f"距离过远，关闭吸盘，距离: {distance_to_obj:.3f}m")
                
                # 调试信息（每10步打印一次）
                if hasattr(self, '_debug_step_count'):
                    self._debug_step_count += 1
                else:
                    self._debug_step_count = 0
                
                if self._debug_step_count % 10 == 0:
                    print(f"动作{discrete_action}: TCP={tcp_pos[:2]}, 物体={obj_pos[:2]}, 距离={distance_to_obj:.3f}m")
            
            # 执行一步物理仿真
            obs, reward, done, truncated, info = super().step(low_level_action)
            
            # 计算奖励
            reward = self.compute_dense_reward(obs, torch.tensor([discrete_action]), info)
            
            return obs, reward, done, truncated, info
            
        except Exception as e:
            print(f"简化控制执行失败: {e}")
            return self._get_failed_step_result()

    def _activate_suction_simple(self, target_object):
        """简化的吸盘激活"""
        if hasattr(self.agent, 'activate_suction') and target_object is not None:
            try:
                # 检查接触距离
                tcp_pos = self.agent.tcp.pose.p[0].cpu().numpy()
                obj_pos = target_object.pose.p[0].cpu().numpy()
                distance = np.linalg.norm(tcp_pos - obj_pos)
                
                if distance <= self.SUCTION_DISTANCE_THRESHOLD:
                    success = self.agent.activate_suction(target_object)
                    if success:
                        self.suction_active = True
                        self.current_target_object = target_object
                        print(f"吸盘激活成功: {target_object.name}")
                        return True
                    else:
                        print(f"吸盘激活失败: 距离={distance:.3f}m")
                        return False
                else:
                    print(f"距离过远，无法激活吸盘: {distance:.3f}m > {self.SUCTION_DISTANCE_THRESHOLD}")
                    return False
            except Exception as e:
                print(f"吸盘激活失败: {e}")
                return False
        return False

    def _deactivate_suction_simple(self):
        """简化的吸盘关闭"""
        if hasattr(self.agent, 'deactivate_suction') and self.suction_active:
            try:
                success = self.agent.deactivate_suction()
                if success:
                    self.suction_active = False
                    self.current_target_object = None
                    print("吸盘关闭成功")
                    return True
                else:
                    print("吸盘关闭失败")
                    return False
            except Exception as e:
                print(f"吸盘关闭失败: {e}")
                return False
        return False

    def _clamp_to_workspace(self, position):
        """将位置限制在工作空间内"""
        x, y, z = position
        x = np.clip(x, self.WORKSPACE_LIMITS['x_min'], self.WORKSPACE_LIMITS['x_max'])
        y = np.clip(y, self.WORKSPACE_LIMITS['y_min'], self.WORKSPACE_LIMITS['y_max'])
        z = np.clip(z, self.WORKSPACE_LIMITS['z_min'], self.WORKSPACE_LIMITS['z_max'])
        return np.array([x, y, z])

    # 移除复杂的状态机方法，保留简化版本
    def _step_with_fsm(self, discrete_action: int):
        """废弃的状态机方法 - 重定向到简化版本"""
        return self._step_discrete_simple(discrete_action)

    def _handle_new_discrete_action(self, action: int):
        """废弃的方法"""
        pass

    def _compute_low_level_action(self):
        """废弃的方法"""
        pass

    def _update_fsm(self, obs):
        """废弃的方法"""
        pass

    def _reset_fsm(self):
        """简化的重置方法"""
        self.suction_active = False
        self.current_target_object = None

    def _handle_suction_control(self, action):
        """处理吸盘控制 - 用于连续动作模式"""
        if hasattr(self.agent, 'suction') and self.current_target_object is not None:
            # 获取末端执行器位置
            tcp_pose = self.agent.tcp.pose
            tcp_pos = tcp_pose.p[0].cpu().numpy()  # 假设单环境
            
            # 获取目标物体位置
            obj_pos = self.current_target_object.pose.p.cpu().numpy()
            
            # 计算距离
            distance = np.linalg.norm(tcp_pos - obj_pos)
            
            # 如果距离小于阈值，激活吸盘
            if distance < self.SUCTION_DISTANCE_THRESHOLD:
                if not self.suction_active:
                    self._activate_suction()
            else:
                if self.suction_active:
                    self._deactivate_suction()

    def _get_object_by_action(self, action: int, env_idx: int):
        """根据动作索引获取对应的物体"""
        try:
            # 计算物体类型和索引
            objects_per_type = self.num_objects_per_type
            obj_type_idx = action // objects_per_type
            obj_idx = action % objects_per_type
            
            if obj_type_idx >= len(self.BOX_OBJECTS):
                return None
            
            obj_type = self.BOX_OBJECTS[obj_type_idx]
            
            # 在所有物体中查找匹配的物体
            for obj in self.all_objects:
                if hasattr(obj, 'name') and obj.name:
                    # 解析物体名称：env_{env_idx}_{obj_type}_{obj_idx}
                    name_parts = obj.name.split('_')
                    if len(name_parts) >= 4:
                        obj_env_idx = int(name_parts[1])
                        obj_type_name = '_'.join(name_parts[2:-1])  # 处理包含下划线的物体名称
                        obj_instance_idx = int(name_parts[-1])
                        
                        if (obj_env_idx == env_idx and 
                            obj_type_name == obj_type and 
                            obj_instance_idx == obj_idx):
                            return obj
            
            return None
            
        except Exception as e:
            print(f"获取物体时出错: {e}")
            return None

    def _get_failed_step_result(self):
        """获取失败的step结果"""
        obs = self.get_obs()
        reward = -1.0  # 失败惩罚
        done = False
        truncated = False
        info = {"success": False, "failure_reason": "action_execution_failed"}
        
        return obs, reward, done, truncated, info

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            
            # 简化的重置
            self._reset_fsm()
            
            # 重置托盘位置
            if hasattr(self, 'merged_trays'):
                if b == self.num_envs:
                    self.merged_trays.pose = self.merged_trays.initial_pose
                else:
                    mask = torch.isin(self.merged_trays._scene_idxs, env_idx)
                    self.merged_trays.pose = self.merged_trays.initial_pose[mask]
            
            # 重置物体到初始位置并验证位置
            if hasattr(self, 'merged_objects'):
                if b == self.num_envs:
                    # 验证并修复物体位置
                    initial_poses = self.merged_objects.initial_pose
                    for i in range(len(initial_poses)):
                        pos = initial_poses[i].p
                        # 检查位置是否合理
                        if (pos[0] < -0.5 or pos[0] > 0.5 or 
                            pos[1] < -0.5 or pos[1] > 0.5 or 
                            pos[2] < 0.01 or pos[2] > 0.5):
                            # 重新生成合理位置
                            new_x = np.random.uniform(-0.05, 0.25)  # 托盘区域
                            new_y = np.random.uniform(-0.1, 0.1)
                            new_z = np.random.uniform(0.03, 0.12)
                            initial_poses[i].p = torch.tensor([new_x, new_y, new_z], device=self.device)
                            print(f"修复物体{i}位置: [{new_x:.3f}, {new_y:.3f}, {new_z:.3f}]")
                    
                    self.merged_objects.pose = initial_poses
                else:
                    mask = torch.isin(self.merged_objects._scene_idxs, env_idx)
                    self.merged_objects.pose = self.merged_objects.initial_pose[mask]
            
            # 设置固定目标位置
            goal_pos = torch.zeros((b, 3), device=self.device)
            for i in range(b):
                goal_pos[i] = torch.tensor(self.FIXED_GOAL_POSITION, device=self.device)
            
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
            
            # 如果是吸盘机器人，确保吸盘处于关闭状态
            if hasattr(self.agent, 'suction') and hasattr(self.agent.suction, 'set_suction'):
                self.agent.suction.set_suction(False)
                self.suction_active = False

    def _get_obs_extra(self, info: Dict):
        """获取额外观测信息 - 简化版本"""
        # 获取批次大小
        batch_size = self.num_envs
        
        # 基础观测信息
        obs = dict()
        
        # is_grasped 已经是正确的批次维度
        obs["is_grasped"] = info["is_grasped"]
        
        # tcp_pose
        tcp_pose = self.agent.tcp.pose.raw_pose
        if tcp_pose.shape[0] != batch_size:
            tcp_pose = tcp_pose.repeat(batch_size, 1)
        obs["tcp_pose"] = tcp_pose
        
        # goal_pos
        goal_pos = self.goal_site.pose.p
        if goal_pos.shape[0] != batch_size:
            goal_pos = goal_pos.repeat(batch_size, 1)
        obs["goal_pos"] = goal_pos
        
        # 简化的状态信息
        obs["suction_active"] = torch.full((batch_size,), self.suction_active, device=self.device, dtype=torch.bool)
        
        if "state" in self.obs_mode:
            if hasattr(self, 'target_object') and self.target_object is not None:
                # target_obj_pose
                target_obj_pose = self.target_object.pose.raw_pose
                if target_obj_pose.shape[0] != batch_size:
                    target_obj_pose = target_obj_pose.repeat(batch_size, 1)
                obs["target_obj_pose"] = target_obj_pose
                
                # tcp_to_obj_pos
                tcp_to_obj_pos = self.target_object.pose.p - self.agent.tcp.pose.p
                if tcp_to_obj_pos.shape[0] != batch_size:
                    tcp_to_obj_pos = tcp_to_obj_pos.repeat(batch_size, 1)
                obs["tcp_to_obj_pos"] = tcp_to_obj_pos
                
                # obj_to_goal_pos
                obj_to_goal_pos = self.goal_site.pose.p - self.target_object.pose.p
                if obj_to_goal_pos.shape[0] != batch_size:
                    obj_to_goal_pos = obj_to_goal_pos.repeat(batch_size, 1)
                obs["obj_to_goal_pos"] = obj_to_goal_pos
            else:
                # 如果没有目标物体，提供零张量
                obs["target_obj_pose"] = torch.zeros((batch_size, 7), device=self.device)
                obs["tcp_to_obj_pos"] = torch.zeros((batch_size, 3), device=self.device)
                obs["obj_to_goal_pos"] = torch.zeros((batch_size, 3), device=self.device)
            
            # 标量观测信息
            num_objects = torch.full((batch_size,), len(self.all_objects), 
                                   device=self.device, dtype=torch.int64)
            obs["num_objects"] = num_objects
        
        return obs

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
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :], axis=1)  # 吸盘版本包含所有关节
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