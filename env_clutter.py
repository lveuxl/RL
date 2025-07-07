import os
import math
import random
from typing import Dict, List, Union, Optional, Tuple

import numpy as np
import sapien
import torch
import gymnasium as gym
from gymnasium import spaces

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

from config import ENV_CONFIG, CATEGORY_MAP, OBJECT_PROPERTIES


class ComplexStackingClutterEnv(BaseEnv):
    """
    复杂堆叠场景的杂乱物体抓取环境
    基于ManiSkill的pick_clutter_ycb环境改进，集成了：
    1. 暴露度计算
    2. 智能物体选择
    3. 复杂堆叠场景支持
    4. 丰富的状态特征
    """
    
    SUPPORTED_REWARD_MODES = ["sparse", "dense"]
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        max_objects=16,
        reward_mode="dense",
        enable_intelligent_selection=True,
        **kwargs,
    ):
        # 环境配置
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.max_objects = max_objects
        self.reward_mode = reward_mode
        self.enable_intelligent_selection = enable_intelligent_selection
        
        # 物体配置
        self.category_map = CATEGORY_MAP
        self.object_properties = OBJECT_PROPERTIES
        
        # 暴露度计算配置
        self.exposure_config = ENV_CONFIG["exposure_config"]
        self.ray_directions = self.exposure_config["ray_directions"]
        self.ray_length = self.exposure_config["ray_length"]
        
        # 智能选择配置
        self.selection_config = ENV_CONFIG["selection_params"]
        
        # 奖励配置
        self.reward_config = ENV_CONFIG["reward_config"]
        
        # 重新配置频率
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )
        
        # 初始化环境状态
        self.current_objects: List[Dict] = []
        self.object_exposures: Dict[int, float] = {}
        self.object_graspabilities: Dict[int, float] = {}
        self.failure_counts: Dict[int, int] = {}
        self.success_count = 0
        self.total_attempts = 0
        
        # 观测空间配置
        self.features_per_object = 10  # 位置(4) + 类别(4) + 暴露度(1) + 失败次数(1)
        self.global_features = 3  # 步数进度 + 剩余物体数 + 成功率
        
        # 设置观测和动作空间
        self._setup_observation_space()
        self._setup_action_space()
    
    def _setup_observation_space(self):
        """设置观测空间"""
        obs_dim = self.max_objects * self.features_per_object + self.global_features
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
    
    def _setup_action_space(self):
        """设置动作空间 - 离散动作，选择要抓取的物体"""
        self.single_action_space = spaces.Discrete(self.max_objects)
    
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
        """配置传感器"""
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
        """配置人类观察用的渲染相机"""
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
        """加载机器人代理"""
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))
    
    def _load_scene(self, options: dict):
        """加载场景"""
        # 创建桌子场景
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        
        # 生成复杂堆叠物体配置
        self._generate_complex_stacking_scene()
        
        # 创建目标位置指示器
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.01,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)
        
        # 初始化物体选择
        self._initialize_object_selection()
    
    def _generate_complex_stacking_scene(self):
        """生成复杂的堆叠场景"""
        all_objects = []
        self.current_objects = []
        
        # 场景配置
        scene_config = ENV_CONFIG["scene_config"]
        workspace_bounds = scene_config["workspace_bounds"]
        
        # 随机生成物体数量
        num_objects = random.randint(8, self.max_objects)
        
        for i in range(num_objects):
            # 随机选择物体类型
            obj_category = random.choice(list(self.category_map.keys()))
            
            # 生成物体位置（支持堆叠）
            if i < num_objects // 2:
                # 底层物体
                x = random.uniform(workspace_bounds["x_min"], workspace_bounds["x_max"])
                y = random.uniform(workspace_bounds["y_min"], workspace_bounds["y_max"])
                z = 0.82  # 桌面高度
            else:
                # 可能堆叠的物体
                if random.random() < 0.6:  # 60%概率堆叠
                    # 选择一个底层物体进行堆叠
                    base_obj = random.choice(self.current_objects[:num_objects//2])
                    x = base_obj["position"][0] + random.uniform(-0.05, 0.05)
                    y = base_obj["position"][1] + random.uniform(-0.05, 0.05)
                    z = base_obj["position"][2] + 0.1  # 堆叠高度
                else:
                    # 独立放置
                    x = random.uniform(workspace_bounds["x_min"], workspace_bounds["x_max"])
                    y = random.uniform(workspace_bounds["y_min"], workspace_bounds["y_max"])
                    z = 0.82
            
            # 创建物体
            builder = self._create_object_builder(obj_category)
            pose = sapien.Pose(p=[x, y, z])
            builder.initial_pose = pose
            builder.set_scene_idxs([0])  # 单环境
            
            obj_actor = builder.build(name=f"object_{i}_{obj_category}")
            all_objects.append(obj_actor)
            
            # 记录物体信息
            obj_info = {
                "id": i,
                "actor": obj_actor,
                "category": obj_category,
                "position": [x, y, z],
                "dimensions": self.object_properties[obj_category]["dimensions"],
                "mass": self.object_properties[obj_category]["mass"],
                "exposure": 0.0,
                "graspability": 0.0,
                "failure_count": 0,
                "is_selected": False
            }
            self.current_objects.append(obj_info)
        
        # 合并所有物体
        self.all_objects = Actor.merge(all_objects, name="all_objects")
        
        print(f"生成了 {len(self.current_objects)} 个物体的复杂堆叠场景")
    
    def _create_object_builder(self, category: str) -> ActorBuilder:
        """创建物体构建器"""
        # 使用YCB物体集
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{category}")
        return builder
    
    def _initialize_object_selection(self):
        """初始化物体选择系统"""
        # 计算所有物体的暴露度
        self._calculate_all_exposures()
        
        # 计算所有物体的可抓取性
        self._calculate_all_graspabilities()
        
        # 如果启用智能选择，选择最佳目标
        if self.enable_intelligent_selection:
            self._select_optimal_target()
    
    def _calculate_all_exposures(self):
        """计算所有物体的暴露度"""
        self.object_exposures = {}
        
        for obj_info in self.current_objects:
            obj_id = obj_info["id"]
            actor = obj_info["actor"]
            
            # 计算暴露度
            exposure = self._calculate_object_exposure(actor, obj_info)
            self.object_exposures[obj_id] = exposure
            obj_info["exposure"] = exposure
    
    def _calculate_object_exposure(self, actor: Actor, obj_info: Dict) -> float:
        """
        计算单个物体的暴露度
        使用射线检测来评估物体的可见性和可达性
        """
        try:
            # 获取物体位置
            pose = actor.pose
            if hasattr(pose, 'p'):
                obj_pos = pose.p.cpu().numpy() if hasattr(pose.p, 'cpu') else pose.p
            else:
                obj_pos = obj_info["position"]
            
            # 射线检测方向（主要检测顶部和侧面）
            ray_directions = [
                [0, 0, 1],    # 顶部
                [1, 0, 0],    # 右侧
                [-1, 0, 0],   # 左侧
                [0, 1, 0],    # 前方
                [0, -1, 0],   # 后方
            ]
            
            total_rays = 0
            hit_rays = 0
            
            for direction in ray_directions:
                # 从物体位置向外发射射线
                ray_start = obj_pos + np.array(direction) * 0.01  # 稍微偏移
                ray_end = obj_pos + np.array(direction) * self.ray_length
                
                # 执行射线检测
                # 注意：这里需要根据实际的SAPIEN API进行调整
                # 暂时使用简化的暴露度计算
                total_rays += 1
                
                # 基于物体高度的简化暴露度计算
                height_factor = max(0, obj_pos[2] - 0.8) / 0.2  # 高度越高暴露度越大
                
                # 基于与其他物体距离的暴露度计算
                min_distance = float('inf')
                for other_obj in self.current_objects:
                    if other_obj["id"] != obj_info["id"]:
                        other_pos = other_obj["position"]
                        distance = np.linalg.norm(np.array(obj_pos[:2]) - np.array(other_pos[:2]))
                        min_distance = min(min_distance, distance)
                
                distance_factor = min(1.0, min_distance / 0.15) if min_distance != float('inf') else 1.0
                
                # 综合暴露度
                if height_factor > 0.3 and distance_factor > 0.5:
                    hit_rays += 1
            
            exposure = hit_rays / total_rays if total_rays > 0 else 0.0
            return max(0.0, min(1.0, exposure))
            
        except Exception as e:
            print(f"计算暴露度时出错: {e}")
            return 0.1  # 默认暴露度
    
    def _calculate_all_graspabilities(self):
        """计算所有物体的可抓取性"""
        self.object_graspabilities = {}
        
        for obj_info in self.current_objects:
            obj_id = obj_info["id"]
            category = obj_info["category"]
            exposure = obj_info["exposure"]
            failure_count = self.failure_counts.get(obj_id, 0)
            
            # 基础成功率
            base_success_rate = self.object_properties[category]["base_success_rate"]
            
            # 暴露度加成
            exposure_bonus = exposure * self.selection_config["exposure_weight"]
            
            # 失败惩罚
            failure_penalty = failure_count * self.selection_config["failure_penalty"]
            
            # 计算总可抓取性
            graspability = base_success_rate + exposure_bonus - failure_penalty
            graspability = max(0.1, min(0.95, graspability))
            
            self.object_graspabilities[obj_id] = graspability
            obj_info["graspability"] = graspability
    
    def _select_optimal_target(self):
        """选择最优目标物体"""
        if not self.current_objects:
            return
        
        # 计算每个物体的综合评分
        best_score = -1
        best_obj = None
        
        for obj_info in self.current_objects:
            if obj_info.get("is_selected", False):
                continue
            
            # 综合评分
            category_score = self.object_properties[obj_info["category"]]["base_success_rate"]
            exposure_score = obj_info["exposure"]
            graspability_score = obj_info["graspability"]
            
            # 权重计算
            total_score = (
                category_score * self.selection_config["category_weight"] +
                exposure_score * self.selection_config["exposure_weight"] +
                graspability_score * self.selection_config["graspability_weight"]
            )
            
            if total_score > best_score:
                best_score = total_score
                best_obj = obj_info
        
        if best_obj:
            best_obj["is_selected"] = True
            print(f"选择最优目标: {best_obj['category']}, 评分: {best_score:.3f}")
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化episode"""
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            
            # 设置目标位置
            goal_pos = torch.rand(size=(b, 3)) * torch.tensor([0.3, 0.5, 0.1]) + torch.tensor([-0.15, -0.25, 0.35])
            self.goal_pos = goal_pos
            self.goal_site.set_pose(Pose.create_from_pq(self.goal_pos))
            
            # 重置物体到初始位置
            if hasattr(self, 'all_objects'):
                if b == self.num_envs:
                    self.all_objects.pose = self.all_objects.initial_pose
                else:
                    mask = torch.isin(self.all_objects._scene_idxs, env_idx)
                    self.all_objects.pose = self.all_objects.initial_pose[mask]
            
            # 重置环境状态
            self.success_count = 0
            self.total_attempts = 0
            self.failure_counts = {}
            
            # 重新初始化物体选择
            self._initialize_object_selection()
    
    def _get_obs_extra(self, info: Dict):
        """获取额外观测信息"""
        obs_extra = {}
        
        # 物体状态特征
        obs_features = []
        
        # 为每个物体槽位生成特征
        for i in range(self.max_objects):
            if i < len(self.current_objects):
                obj_info = self.current_objects[i]
                
                # 位置特征 (4维)
                pos = obj_info["position"]
                scene_center = [0.5, -0.2]
                robot_pos = [-0.615, 0, 0]
                
                pos_features = [
                    pos[0] - scene_center[0],  # 相对场景中心X
                    pos[1] - scene_center[1],  # 相对场景中心Y
                    pos[2] - 0.8,              # 相对桌面高度
                    np.sqrt((pos[0] - robot_pos[0])**2 + (pos[1] - robot_pos[1])**2)  # 到机器人距离
                ]
                
                # 类别特征 (4维 one-hot)
                category_features = [0.0] * len(self.category_map)
                category_idx = self.category_map[obj_info["category"]] - 1
                category_features[category_idx] = 1.0
                
                # 暴露度特征 (1维)
                exposure = obj_info.get("exposure", 0.0)
                
                # 失败次数特征 (1维)
                failure_count = self.failure_counts.get(obj_info["id"], 0)
                failure_normalized = np.tanh(failure_count / 3.0)
                
                # 合并特征
                obj_features = pos_features + category_features + [exposure, failure_normalized]
                
            else:
                # 空槽位，填充零
                obj_features = [0.0] * self.features_per_object
            
            obs_features.extend(obj_features)
        
        # 全局特征
        step_progress = min(self.total_attempts / 16, 1.0)  # 假设最大16步
        remaining_ratio = len(self.current_objects) / self.max_objects
        success_rate = self.success_count / max(self.total_attempts, 1)
        
        global_features = [
            np.sqrt(step_progress),
            remaining_ratio,
            min(success_rate, 1.0)
        ]
        
        obs_features.extend(global_features)
        
        # 转换为numpy数组
        obs_extra["state"] = np.array(obs_features, dtype=np.float32)
        
        return obs_extra
    
    def step(self, action):
        """执行一步动作"""
        # 动作验证
        if isinstance(action, (np.ndarray, torch.Tensor)):
            action = int(action.item())
        
        # 检查动作有效性
        if action >= len(self.current_objects):
            # 无效动作，给予惩罚
            reward = self.reward_config["failure_penalty"]
            done = len(self.current_objects) == 0
            info = {"success": False, "invalid_action": True}
            return self._get_obs(), reward, done, False, info
        
        # 获取目标物体
        target_obj = self.current_objects[action]
        
        # 执行抓取
        success, displacement = self._execute_grasp(target_obj)
        
        # 更新统计
        self.total_attempts += 1
        if success:
            self.success_count += 1
            # 移除成功抓取的物体
            self.current_objects.pop(action)
        else:
            # 更新失败计数
            obj_id = target_obj["id"]
            self.failure_counts[obj_id] = self.failure_counts.get(obj_id, 0) + 1
        
        # 计算奖励
        reward = self._calculate_reward(success, target_obj, displacement)
        
        # 检查是否结束
        done = len(self.current_objects) == 0 or self.total_attempts >= 16
        
        # 更新暴露度和可抓取性
        if not done:
            self._calculate_all_exposures()
            self._calculate_all_graspabilities()
        
        # 构建info
        info = {
            "success": success,
            "remaining_objects": len(self.current_objects),
            "success_rate": self.success_count / self.total_attempts,
            "displacement": displacement,
            "target_category": target_obj["category"],
            "target_exposure": target_obj.get("exposure", 0.0),
            "target_graspability": target_obj.get("graspability", 0.0)
        }
        
        return self._get_obs(), reward, done, False, info
    
    def _execute_grasp(self, target_obj: Dict) -> Tuple[bool, float]:
        """
        执行抓取操作
        返回: (成功标志, 物体位移)
        """
        # 简化的抓取成功判定
        success_prob = target_obj.get("graspability", 0.5)
        
        # 添加随机性
        success = random.random() < success_prob
        
        # 计算位移（简化）
        displacement = random.uniform(0.01, 0.05) if success else 0.0
        
        return success, displacement
    
    def _calculate_reward(self, success: bool, target_obj: Dict, displacement: float) -> float:
        """计算奖励"""
        reward = 0.0
        
        if success:
            # 成功奖励
            reward += self.reward_config["success_reward"]
            
            # 基于暴露度的奖励
            exposure_bonus = target_obj.get("exposure", 0.0) * self.reward_config["exposure_bonus_factor"]
            reward += exposure_bonus
            
        else:
            # 失败惩罚
            reward += self.reward_config["failure_penalty"]
            
            # 基于失败次数的额外惩罚
            obj_id = target_obj["id"]
            failure_count = self.failure_counts.get(obj_id, 0)
            if failure_count > 2:
                reward += self.reward_config["failure_penalty"] * 0.5
        
        # 位移惩罚
        displacement_penalty = displacement * self.reward_config["displacement_penalty_factor"]
        reward += displacement_penalty
        
        # 时间惩罚
        reward += self.reward_config["time_penalty"]
        
        return reward
    
    def _get_obs(self):
        """获取观测"""
        return self._get_obs_extra({})
    
    def evaluate(self):
        """评估环境状态"""
        success = len(self.current_objects) == 0
        fail = self.total_attempts >= 16 and len(self.current_objects) > 0
        
        return {
            "success": torch.tensor([success], device=self.device, dtype=torch.bool),
            "fail": torch.tensor([fail], device=self.device, dtype=torch.bool),
        }


@register_env(
    "ComplexStackingClutter-v1",
    max_episode_steps=16,
)
class ComplexStackingClutterEnvV1(ComplexStackingClutterEnv):
    """复杂堆叠杂乱环境 v1"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("ComplexStackingClutter-v1 环境已初始化") 