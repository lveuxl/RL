import os
import numpy as np
import sapien
import torch
import random
from typing import List, Tuple

import mani_skill.envs.utils.randomization as randomization
from mani_skill import ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# 导入场景配置
from scene_config import SceneConfig


@register_env(
    "PaperScene-v1",
    asset_download_ids=["ycb"],
    max_episode_steps=1,
)
class PaperSceneEnv(BaseEnv):
    """
    **论文展示场景:**
    一个包含12个YCB物体的结构化堆叠场景，用于论文配图展示。
    
    场景包含：
    - 1个目标物体 O_i (底层，大物体如sugar_box)
    - 1个直接风险物体 (直接压在O_i上，部分遮挡)
    - 3-4个间接风险物体 (在直接风险物体上形成子结构)
    - 6-7个中性物体 (散布在场景其他位置)
    
    **物体配置:**
    使用YCB数据集中的多种物体类型，确保视觉丰富性和真实性
    """
    
    # 托盘参数 (与原环境保持一致)
    tray_size = [0.6, 0.6, 0.15]
    tray_spawn_area = [0.23, 0.23]

    def __init__(self, *args, num_envs=1, scene_config='balanced', camera_style='paper_presentation', **kwargs):
        """
        初始化论文展示场景
        
        Args:
            scene_config: 场景配置名称 ('balanced', 'challenging', 'realistic')
            camera_style: 相机风格 ('paper_presentation', 'detailed_analysis')
        """
        # 在super().__init__之前保存配置，避免初始化过程中丢失
        self.scene_config_name = scene_config
        self.camera_style = camera_style
        
        # 验证配置
        if not SceneConfig.validate_config(scene_config):
            raise ValueError(f"无效的场景配置: {scene_config}")
        
        print(f"📋 使用场景配置: {scene_config}")
        print(f"📸 使用相机风格: {camera_style}")
        SceneConfig.print_config_summary(scene_config)
        
        # 为论文展示环境设置空的robot_uids以满足基类需求
        self.robot_uids = None
        
        super().__init__(*args, num_envs=num_envs, **kwargs)
        
        # 存储创建的物体
        self.all_objects = []
        self.target_object = None
        self.risk_objects = []
        self.neutral_objects = []

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21,
                max_rigid_patch_count=2**19
            )
        )

    @property
    def scene_config(self):
        """动态获取场景配置"""
        return SceneConfig.get_scene_config(self.scene_config_name)
    
    @property 
    def camera_config(self):
        """动态获取相机配置"""
        return SceneConfig.get_camera_config(self.camera_style)
    
    @property
    def physics_config(self):
        """动态获取物理配置"""
        return SceneConfig.PHYSICS_CONFIG

    @property
    def _default_sensor_configs(self):
        """根据配置文件动态生成相机配置"""
        camera_configs = []
        
        for camera_name, config in self.camera_config.items():
            pose = sapien_utils.look_at(
                eye=config['eye'],
                target=config['target']
            )
            
            width, height = config['resolution']
            camera_configs.append(CameraConfig(
                camera_name,
                pose=pose,
                width=width,
                height=height,
                fov=config['fov'],
                near=0.01,
                far=100,
            ))
            
            print(f"📷 配置相机 {camera_name}: {config['description']}")
        
        return camera_configs

    @property
    def _default_human_render_camera_configs(self):
        # 使用主相机作为人类渲染相机
        return self._default_sensor_configs[0]

    def _load_agent(self, options: dict):
        """重写agent加载方法 - 论文展示环境不需要机器人"""
        # 论文展示环境不需要机器人，创建一个空的agent列表
        from mani_skill.agents.multi_agent import MultiAgent
        self.agent = None  # 设置为None，表示没有agent
        print("📝 论文展示环境：跳过机器人加载")

    def _load_scene(self, options: dict):
        """加载场景：桌面、托盘和结构化物体布局"""
        # 确保初始化属性
        self.all_objects = []
        self.target_object = None
        self.risk_objects = []
        self.neutral_objects = []
        
        # 构建桌面场景
        self.scene_builder = TableSceneBuilder(self)
        self.scene_builder.build()
        
        # 创建结构化物体布局（直接放在桌面上，无托盘）
        self._create_structured_scene()



    def _create_structured_scene(self):
        """创建结构化的12物体堆叠场景 - 新架构：底层支撑 -> 中层目标 -> 上层风险"""
        for env_idx in range(self.num_envs):
            print(f"\n=== 创建环境 {env_idx} 的中层目标场景结构 ===")
            
            # 1. 放置底层支撑物体 (长方体，稳定的基础)
            support_objs = self._place_support_objects(env_idx)
            
            # 2. 放置中层目标物体 O_i (在支撑物体上)
            target_obj = self._place_target_object_middle_layer(support_objs, env_idx)
            
            # 3. 放置直接风险物体 (直接压在目标物体上)
            direct_risk_obj = self._place_direct_risk_object(target_obj, env_idx)
            
            # 4. 放置间接风险物体 (在直接风险物体上形成子结构)
            indirect_risk_objs = self._place_indirect_risk_objects(direct_risk_obj, env_idx)
            
            # 5. 放置中性物体 (场景其他位置，不规则形状)
            neutral_objs = self._place_neutral_objects(env_idx)
            
            # 收集所有物体
            env_objects = support_objs + [target_obj, direct_risk_obj] + indirect_risk_objs + neutral_objs
            self.all_objects.extend(env_objects)
            
            print(f"环境 {env_idx} 场景创建完成：")
            print(f"  - 底层支撑物体: {[obj.name for obj in support_objs]}")
            print(f"  - 中层目标物体: {target_obj.name}")
            print(f"  - 直接风险物体: {direct_risk_obj.name}")
            print(f"  - 间接风险物体: {[obj.name for obj in indirect_risk_objs]}")
            print(f"  - 中性物体(不规则): {[obj.name for obj in neutral_objs]}")
        
        # 合并所有物体以便管理
        if self.all_objects:
            self.merged_objects = Actor.merge(self.all_objects, name="all_objects")
            print(f"\n场景总计：{len(self.all_objects)} 个物体")

    def _place_support_objects(self, env_idx: int) -> List[Actor]:
        """放置底层支撑物体（长方体形状，密集基础）"""
        obj_types = self.scene_config['support_objects']
        support_objs = []
        
        # 桌面中心附近的密集支撑物体分布 - 4个物体紧密排列（无托盘）
        support_positions = [
            (-0.06, -0.06, "中心左前"),    # 主要支撑位置
            (0.06, -0.06, "中心右前"),     # 次要支撑位置
            (-0.06, 0.06, "中心左后"),     # 左后支撑
            (0.06, 0.06, "中心右后")       # 右后支撑
        ]
        
        for i, obj_type in enumerate(obj_types):
            # 创建物体
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{obj_type}")
            
            # 确定放置位置 - 密集排列
            if i < len(support_positions):
                base_x, base_y, zone_name = support_positions[i]
                # 添加更小的随机偏移保持密集度
                support_x = base_x + random.uniform(-0.01, 0.01)
                support_y = base_y + random.uniform(-0.01, 0.01)
            else:
                # 额外支撑物体在中心区域密集分布
                support_x = random.uniform(-0.05, 0.05)  # 桌面中心范围
                support_y = random.uniform(-0.05, 0.05)  # 桌面中心范围
                zone_name = "密集支撑位置"
            
            support_z = 0.02 + 0.01  # 桌面高度
            
            # 支撑物体使用稳定姿态（更小的随机旋转）
            yaw_angle = random.uniform(-3, 3) * np.pi / 180  # 只有±3度的小角度旋转
            quat = [np.cos(yaw_angle/2), 0, 0, np.sin(yaw_angle/2)]
            
            pose = sapien.Pose(p=[support_x, support_y, support_z], q=quat)
            
            builder.initial_pose = pose
            builder.set_scene_idxs([env_idx])
            
            obj_name = f"env_{env_idx}_support_{i}_{obj_type}"
            support_obj = builder.build(name=obj_name)
            support_objs.append(support_obj)
            
            print(f"  底层支撑{i+1} {obj_type} 密集放置在{zone_name}: [{support_x:.3f}, {support_y:.3f}, {support_z:.3f}]")
        
        return support_objs

    def _place_target_object_middle_layer(self, support_objs: List[Actor], env_idx: int) -> Actor:
        """放置中层目标物体O_i（在支撑物体上）"""
        obj_type = self.scene_config['target_object']
        
        # 创建物体
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{obj_type}")
        
        # 计算支撑物体的中心位置和最大高度
        support_positions = []
        max_height = 0
        
        for support_obj in support_objs:
            support_pos = support_obj.pose.p[0].cpu().numpy() if support_obj.pose.p.dim() > 1 else support_obj.pose.p.cpu().numpy()
            support_positions.append(support_pos)
            
            # 获取支撑物体的高度
            support_type = support_obj.name.split('_')[-1]  # 从名称中提取物体类型
            if support_type in SceneConfig.YCB_OBJECTS:
                support_height = support_pos[2] + SceneConfig.YCB_OBJECTS[support_type]['size'][2]
                max_height = max(max_height, support_height)
        
        # 计算目标物体的中心位置（支撑物体的几何中心）
        center_pos = np.mean(support_positions, axis=0)
        target_x = center_pos[0] + random.uniform(-0.02, 0.02)  # 小偏移
        target_y = center_pos[1] + random.uniform(-0.02, 0.02)
        
        # 🔧 修复：如果max_height计算异常，使用合理的默认高度
        if max_height <= 0.03:  # 如果计算的最大高度异常小
            # 使用支撑物体的标准高度作为基准
            base_height = 0.03  # 托盘底部高度
            typical_support_height = 0.04  # 典型支撑物体高度
            target_z = base_height + typical_support_height + 0.01
            print(f"  ⚠️ 使用默认高度计算：{target_z:.3f}m (max_height={max_height:.3f}异常)")
        else:
            target_z = max_height + 0.01  # 在最高支撑物体上方1cm
            print(f"  📏 中层目标高度：{target_z:.3f}m (基于支撑物体最大高度{max_height:.3f}m)")
        
        # 目标物体使用稳定的姿态
        pose = sapien.Pose(p=[target_x, target_y, target_z], q=[1, 0, 0, 0])
        
        builder.initial_pose = pose
        builder.set_scene_idxs([env_idx])
        
        obj_name = f"env_{env_idx}_target_{obj_type}"
        target_obj = builder.build(name=obj_name)
        
        print(f"  中层目标物体 {obj_type} 放置在: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]")
        return target_obj

    def _place_target_object(self, env_idx: int) -> Actor:
        """放置目标物体O_i在托盘中心位置"""
        obj_type = self.scene_config['target_object']
        
        # 创建物体
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{obj_type}")
        
        # 托盘中心位置，稍微偏移以避免完全对称
        center_x = -0.2 + random.uniform(-0.03, 0.03)
        center_y = 0.0 + random.uniform(-0.03, 0.03)
        base_z = 0.02 + 0.01  # 托盘底部 + 小偏移
        
        # 目标物体使用稳定的姿态（不旋转）
        pose = sapien.Pose(p=[center_x, center_y, base_z], q=[1, 0, 0, 0])
        
        builder.initial_pose = pose
        builder.set_scene_idxs([env_idx])
        
        obj_name = f"env_{env_idx}_target_{obj_type}"
        target_obj = builder.build(name=obj_name)
        
        print(f"  目标物体 {obj_type} 放置在: [{center_x:.3f}, {center_y:.3f}, {base_z:.3f}]")
        return target_obj

    def _place_direct_risk_object(self, target_obj: Actor, env_idx: int) -> Actor:
        """放置直接风险物体，直接压在目标物体上但不完全遮挡"""
        obj_type = self.scene_config['direct_risk']
        
        # 获取目标物体的位置和尺寸
        target_pos = target_obj.pose.p[0].cpu().numpy() if target_obj.pose.p.dim() > 1 else target_obj.pose.p.cpu().numpy()
        target_size = SceneConfig.YCB_OBJECTS[self.scene_config['target_object']]['size']
        
        # 创建物体
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{obj_type}")
        
        # 计算放置位置：在目标物体上方，稍微偏移以实现部分遮挡
        offset_x = target_size[0] * 0.3  # 30%偏移，确保部分遮挡
        offset_y = target_size[1] * 0.2  # 20%偏移
        
        risk_x = target_pos[0] + offset_x
        risk_y = target_pos[1] + offset_y
        risk_z = target_pos[2] + target_size[2] + 0.005  # 目标物体高度 + 小间隙
        
        # 添加小角度旋转增加真实感
        rotation_angle = random.uniform(-15, 15) * np.pi / 180  # ±15度
        # 使用简单的Z轴旋转四元数
        quat = [np.cos(rotation_angle/2), 0, 0, np.sin(rotation_angle/2)]
        
        pose = sapien.Pose(p=[risk_x, risk_y, risk_z], q=quat)
        
        builder.initial_pose = pose
        builder.set_scene_idxs([env_idx])
        
        obj_name = f"env_{env_idx}_direct_risk_{obj_type}"
        risk_obj = builder.build(name=obj_name)
        
        print(f"  直接风险物体 {obj_type} 放置在: [{risk_x:.3f}, {risk_y:.3f}, {risk_z:.3f}]")
        return risk_obj

    def _place_indirect_risk_objects(self, direct_risk_obj: Actor, env_idx: int) -> List[Actor]:
        """放置间接风险物体，在直接风险物体上形成密集子结构（6个）"""
        obj_types = self.scene_config['indirect_risks']
        indirect_objs = []
        
        # 获取直接风险物体的位置和尺寸
        base_pos = direct_risk_obj.pose.p[0].cpu().numpy() if direct_risk_obj.pose.p.dim() > 1 else direct_risk_obj.pose.p.cpu().numpy()
        base_size = SceneConfig.YCB_OBJECTS[self.scene_config['direct_risk']]['size']
        
        current_height = base_pos[2] + base_size[2]  # 当前堆叠高度
        
        for i, obj_type in enumerate(obj_types):
            # 创建物体
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{obj_type}")
            
            # 计算密集堆叠位置：6个物体紧密排列
            if i == 0:
                # 第一个间接风险物体：直接在直接风险物体上方
                stack_x = base_pos[0] + random.uniform(-0.01, 0.01)
                stack_y = base_pos[1] + random.uniform(-0.01, 0.01)
            elif i < 3:
                # 前3个物体在底层紧密排列
                prev_obj = indirect_objs[0]
                prev_pos = prev_obj.pose.p[0].cpu().numpy() if prev_obj.pose.p.dim() > 1 else prev_obj.pose.p.cpu().numpy()
                
                # 围绕第一个物体紧密排列
                angle = (i-1) * np.pi / 2  # 90度间隔
                radius = 0.025  # 2.5cm半径，更密集
                stack_x = prev_pos[0] + radius * np.cos(angle)
                stack_y = prev_pos[1] + radius * np.sin(angle)
            else:
                # 后3个物体在上层密集堆叠
                base_obj_idx = i - 3
                base_obj = indirect_objs[base_obj_idx]
                base_pos = base_obj.pose.p[0].cpu().numpy() if base_obj.pose.p.dim() > 1 else base_obj.pose.p.cpu().numpy()
                
                stack_x = base_pos[0] + random.uniform(-0.015, 0.015)  # 更小偏移
                stack_y = base_pos[1] + random.uniform(-0.015, 0.015)
                current_height = base_pos[2] + SceneConfig.YCB_OBJECTS[obj_types[base_obj_idx]]['size'][2]
            
            stack_z = current_height + 0.003  # 更小间隙，更密集
            
            # 随机旋转减小角度保持稳定
            yaw_angle = random.uniform(-15, 15) * np.pi / 180  # 减小到±15度
            quat = [np.cos(yaw_angle/2), 0, 0, np.sin(yaw_angle/2)]
            
            pose = sapien.Pose(p=[stack_x, stack_y, stack_z], q=quat)
            
            builder.initial_pose = pose
            builder.set_scene_idxs([env_idx])
            
            obj_name = f"env_{env_idx}_indirect_risk_{i}_{obj_type}"
            indirect_obj = builder.build(name=obj_name)
            indirect_objs.append(indirect_obj)
            
            # 只为底层物体更新高度
            if i < 3:
                obj_size = SceneConfig.YCB_OBJECTS[obj_type]['size']
                current_height += obj_size[2]
            
            print(f"  间接风险物体{i+1} {obj_type} 密集放置在: [{stack_x:.3f}, {stack_y:.3f}, {stack_z:.3f}]")
        
        return indirect_objs

    def _place_neutral_objects(self, env_idx: int) -> List[Actor]:
        """放置中性物体（3个长方体，在托盘边缘密集分布）"""
        obj_types = self.scene_config['neutral_objects']  # 3个长方体物体
        neutral_objs = []
        
        # 定义3个长方体的密集放置区域（桌面边缘，相对紧密）
        # 堆叠结构在桌面中心 (0.0, 0.0) 附近，中性物体在边缘但密集
        placement_zones = [
            (-0.15, -0.10, "左前区域"),      # 第1个长方体
            (0.12, 0.12, "右后区域"),        # 第2个长方体  
            (-0.12, 0.15, "左后区域")        # 第3个长方体
        ]
        
        for i, obj_type in enumerate(obj_types):
            # 创建物体
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{obj_type}")
            
            # 使用预定义区域，确保4个物体分散但相对密集放置
            if i < len(placement_zones):
                zone_x, zone_y, zone_name = placement_zones[i]
                # 添加更小的随机偏移，保持密集度
                place_x = zone_x + random.uniform(-0.02, 0.02)
                place_y = zone_y + random.uniform(-0.02, 0.02)
            else:
                # 额外物体在边缘区域相对密集分布
                place_x = random.uniform(-0.18, 0.18)
                place_y = random.uniform(-0.18, 0.18)
                zone_name = "边缘密集位置"
                
            # 确保不与堆叠结构冲突，但允许更接近
            while self._is_position_near_stack(place_x, place_y):
                # 重新随机选择位置，但范围相对紧密
                place_x = random.uniform(-0.20, 0.20)
                place_y = random.uniform(-0.20, 0.20)
                # 确保远离中心堆叠区域，但不需要太远
                if abs(place_x) < 0.08 and abs(place_y) < 0.08:  # 桌面中心区域
                    continue
                else:
                    break
            
            place_z = 0.02 + 0.01  # 桌面高度
            
            # 长方体物体使用适度的随机姿态
            yaw_angle = random.uniform(-45, 45) * np.pi / 180  # ±45度，适中的随机性
            quat = [np.cos(yaw_angle/2), 0, 0, np.sin(yaw_angle/2)]
            
            pose = sapien.Pose(p=[place_x, place_y, place_z], q=quat)
            
            builder.initial_pose = pose
            builder.set_scene_idxs([env_idx])
            
            obj_name = f"env_{env_idx}_neutral_{i}_{obj_type}"
            neutral_obj = builder.build(name=obj_name)
            neutral_objs.append(neutral_obj)
            
            # 打印时标注这是长方体物体
            shape_type = SceneConfig.YCB_OBJECTS[obj_type]['type']
            print(f"  中性长方体{i+1} {obj_type}({shape_type}) 密集放置在{zone_name}: [{place_x:.3f}, {place_y:.3f}, {place_z:.3f}]")
        
        return neutral_objs

    def _is_position_near_stack(self, x: float, y: float) -> bool:
        """检查位置是否与堆叠结构太近"""
        # 堆叠结构大概在桌面中心 (0.0, 0.0) 附近
        stack_center = np.array([0.0, 0.0])
        position = np.array([x, y])
        distance = np.linalg.norm(position - stack_center)
        
        # 如果距离小于10cm认为太近（桌面空间相对较小）
        return distance < 0.10

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化episode - 主要用于重置物体位置"""
        with torch.device(self.device):
            b = len(env_idx)
            # 论文展示环境不需要agent初始化，跳过scene_builder初始化
            # TableSceneBuilder会尝试访问robot_uids，但我们的环境没有机器人
            print(f"📝 论文展示环境：跳过scene_builder初始化（环境 {env_idx}）")
            
            # 重置物体到初始位置（无托盘版本）
            if hasattr(self, 'merged_objects'):
                if b == self.num_envs:
                    self.merged_objects.pose = self.merged_objects.initial_pose
                else:
                    mask = torch.isin(self.merged_objects._scene_idxs, env_idx)
                    self.merged_objects.pose = self.merged_objects.initial_pose[mask]

    def get_obs(self, info=None):
        """获取观测 - 论文展示环境返回简化观测"""
        # 🔧 简化修复：论文展示环境在初始化时不获取相机图像
        # 只返回基本状态观测，避免相机渲染问题
        obs = {
            "scene_stable": torch.ones(self.num_envs, device=self.device, dtype=torch.bool),
            "scene_ready": torch.ones(self.num_envs, device=self.device, dtype=torch.bool),
        }
        
        print("📝 返回基本观测（跳过相机图像以避免初始化问题）")
        return obs

    def step(self, action):
        """简化的step函数 - 主要用于仿真稳定"""
        # 不执行任何动作，只让物理仿真运行几步以稳定物体
        for _ in range(10):
            self.scene.step()
        
        obs = self.get_obs()
        reward = torch.zeros(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info = {"scene_stable": True}
        
        return obs, reward, terminated, truncated, info

    def get_state_dict(self):
        """重写状态获取方法 - 论文展示环境没有agent"""
        # 返回空状态字典，因为论文展示环境没有agent
        return {}
    
    def set_state_dict(self, state):
        """重写状态设置方法 - 论文展示环境没有agent"""
        # 不执行任何操作，因为论文展示环境没有agent状态需要恢复
        pass

    def evaluate(self):
        """重写评估方法 - 论文展示环境不需要任务评估"""
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "fail": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def get_camera_images(self):
        """专门用于获取相机图像的方法"""
        print("🔄 开始获取相机图像...")
        
        # 稳定场景
        print("稳定化场景中...")
        stabilization_steps = self.physics_config['stabilization_steps']
        for _ in range(stabilization_steps):
            self.scene.step()
        
        # 隐藏所有隐藏对象
        for obj in self._hidden_objects:
            obj.hide_visual()
        
        # 更新场景渲染状态
        self.scene.update_render(update_sensors=True, update_human_render_cameras=True)
        
        camera_images = {}
        
        # 获取所有相机的RGB图像
        for camera_name in ["main_camera", "side_camera", "top_camera"]:
            if camera_name in self._sensors:
                try:
                    camera = self._sensors[camera_name]
                    
                    # 强制更新相机
                    camera.camera.take_picture()
                    
                    # 获取相机观测
                    camera_obs = camera.get_obs(rgb=True, depth=False, segmentation=False)
                    camera_images[f"{camera_name}_rgb"] = camera_obs["rgb"]
                    print(f"✅ 成功获取相机 {camera_name} 图像")
                    
                except Exception as e:
                    print(f"⚠️ 获取相机 {camera_name} 图像失败: {e}")
                    camera_images[f"{camera_name}_rgb"] = None
        
        return camera_images

    def save_scene_images(self, save_dir: str = "./paper_scene_images"):
        """保存场景的多角度图像用于论文展示"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("生成场景图像...")
        
        # 使用专门的方法获取相机图像
        camera_images = self.get_camera_images()
        
        # 保存各个角度的图像
        camera_names = ["main_camera", "side_camera", "top_camera"]
        descriptions = ["主视角_45度俯视", "侧面视角_展示堆叠高度", "顶视角_鸟瞰图"]
        
        saved_count = 0
        for camera_name, desc in zip(camera_names, descriptions):
            if f"{camera_name}_rgb" in camera_images and camera_images[f"{camera_name}_rgb"] is not None:
                rgb_tensor = camera_images[f"{camera_name}_rgb"]
                
                try:
                    # 转换为numpy数组
                    if isinstance(rgb_tensor, torch.Tensor):
                        rgb_array = rgb_tensor[0].cpu().numpy()  # 取第一个环境
                    else:
                        rgb_array = rgb_tensor[0]
                    
                    # 确保数据范围正确
                    if rgb_array.max() <= 1.0:
                        rgb_array = (rgb_array * 255).astype(np.uint8)
                    
                    # 保存图像
                    from PIL import Image
                    image = Image.fromarray(rgb_array)
                    filename = f"{save_dir}/{camera_name}_{desc}.png"
                    image.save(filename)
                    print(f"✅ 已保存: {filename}")
                    saved_count += 1
                except Exception as e:
                    print(f"⚠️ 保存相机 {camera_name} 图像失败: {e}")
            else:
                print(f"⚠️ 相机 {camera_name} 图像不可用，跳过保存")
        
        if saved_count > 0:
            print(f"✅ 成功保存 {saved_count} 张图像到: {save_dir}")
        else:
            print("❌ 没有成功保存任何图像")


def create_demo_scene():
    """创建演示场景并生成图像"""
    import gymnasium as gym
    
    print("=== 论文场景环境演示 ===")
    
    # 创建环境
    env = gym.make("PaperScene-v1", num_envs=1)
    
    print("\n初始化场景...")
    obs, _ = env.reset()
    
    print("\n场景结构:")
    print("📦 12个YCB物体的结构化堆叠")
    print("🎯 目标物体O_i: sugar_box (底层)")  
    print("⚠️  直接风险物体: pudding_box (压在O_i上)")
    print("🔺 间接风险物体: 3个小物体形成子结构")
    print("🌟 中性物体: 6个物体分布在场景其他位置")
    
    print("\n保存场景图像...")
    env.unwrapped.save_scene_images()
    
    env.close()
    print("\n演示完成！")


if __name__ == "__main__":
    create_demo_scene()
