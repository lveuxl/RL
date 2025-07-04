import pybullet as p
import pybullet_data
import random
import numpy as np
import time
import math
import gym
from gym import spaces
import sys
from config import ENV_CONFIG, PHYSICS_CONFIG, ROBOT_CONFIG, TABLE_CONFIG
from utils import random_objects
from simulation import setup_scene, generate_objects, setup_robot, setup_walls
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# 自定义强化学习环境类
class PickAndPlaceEnv(gym.Env):
    def __init__(self, render=False, viewMatrix=None, proj_matrix=None, physics_client=None):
        super(PickAndPlaceEnv, self).__init__()
        
        # 新增渲染模式参数
        self.render_mode = render
        
        # 从kwargs或全局配置获取参数
        self.category_map =  ENV_CONFIG['category_map']
        self.obj_num =  ENV_CONFIG['obj_num']
        self.viewMatrix = viewMatrix
        self.proj_matrix = proj_matrix
        self.external_physics_client = physics_client  # 存储外部传入的physics_client
        
        # 动作空间设计
        # 使用固定的最大物体数量，但通过掩码处理无效动作
        self.max_objects = 16
        self.action_space = spaces.Discrete(self.max_objects)
        
        # 添加物体ID到索引的映射，保持一致性
        self.obj_id_to_index = {}  # 物体ID -> 在观察空间中的固定索引
        self.index_to_obj_id = {}  # 索引 -> 物体ID
        self.available_indices = set(range(self.max_objects))  # 可用的索引集合
        
        # 观察空间
        # 每个物体特征：4个位置坐标 + len(self.category_map)个类别one-hot + 1个暴露度(0-1比例) + 1个失败次数
        # 全局特征：1个步数进度 + 1个剩余数量 + 1个成功率
        self.features_per_obj = 4 + len(self.category_map) + 1 + 1  
        self.global_features = 3  
        
        # 计算观察空间总维度
        obs_dim = self.max_objects * self.features_per_obj + self.global_features
        #print(f"观察空间维度: {obs_dim} (16物体 × {self.features_per_obj}特征/物体 + {self.global_features}全局特征)")
        #print(f"每个物体的特征构成: 4位置特征 + {len(self.category_map)}类别特征 + 1暴露度(0-1比例) + 1失败次数")
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.max_objects * self.features_per_obj + self.global_features,), dtype=np.float32
        )
        
        self.obj_ids = []
        self.initial_positions = {}
        self.robot_id = None
        self.kuka_end_effector_idx = None
        self.step_counter = 0
        self.max_steps = 16  # 最多抓取次数
        self.start_time = None
        self.total_distance = 0
        self.success_count = 0
        self.kuka_cid = None
        self.wall_ids = {}
        self.scene = None
        self.is_connected = False
        
        self.episode_rewards = []  # 存储每个回合的累积奖励
        self.current_episode_reward = 0  # 当前回合的累积奖励
        self.episode_success_count = []  # 每个回合的成功抓取次数
        self.episode_displacement = []  # 每个回合的位移总量
        self.episode_steps = []  # 每个回合的步数
        
        # 暴露度计算相关参数
        self.exposure_directions = [
            (0, 0, 1),    # 仅计算顶部暴露度
        ]
        self.ray_density = 5       # 每个方向的光线密度
        self.ray_distance = 0.5    # 光线起点距离物体中心的距离
        
        # 初始化物体暴露度字典
        self.object_exposures = {}
        
        # 初始化连续失败计数器和相关属性
        self.consecutive_failures = 0
        self.recent_successes = []
        self.recent_success_rate = 0.0
        self.last_pick_distance = 0.0
        
    def generate_ray_origins(self, obj_id):
        """
        为特定物体生成不同方向的光线起点网格
        """
        # 获取物体位置和包围盒
        pos, _ = p.getBasePositionAndOrientation(obj_id)
        aabb_min, aabb_max = p.getAABB(obj_id)
        
        # 计算物体尺寸
        size_x = aabb_max[0] - aabb_min[0]
        size_y = aabb_max[1] - aabb_min[1]
        size_z = aabb_max[2] - aabb_min[2]
        
        # 为每个方向生成光线网格
        rays_by_direction = {}
        
        for dir_idx, direction in enumerate(self.exposure_directions):
            rays = []
            
            # 确定光线平面的两个维度
            if abs(direction[0]) > 0:  # X方向
                dim1, dim2 = 1, 2  # Y和Z轴
                size1, size2 = size_y, size_z
            elif abs(direction[1]) > 0:  # Y方向
                dim1, dim2 = 0, 2  # X和Z轴
                size1, size2 = size_x, size_z
            else:  # Z方向
                dim1, dim2 = 0, 1  # X和Y轴
                size1, size2 = size_x, size_y
            
            # 计算网格大小 (以物体尺寸为基础)
            grid_size1 = max(size1, 0.05) # 至少5厘米
            grid_size2 = max(size2, 0.05)
            
            # 计算网格步长
            step1 = grid_size1 / (self.ray_density - 1) if self.ray_density > 1 else grid_size1
            step2 = grid_size2 / (self.ray_density - 1) if self.ray_density > 1 else grid_size2
            
            # 计算开始位置 (网格中心位于物体中心)
            start1 = -grid_size1 / 2
            start2 = -grid_size2 / 2
            
            # 生成网格
            for i in range(self.ray_density):
                for j in range(self.ray_density):
                    # 计算当前网格点的偏移量
                    offset1 = start1 + i * step1
                    offset2 = start2 + j * step2
                    
                    # 创建光线起点坐标
                    ray_origin = [0, 0, 0]
                    ray_origin[dim1] = offset1
                    ray_origin[dim2] = offset2
                    
                    # 计算光线方向向量的位置分量
                    ray_origin_dir = direction[0] * self.ray_distance
                    ray_origin[0] = pos[0] + ray_origin[0] + direction[0] * self.ray_distance
                    ray_origin[1] = pos[1] + ray_origin[1] + direction[1] * self.ray_distance
                    ray_origin[2] = pos[2] + ray_origin[2] + direction[2] * self.ray_distance
                    
                    # 计算光线终点 (物体中心)
                    ray_end = pos
                    
                    rays.append((ray_origin, ray_end))
            
            rays_by_direction[dir_idx] = rays
        
        return rays_by_direction

    def _calculate_object_exposure(self, obj_id):
        """
        计算物体的顶部暴露度 - 定义为顶部暴露面积占顶部总面积的比例(0-1)
        """
        # 生成顶部方向的光线
        rays_by_direction = self.generate_ray_origins(obj_id)
        
        # 计算顶部方向的暴露度 (方向索引0对应顶部)
        if 0 in rays_by_direction:
            rays = rays_by_direction[0]
            
            # 统计击中物体的光线数量
            hit_count = 0
            total_rays = len(rays)
            
            for ray_origin, ray_end in rays:
                results = p.rayTest(ray_origin, ray_end)
                
                # 检查光线是否首先击中目标物体
                if results[0][0] == obj_id:
                    hit_count += 1
            
            # 计算顶部方向的暴露比例 (0-1之间)
            exposure_ratio = hit_count / total_rays if total_rays > 0 else 0
        else:
            exposure_ratio = 0
        
        return exposure_ratio
    
    def _calculate_all_objects_exposure(self):
        """计算所有物体的暴露度"""
        exposure_dict = {}
        
        for obj in self.obj_ids:
            obj_id = obj['id']
            exposure = self._calculate_object_exposure(obj_id)
            exposure_dict[obj_id] = exposure
            
            # 更新物体的暴露度属性
            obj['exposure'] = exposure
            
            # 附加每个方向的暴露度，用于高级决策
            # obj['exposure_by_direction'] = exposure_by_direction
        
        return exposure_dict

    def _connect_physics_engine(self):
        """连接到物理引擎，根据渲染模式选择GUI或DIRECT"""
        # 如果已经提供了physics_client，就直接使用它
        if self.external_physics_client is not None:
            self.physics_client = self.external_physics_client
            self.is_connected = True
            return
        
        # 检查是否已经连接
        if self.is_connected:
            # 尝试检查连接是否有效
            try:
                p.getConnectionInfo(self.physics_client)
                return  # 连接有效，直接返回
            except:
                # 连接无效，继续创建新连接
                pass
        
        # 检查环境变量是否允许GUI连接
        import os
        allow_gui = os.environ.get('PYBULLET_ALLOW_GUI', '1') == '1'
        
        # 根据render参数和环境变量创建新的连接
        try:
            if self.render_mode and allow_gui:
                # 只有在明确允许GUI且render_mode为True时才创建GUI连接
                try:
                    self.physics_client = p.connect(p.GUI)
                    print(f"成功创建GUI连接，连接ID: {self.physics_client}")
                    p.resetDebugVisualizerCamera(cameraDistance=0.58, cameraYaw=-90, cameraPitch=-89, 
                                                cameraTargetPosition=[1.0, -0.2, 1.3])
                    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 禁用 GUI 渲染
                except Exception as gui_error:
                    print(f"创建GUI连接失败: {gui_error}，回退到DIRECT模式")
                    self.physics_client = p.connect(p.DIRECT)
            else:
                # 使用DIRECT模式
                self.physics_client = p.connect(p.DIRECT)
                if not allow_gui:
                    print(f"环境变量禁止GUI连接，使用DIRECT模式，连接ID: {self.physics_client}")
                else:
                    print(f"创建DIRECT连接，连接ID: {self.physics_client}")
            
            # 设置搜索路径，确保能找到模型文件
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.is_connected = True
        except Exception as e:
            print(f"连接物理引擎失败: {e}")
            # 如果连接失败，尝试使用DIRECT模式
            self.physics_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.is_connected = True
        
    def seed(self, seed=None):
        """设置随机种子"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed]
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            self.seed(seed)
        
        # 确保已连接到物理引擎
        self._connect_physics_engine()
        
        # 重置环境
        p.resetSimulation()
        p.setGravity(*PHYSICS_CONFIG["gravity"])
        
        # 加载场景
        self.scene = setup_scene()
        # 初始化机械臂
        self.robot_id = self.scene["robot_id"]
        self.num_joints = self.scene["num_joints"]
        self.kuka_end_effector_idx = self.scene["kuka_end_effector_idx"]
        self.kuka_cid = self.scene["kuka_cid"]
        
        # 获取墙壁ID
        self.wall_ids = self.scene["wall_ids"]
        wall_id1 = self.wall_ids["wall_id1"]
        wall_id2 = self.wall_ids["wall_id2"]
        wall_id3 = self.wall_ids["wall_id3"]
        wall_id4 = self.wall_ids["wall_id4"]
        wall_id5 = self.wall_ids["wall_id5"]
        wall_id6 = self.wall_ids["wall_id6"]
        wall_id7 = self.wall_ids["wall_id7"]

        # 生成物体
        self.obj_ids = generate_objects(self.obj_num)
        self.obj_number = len(self.obj_ids)

        # walls_to_remove = [wall_id1, wall_id3, wall_id4, wall_id5, wall_id6, wall_id7]
        # for wall_id in walls_to_remove:
        p.removeBody(wall_id1)
        p.removeBody(wall_id2)
        p.removeBody(wall_id3)
        p.removeBody(wall_id5)
        p.removeBody(wall_id7)

        # 拍照如果需要
        if self.viewMatrix is not None and self.proj_matrix is not None:
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                1408, 1024, self.viewMatrix, projectionMatrix=self.proj_matrix, 
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
        
        # 记录初始位置
        self.initial_positions = {}
        for obj in self.obj_ids:
            self.initial_positions[obj['id']] = p.getBasePositionAndOrientation(obj['id'])[0]
        
        self.step_counter = 0
        self.start_time = time.time()
        self.total_distance = 0
        self.success_count = 0
        
        self.current_episode_reward = 0  # 重置当前回合奖励
        
        # 重置索引映射
        self.obj_id_to_index = {}
        self.index_to_obj_id = {}
        self.available_indices = set(range(self.max_objects))
        
        # 为每个物体分配固定的索引
        for i, obj in enumerate(self.obj_ids[:self.max_objects]):
            obj_id = obj['id']
            self.obj_id_to_index[obj_id] = i
            self.index_to_obj_id[i] = obj_id
            self.available_indices.discard(i)
        
        # 重置物体暴露度信息
        self.object_exposures = {}
        
        # 重置连续失败计数器和相关属性
        self.consecutive_failures = 0
        self.recent_successes = []
        self.recent_success_rate = 0.0
        self.last_pick_distance = 0.0
        
        # 计算初始暴露度
        try:
            self._calculate_all_objects_exposure()
        except Exception as e:
            print(f"计算初始暴露度时出错: {e}")
            self.object_exposures = {}
        
        # 返回观察和info字典，兼容新版Gym API
        return self._get_observation(), {}

    
    def _get_observation(self):
        """
        观察获取方法，使用固定索引映射
        """
        # 首先清理无效物体
        self._cleanup_invalid_objects()
        
        # 使用0初始化整个观察空间
        obs = np.zeros(self.max_objects * self.features_per_obj + self.global_features, dtype=np.float32)
        
        # 只在有物体时计算暴露度
        if len(self.obj_ids) > 0:
            exposure_dict = self._calculate_all_objects_exposure()
        else:
            exposure_dict = {}
        
        scene_center = (0.5, -0.2) 
        
        # 使用固定索引填充物体特征
        for obj in self.obj_ids:
            obj_id = obj['id']
            
            # 获取该物体的固定索引
            if obj_id not in self.obj_id_to_index:
                continue  # 跳过没有分配索引的物体
            
            try:
                pos, _ = p.getBasePositionAndOrientation(obj_id)
            except:
                print(f"警告：物体 {obj_id} 在获取观察时不存在，跳过")
                continue
            
            fixed_index = self.obj_id_to_index[obj_id]
            
            # 基础索引
            base_idx = fixed_index * self.features_per_obj
            
            # 1. 位置信息 (4个值) - 使用相对于场景中心的位置
            relative_position_features = self._get_balanced_position_features(pos, scene_center)
            obs[base_idx:base_idx+4] = relative_position_features
            
            # 2. 类别信息（one-hot编码）
            category_start_idx = base_idx + 4
            category_end_idx = category_start_idx + len(self.category_map)
            obs[category_start_idx:category_end_idx] = 0  # 先清零
            
            category_idx = self.category_map[obj['category']] - 1
            obs[category_start_idx + category_idx] = 1
            
            # 3. 暴露度信息 (1个值)
            exposure_idx = base_idx + 4 + len(self.category_map)
            if obj_id in exposure_dict:
                obs[exposure_idx] = exposure_dict[obj_id]
            else:
                obs[exposure_idx] = 0.0
            
            # 4. 抓取失败次数 (1个值) 
            fail_count_idx = exposure_idx + 1
            fail_count = obj.get('suck_times', 0)
            # 使用更平滑的归一化：tanh函数
            normalized_fail_count = np.tanh(fail_count / 3.0)  # 将失败次数映射到[0,1)
            obs[fail_count_idx] = normalized_fail_count
        
        # 全局信息部分
        global_start_idx = self.max_objects * self.features_per_obj

        # 1. 当前步数进度 (归一化到[0,1])
        step_progress = min(self.step_counter / self.max_steps, 1.0)
        obs[global_start_idx] = np.sqrt(step_progress)

        # 2. 剩余物体数量 (归一化到[0,1])
        obs[global_start_idx + 1] = len(self.obj_ids) / self.max_objects

        # 3. 当前成功率 (归一化到[0,1])
        current_success_rate = self.success_count / max(self.step_counter, 1)
        obs[global_start_idx + 2] = min(current_success_rate, 1.0)
        
        # 确保观察值在合理范围内
        obs = np.clip(obs, -5.0, 5.0)
        
        return obs
    
    def _cleanup_invalid_objects(self):
        """清理无效物体，确保状态一致性"""
        valid_objects = []
        cleaned_objects = []
        
        for obj in self.obj_ids:
            try:
                # 验证物体是否还存在于物理世界
                p.getBasePositionAndOrientation(obj['id'])
                valid_objects.append(obj)
            except:
                # 物体已被移除，清理映射
                obj_id = obj['id']
                cleaned_objects.append(obj_id)
                if obj_id in self.obj_id_to_index:
                    removed_index = self.obj_id_to_index[obj_id]
                    del self.obj_id_to_index[obj_id]
                    if removed_index in self.index_to_obj_id:
                        del self.index_to_obj_id[removed_index]
                    self.available_indices.add(removed_index)
        
        if cleaned_objects:
            print(f"自动清理无效物体: {cleaned_objects}")
        
        self.obj_ids = valid_objects

    def _get_balanced_position_features(self, obj_pos, scene_center):
        robot_base = (0.5, -0.2)
        dx = obj_pos[0] - robot_base[0]
        dy = obj_pos[1] - robot_base[1]
        
        return [
            obj_pos[0] - scene_center[0],    # 相对场景中心X
            obj_pos[1] - scene_center[1],    # 相对场景中心Y
            obj_pos[2] - 0.6,                # 相对桌面高度
            np.sqrt(dx**2 + dy**2)           # 到机械臂距离
        ]
    
    def _pick_object(self, obj):
        # 执行抓取操作
        obj_id = obj['id']
        obj_pos = p.getBasePositionAndOrientation(obj_id)[0]
        success = False
        dist_obj = 0
        fail = 0

        # 在抓取开始时记录物体位置和暴露度
        obj_exposure = obj.get('exposure', 0.01)
        print(f"尝试抓取物体 {obj_id-6}，暴露度比例: {obj_exposure:.2f}")
        
        if obj['suck_times'] >= 5:
            print(f"物体{obj_id-6}吸取次数超过5次,放弃吸取")
            return success, dist_obj
        else:
            # 抓取过程的状态机
            for state in range(8):
                if state == 0:
                    # 机械臂上升，防止碰到物体
                    roblist = list(obj_pos)
                    roblist[2] += 0.3
                    target_pos, gripper_val = roblist, 0
                   #print(f"状态{state}: 机械臂上升")
                elif state == 1:
                    # 机械臂下降到物体上方，准备吸取
                    roblist[2] = obj_pos[2] + 0.05
                    target_pos, gripper_val = roblist, 0
                    #print(f"状态{state}: 机械臂下降到物体上方")
                elif state == 2:
                    # 吸取物体
                    target_pos, gripper_val = roblist, 1
                    #print(f"状态{state}: 吸取物体")
                elif state == 3:
                    # 物体上升
                    roblist[2] += 0.5
                    target_pos, gripper_val = roblist, 1
                    #print(f"状态{state}: 物体上升") 
                    
                elif state == 4:
                    # 移动到放置位置
                    roblist[0] = 0.7
                    roblist[1] = -0.6
                    target_pos, gripper_val = roblist, 1
                    #print(f"状态{state}: 移动到放置位置")
                elif state == 5:
                    # 下降
                    roblist[2] -= 0.3
                    target_pos, gripper_val = roblist, 1
                    #print(f"状态{state}: 下降")

                elif state == 6:
                    # 放下物体
                    target_pos, gripper_val = roblist, 0
                    #print(f"状态{state}: 放下物体")
                elif state == 7:
                    # 回到初始位置
                    roblist[0] = 0.5
                    roblist[1] = -0.6
                    roblist[2] += 0.4
                    target_pos, gripper_val = roblist, 0
                    #print(f"状态{state}: 回到初始位置")
                # 设置机械臂姿态
                target_orn = p.getQuaternionFromEuler([0, 1.01*math.pi, 0])
                joint_poses = p.calculateInverseKinematics(self.robot_id, self.kuka_end_effector_idx, target_pos, target_orn)
                
                # 控制机械臂关节
                for j in range(p.getNumJoints(self.robot_id)):
                    if j <= 6:
                        p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=j, 
                                        controlMode=p.POSITION_CONTROL, 
                                        targetPosition=joint_poses[j])
                
                # 模拟步骤
                for _ in range(30):
                    p.stepSimulation()
                    #time.sleep(1 / 240)
                
                # 获取末端执行器位置
                link_state = p.getLinkState(self.robot_id, self.kuka_end_effector_idx)
                end_effector_pos = link_state[0]
                
                # 检查是否可以抓取
                if 0 < state < 7 and self.kuka_cid == None and (end_effector_pos[0] > target_pos[0] + 0.1 or end_effector_pos[0] < target_pos[0] - 0.1 or end_effector_pos[1] < target_pos[1] - 0.1 or end_effector_pos[1] > target_pos[1] + 0.1):
                    if state == 2 :
                        # 检查是否在机械臂范围内
                        x, y, _ = obj_pos
                        dx = x - 0.5  # 圆心x
                        dy = y - (-0.2)  # 圆心y
                        distance = math.sqrt(dx**2 + dy**2)
                        dist = math.sqrt((end_effector_pos[0] - x)**2 + (end_effector_pos[1] - y)**2)
                        print(f"物体{obj_id-6}已经超过机械臂最大吸取范围，距离为{distance}吸取失败,距离机械臂为{dist}")
                        obj["suck_times"] += 1
                        obj["skip"] = True
                        fail += 1
                
                elif 0 < state < 7 and self.kuka_cid == None:
                        # 检查是否被遮挡
                        ray_start_1 = [obj_pos[0]-0.03, obj_pos[1], obj_pos[2] + 0.5]
                        ray_end_1 = [obj_pos[0]-0.03, obj_pos[1], obj_pos[2]]
                        ray_start_2 = [obj_pos[0], obj_pos[1]-0.03, obj_pos[2] + 0.5]
                        ray_end_2 = [obj_pos[0], obj_pos[1]-0.03, obj_pos[2]]
                        ray_start_3 = [obj_pos[0]+0.03, obj_pos[1], obj_pos[2] + 0.5]
                        ray_end_3 = [obj_pos[0]+0.03, obj_pos[1], obj_pos[2]]
                        ray_start_4 = [obj_pos[0], obj_pos[1]+0.03, obj_pos[2] + 0.5]
                        ray_end_4 = [obj_pos[0], obj_pos[1]+0.03, obj_pos[2]]
                        
                        ray_results = [p.rayTest(ray_start, ray_end) for ray_start, ray_end in 
                                    [(ray_start_1, ray_end_1), (ray_start_2, ray_end_2), 
                                    (ray_start_3, ray_end_3), (ray_start_4, ray_end_4)]]
                        
                        is_blocked = False
                        if (ray_results[0][0][0] != obj_id and ray_results[0][0][0] > 13) or (ray_results[1][0][0] != obj_id and ray_results[1][0][0] > 13) or (ray_results[2][0][0] != obj_id and ray_results[2][0][0] > 13) or (ray_results[3][0][0] != obj_id and ray_results[3][0][0] > 13):
                            if state == 2:
                                print(f"物体{obj_id-6}被遮挡，吸取失败，等待后续抓取")
                                obj["skip"] = True
                                is_blocked = True
                                obj["suck_times"] += 1
                                fail += 1
                                #将物体往后放，重新等待抓取
                                # self.obj_ids.append(obj)

                    
                        if is_blocked:
                            break
                
                 # 判断是否成功抓起物体
                elif state == 6 and self.kuka_cid is not None:
                    success = True
                    obj['suck_state'] = True
                    print(f"抓取{obj_id}物体成功")
                # 处理吸盘约束
                if gripper_val == 1 and self.kuka_cid is None and obj['skip'] == False:
                    # 创建吸盘约束
                    cube_orn = p.getQuaternionFromEuler([0, math.pi, 0])
                    self.kuka_cid = p.createConstraint(self.robot_id, 11, obj_id, -1, 
                                            p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.01], [0, 0, 0], 
                                            childFrameOrientation=cube_orn)
                elif gripper_val == 0 and self.kuka_cid is not None:
                    # 移除约束
                    p.removeConstraint(self.kuka_cid)
                    self.kuka_cid = None
                    for _ in range(60):
                        p.stepSimulation()
                        #time.sleep(1 / 240)
                
                # 在状态6时移除物体（如果已经成功抓取）
                if state == 6 and success:
                    p.removeBody(obj_id)   
                    # 在物体抓起后计算其他物体的位移
                    for other_obj in self.obj_ids:
                        if not other_obj.get('suck_state', False):
                            current_pos = p.getBasePositionAndOrientation(other_obj['id'])[0]
                            initial_pos = self.initial_positions[other_obj['id']]
                            dist = math.sqrt(sum((current_pos[i] - initial_pos[i])**2 for i in range(3)))
                            dist_obj += dist
                            # 更新初始位置
                            self.initial_positions[other_obj['id']] = current_pos
            return success, dist_obj
    
    def _get_valid_actions(self):
        """
        获取当前有效的动作列表
        """
        valid_actions = []
        for obj in self.obj_ids:
            obj_id = obj['id']
            if obj_id in self.obj_id_to_index:
                valid_actions.append(self.obj_id_to_index[obj_id])
        return valid_actions

    def _get_action_mask(self):
        """
        获取动作掩码，用于屏蔽无效动作
        """
        mask = np.zeros(self.max_objects, dtype=bool)
        valid_actions = self._get_valid_actions()
        if valid_actions:
            mask[valid_actions] = True
        return mask

    def step(self, action):
        # 处理动作类型转换
        if isinstance(action, np.ndarray):
            if action.ndim == 0:  # 标量数组
                action = int(action.item())
            else:  # 多维数组，取第一个元素
                action = int(action.flatten()[0])
        elif not isinstance(action, (int, np.integer)):
            action = int(action)
        
        # 检查是否还有物体可抓取
        if len(self.obj_ids) == 0:
            print("警告：没有可抓取的物体了")
            return self._get_observation(), 0, True, False, {
                "success": False, 
                "dist_moved": 0,
                "action_mask": self._get_action_mask()
            }
        
        # 获取有效动作和动作掩码
        valid_actions = self._get_valid_actions()
        action_mask = self._get_action_mask()
        
        # 动作验证和选择逻辑
        action_penalty = 0.0
        obj_to_pick = None
        forced_selection = False

        
        # 验证动作是否有效
        if action < self.max_objects and action_mask[action]:
            # 动作有效，获取目标物体
            target_obj_id = self.index_to_obj_id[action]
            obj_to_pick = next((obj for obj in self.obj_ids if obj['id'] == target_obj_id), None)
            
            if obj_to_pick is None:
                print(f"警告：动作{action}对应的物体{target_obj_id}不存在，重新选择")
                forced_selection = True
            else:
                # 检查物体是否已经失败太多次
                fail_count = obj_to_pick.get('suck_times', 0)
                if fail_count >= 5:  
                    print(f"物体{target_obj_id-6}已失败{fail_count}次，强制选择其他物体")
                    forced_selection = True
                    action_penalty = -1.0  # 重度惩罚选择高失败率物体
                else:
                    print(f"有效动作: {action}, 目标物体ID: {target_obj_id}, 失败次数: {fail_count}")
        else:
            print(f"无效动作: {action}, 动作掩码: {action_mask}")
            forced_selection = True
            action_penalty = -0.3  
        
        # 如果需要强制选择，使用改进的选择策略
        if forced_selection:
            obj_to_pick = self._select_best_alternative_object()
            if obj_to_pick:
                action = self.obj_id_to_index[obj_to_pick['id']]
                print(f"强制选择最佳替代物体: 动作={action}, 物体ID={obj_to_pick['id']}")
            else:
                print("错误：无法找到任何可选择的物体")
                return self._get_observation(), -2.0, True, False, {
                    "success": False,
                    "dist_moved": 0,
                    "action_mask": action_mask,
                    "error": "no_selectable_objects"
                }
        
        # 计算当前所有物体的暴露度
        exposure_dict = self._calculate_all_objects_exposure()
        chosen_exposure = exposure_dict.get(obj_to_pick['id'], 0.01)
        
        # 执行抓取前记录失败次数
        pre_fail_count = obj_to_pick.get('suck_times', 0)
        
        print(f"选择物体ID: {obj_to_pick['id']}, 固定索引: {action}, 暴露度比例: {chosen_exposure:.2f}, 历史失败: {pre_fail_count}次")
        
        # 执行抓取
        success, dist_moved = self._pick_object(obj_to_pick)
        
        # 更新失败计数和连续失败追踪
        if not success:
            obj_to_pick['suck_times'] = obj_to_pick.get('suck_times', 0) + 1
            self.consecutive_failures = getattr(self, 'consecutive_failures', 0) + 1
            
            # 如果物体失败次数过多，标记为跳过
            if obj_to_pick['suck_times'] >= 4:
                obj_to_pick['skip'] = True
                print(f"⚠️ 物体{obj_to_pick['id']-6}失败次数达到{obj_to_pick['suck_times']}次，标记为跳过")
        else:
            self.consecutive_failures = 0
        
        # 更新成功率追踪
        self._update_success_rate_tracking(success)
        
        # 更新计数器
        self.step_counter += 1
        self.total_distance += dist_moved
        
        if success:
            self.success_count += 1
            # 从列表和映射中移除已抓取的物体
            try:
                obj_id = obj_to_pick['id']
                self.obj_ids.remove(obj_to_pick)
                
                # 清理索引映射
                if obj_id in self.obj_id_to_index:
                    removed_index = self.obj_id_to_index[obj_id]
                    del self.obj_id_to_index[obj_id]
                    del self.index_to_obj_id[removed_index]
                    self.available_indices.add(removed_index)
                
                print(f"✅ 成功移除物体: {obj_id}, 释放索引: {removed_index}, 剩余物体: {len(self.obj_ids)}")
            except ValueError:
                print(f"无法从列表中移除物体: {obj_to_pick['id']}")
        
        # 使用改进的奖励函数
        reward = self._calculate_reward(success, dist_moved, chosen_exposure, obj_to_pick, forced_selection) + action_penalty
        
        # 更新当前回合累积奖励
        self.current_episode_reward += reward
        
        # 检查是否结束
        done = (self.step_counter >= self.max_steps) or (len(self.obj_ids) == 0)
        
        # 获取新的观察和动作掩码
        obs = self._get_observation()
        new_action_mask = self._get_action_mask()
        
        # 构建info字典
        info = {
            'success': success,
            'dist_moved': dist_moved,
            'remaining_objects': len(self.obj_ids),
            'success_count': self.success_count,
            'object_exposure': exposure_dict,
            'action_mask': new_action_mask,
            'valid_actions': self._get_valid_actions(),
            'action_penalty': action_penalty,
            'forced_selection': forced_selection,
            'object_fail_count': obj_to_pick.get('suck_times', 0),
            'consecutive_failures': getattr(self, 'consecutive_failures', 0),
            'recent_success_rate': getattr(self, 'recent_success_rate', 0.0),
            'reward_breakdown': {
                'base_reward': reward - action_penalty,
                'action_penalty': action_penalty,
                'total_reward': reward
            }
        }
        
        print(f"步骤 {self.step_counter}: 动作={action}, 成功={success}, 奖励={reward:.2f}, 剩余={len(self.obj_ids)}, 连续失败={getattr(self, 'consecutive_failures', 0)}")
        
        return obs, reward, done, False, info

    def _select_best_alternative_object(self):
        """
        选择最佳替代物体的改进策略
        """
        if not self.obj_ids:
            return None
        
        # 过滤掉失败次数过多的物体
        available_objects = [obj for obj in self.obj_ids if obj.get('suck_times', 0) < 4 and not obj.get('skip', False)]
        
        if not available_objects:
            # 如果所有物体都失败过多，选择失败次数最少的
            available_objects = sorted(self.obj_ids, key=lambda x: x.get('suck_times', 0))[:3]
            print("所有物体都有较高失败率，选择失败次数最少的物体")
        
        if not available_objects:
            return None
        
        # 计算可用物体的暴露度
        exposure_dict = {}
        for obj in available_objects:
            try:
                exposure_dict[obj['id']] = self._calculate_object_exposure(obj['id'])
            except:
                exposure_dict[obj['id']] = 0.01
        
        # 综合评分：暴露度权重70%，失败率权重30%
        def calculate_score(obj):
            exposure = exposure_dict.get(obj['id'], 0.01)
            fail_count = obj.get('suck_times', 0)
            
            # 暴露度分数 (0-1)
            exposure_score = exposure
            
            # 失败率分数 (失败次数越少分数越高)
            failure_score = max(0, 1.0 - fail_count / 5.0)
            
            # 综合分数
            total_score = 0.7 * exposure_score + 0.3 * failure_score
            
            return total_score
        
        # 选择得分最高的物体
        best_obj = max(available_objects, key=calculate_score)
        
        print(f"最佳替代物体: ID={best_obj['id']}, 暴露度={exposure_dict.get(best_obj['id'], 0)*10000:.1f}平方厘米, 失败次数={best_obj.get('suck_times', 0)}")
        
        return best_obj

    def _calculate_reward(self, success, dist_moved, exposure, obj_to_pick, forced_selection):
        """
        优化的奖励函数 - 更稳定的奖励结构，适合快速收敛
        """
        fail_count = obj_to_pick.get('suck_times', 0) if obj_to_pick else 0
        
        if success:
            # 成功奖励：固定基础奖励 + 小幅暴露度奖励
            base_success_reward = 10.0  # 降低基础奖励，提高稳定性
            
            # 暴露度奖励：0-2范围，鼓励选择暴露度高的物体
            exposure_bonus = min(exposure * 2.0, 2.0)
            
            # 效率奖励：鼓励快速完成
            efficiency_bonus = max(0, 2.0 - fail_count * 0.3)
            
            total_reward = base_success_reward + exposure_bonus + efficiency_bonus
            
            print(f"✅ 成功奖励: 基础={base_success_reward}, 暴露度={exposure_bonus:.2f}, 效率={efficiency_bonus:.2f}")
            
        else:
            # 失败惩罚：适度惩罚，避免过度负奖励
            base_penalty = -2.0  # 减少惩罚强度
            
            # 重复失败惩罚：线性增长但有上限
            repeat_failure_penalty = -min(fail_count * 0.5, 2.0)
            
            # 暴露度补偿：失败时给予小额补偿
            exposure_compensation = min(exposure * 1.0, 1.0)
            
            # 强制选择惩罚
            forced_penalty = -0.3 if forced_selection else 0
            
            total_reward = base_penalty + repeat_failure_penalty + exposure_compensation + forced_penalty
            
            print(f"❌ 失败惩罚: 基础={base_penalty}, 重复失败={repeat_failure_penalty:.2f}, 暴露度补偿={exposure_compensation:.2f}")
        
        # 位移惩罚：减少对环境扰动的惩罚
        displacement_penalty = -0.1 * dist_moved
        
        # 时间惩罚：轻微的时间压力
        time_penalty = -0.05
        
        # 连续失败惩罚：适度惩罚连续失败
        consecutive_penalty = 0
        if hasattr(self, 'consecutive_failures') and self.consecutive_failures > 2:
            consecutive_penalty = -min((self.consecutive_failures - 2) * 0.3, 1.0)
        
        final_reward = total_reward + displacement_penalty + time_penalty + consecutive_penalty
        
        # 限制奖励范围到更合理的区间
        final_reward = np.clip(final_reward, -5.0, 15.0)
        
        print(f"最终奖励: {final_reward:.2f} (范围: [-5, 15], 物体失败次数: {fail_count})")
        
        return final_reward

    def _update_success_rate_tracking(self, success):
        """
        更新成功率追踪，用于自适应奖励调整
        """
        if not hasattr(self, 'recent_successes'):
            self.recent_successes = []
        
        self.recent_successes.append(success)
        
        # 只保留最近20次的记录
        if len(self.recent_successes) > 20:
            self.recent_successes.pop(0)
        
        # 计算最近的成功率
        self.recent_success_rate = sum(self.recent_successes) / len(self.recent_successes)

    def close(self):
        """安全关闭环境和物理引擎连接"""
        try:
            if self.is_connected and hasattr(self, 'physics_client'):
                # 如果使用的是外部提供的physics_client，不要断开它
                if self.external_physics_client is None:
                    print(f"断开物理引擎连接: {self.physics_client}")
                    p.disconnect(self.physics_client)
                else:
                    print("使用外部physics_client，跳过断开连接")
                self.is_connected = False
        except Exception as e:
            print(f"关闭环境时出错: {e}")
            # 即使出错也要重置连接状态
            self.is_connected = False
    
    def _save_episode_stats(self):
        """保存回合统计数据到CSV文件"""
        import pandas as pd
        import os
        
        # 创建数据字典
        data = {
            'episode': list(range(1, len(self.episode_rewards) + 1)),
            'total_reward': self.episode_rewards,
            'success_count': self.episode_success_count,
            'total_displacement': self.episode_displacement,
            'steps': self.episode_steps
        }
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        os.makedirs('logs', exist_ok=True)
        df.to_csv(f'logs/episode_stats.csv', index=False)

    def visualize_exposure_rays(self, obj_id, show_all_directions=False):
        """
        可视化用于顶部暴露度计算的光线，并显示顶部暴露面积
        """
        # 获取物体包围盒
        aabb_min, aabb_max = p.getAABB(obj_id)
        
        # 计算物体尺寸
        size_x = aabb_max[0] - aabb_min[0]
        size_y = aabb_max[1] - aabb_min[1]
        size_z = aabb_max[2] - aabb_min[2]
        
        # 只计算顶部面的面积
        top_face_area = size_x * size_y  # Z+ 方向 (顶部)
        
        rays_by_direction = self.generate_ray_origins(obj_id)
        
        # 计算顶部方向的暴露度 (方向索引0对应顶部)
        if 0 in rays_by_direction:
            rays = rays_by_direction[0]
            hit_count = 0
            total_rays = len(rays)
            
            for ray_origin, ray_end in rays:
                results = p.rayTest(ray_origin, ray_end)
                if results[0][0] == obj_id:
                    hit_count += 1
            
            # 计算顶部暴露面积
            exposure_ratio = hit_count / total_rays if total_rays > 0 else 0
        else:
            hit_count = 0
            total_rays = 0
            exposure_ratio = 0
        
        # 清除之前的调试线
        p.removeAllUserDebugItems()
        
        # 可视化顶部方向的光线
        if 0 in rays_by_direction:
            for ray_origin, ray_end in rays_by_direction[0]:
                results = p.rayTest(ray_origin, ray_end)
                
                if results[0][0] == obj_id:
                    # 绿色表示光线击中目标物体
                    p.addUserDebugLine(ray_origin, ray_end, [0, 1, 0], 2, 0.05)
                else:
                    # 红色表示光线被其他物体遮挡
                    p.addUserDebugLine(ray_origin, ray_end, [1, 0, 0], 2, 0.05)
        
        # 显示暴露度信息
        text_pos, _ = p.getBasePositionAndOrientation(obj_id)
        text_pos = [text_pos[0], text_pos[1], text_pos[2] + 0.1]  # 在物体上方显示
        
        # 显示顶部暴露面积（平方厘米）
        top_exposure_cm2 = top_face_area * exposure_ratio * 10000  # 转换为平方厘米
        top_face_area_cm2 = top_face_area * 10000  # 转换为平方厘米
        
        # 添加详细调试文本
        debug_text = f"顶部暴露度: {exposure_ratio*100:.1f}% (暴露面积: {top_exposure_cm2:.1f}平方厘米)"
        p.addUserDebugText(debug_text, text_pos, [0, 0, 0], 1.5)
        
        # 添加面积详情
        text_pos2 = [text_pos[0], text_pos[1], text_pos[2] + 0.05]
        detail_text = f"光线命中: {hit_count}/{total_rays}, 顶部总面积: {top_face_area_cm2:.1f}平方厘米"
        p.addUserDebugText(detail_text, text_pos2, [0, 0, 0], 1.2)

# 合并后的回调函数，用于记录所有指标
class RewardMetricsCallback(BaseCallback):
    """
    回调函数，记录训练指标到TensorBoard
    """
    def __init__(self, verbose=0, flush_freq=100, record_freq=50):  # 大幅降低记录和刷新频率
        super(RewardMetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.flush_freq = flush_freq  # 每100步刷新一次
        self.record_freq = record_freq  # 每50步记录一次步级指标
        
        # 添加步级别的追踪
        self.step_rewards = []
        self.step_successes = []
        self.optimal_choices = []
        
        # 添加累积统计
        self.total_steps = 0
        self.total_episodes = 0
        
        # 用于累积指标的缓存
        self.step_cache = {
            'rewards': [],
            'successes': [],
            'exposures': [],
            'optimal_choices': []
        }
        
    def _on_step(self):
        # 增加总步数
        self.total_steps += 1
        
        # 记录当前步的奖励
        current_reward = self.locals['rewards'][0] if len(self.locals['rewards']) > 0 else 0
        self.current_episode_reward += current_reward
        self.current_episode_length += 1
        self.step_rewards.append(current_reward)
        
        # 将数据添加到缓存中
        self.step_cache['rewards'].append(current_reward)
        
        # 从info中收集数据到缓存
        if self.locals.get('infos') and len(self.locals['infos']) > 0:
            for i, info in enumerate(self.locals['infos']):
                if not info:
                    continue
                
                # 缓存成功信息
                if 'success' in info:
                    success_value = float(info['success'])
                    self.step_successes.append(success_value)
                    self.step_cache['successes'].append(success_value)
                
                # 缓存最佳选择信息
                if 'optimal_choice' in info:
                    optimal_choice_value = float(info['optimal_choice'])
                    self.optimal_choices.append(optimal_choice_value)
                    self.step_cache['optimal_choices'].append(optimal_choice_value)
                
                # 缓存暴露度信息
                if 'object_exposure' in info:
                    exposures = list(info['object_exposure'].values())
                    if exposures:
                        avg_exposure = sum(exposures) / len(exposures)
                        self.step_cache['exposures'].append(avg_exposure)
                
                # 只处理第一个环境的info
                break
        
        # 只在指定频率记录步级指标
        if self.total_steps % self.record_freq == 0:
            # 记录基础训练指标
            if self.step_cache['rewards']:
                avg_reward = sum(self.step_cache['rewards']) / len(self.step_cache['rewards'])
                self.logger.record('train/avg_step_reward', avg_reward)
            
            # 记录成功率指标
            if self.step_cache['successes']:
                recent_success_rate = sum(self.step_cache['successes']) / len(self.step_cache['successes'])
                self.logger.record('success/recent_success_rate', recent_success_rate)
            
            # 记录最佳选择率
            if self.step_cache['optimal_choices']:
                optimal_rate = sum(self.step_cache['optimal_choices']) / len(self.step_cache['optimal_choices'])
                self.logger.record('strategy/optimal_choice_rate', optimal_rate)
            
            # 记录暴露度
            if self.step_cache['exposures']:
                avg_exposure = sum(self.step_cache['exposures']) / len(self.step_cache['exposures'])
                self.logger.record('environment/average_exposure', avg_exposure)
            
            # 记录总体进度
            self.logger.record('train/total_steps', self.total_steps)
            
            # 清空缓存
            for key in self.step_cache:
                self.step_cache[key] = []
        
        # 检查回合是否结束
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for i, done in enumerate(dones):
            if done:
                self.total_episodes += 1
                
                # 记录回合级别的统计
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                
                # 记录回合结束时的成功信息
                episode_success_count = 0
                if len(infos) > i and infos[i] and 'success_count' in infos[i]:
                    episode_success_count = infos[i]['success_count']
                
                self.episode_successes.append(episode_success_count)
                
                # 回合结束时总是记录关键指标
                self.logger.record('training/episode_reward', self.current_episode_reward)
                self.logger.record('training/episode_length', self.current_episode_length)
                self.logger.record('training/episode_success_count', episode_success_count)
                self.logger.record('training/total_episodes', self.total_episodes)
                
                # 计算和记录滚动平均值
                if len(self.episode_rewards) > 0:
                    # 最近10个回合的统计
                    recent_rewards = self.episode_rewards[-10:]
                    recent_successes = self.episode_successes[-10:]
                    recent_lengths = self.episode_lengths[-10:]
                    
                    mean_reward_10 = np.mean(recent_rewards)
                    mean_success_10 = np.mean(recent_successes)
                    mean_length_10 = np.mean(recent_lengths)
                    
                    self.logger.record("training/mean_reward_10ep", mean_reward_10)
                    self.logger.record("success/mean_success_count_10ep", mean_success_10)
                    self.logger.record("training/mean_length_10ep", mean_length_10)
                    
                    # 最近100个回合的统计（如果有足够数据）
                    if len(self.episode_rewards) >= 100:
                        recent_rewards_100 = self.episode_rewards[-100:]
                        recent_successes_100 = self.episode_successes[-100:]
                        recent_lengths_100 = self.episode_lengths[-100:]
                        
                        mean_reward_100 = np.mean(recent_rewards_100)
                        mean_success_100 = np.mean(recent_successes_100)
                        mean_length_100 = np.mean(recent_lengths_100)
                        
                        self.logger.record("training/mean_reward_100ep", mean_reward_100)
                        self.logger.record("success/mean_success_count_100ep", mean_success_100)
                        self.logger.record("training/mean_length_100ep", mean_length_100)
                    
                    # 记录关键奖励统计
                    self.logger.record("rewards/mean_reward", np.mean(self.episode_rewards[-50:]))  # 减少到最近50回合
                    
                    # 计算成功率
                    if len(recent_successes) > 0:
                        success_rate_10 = mean_success_10 / max(mean_length_10, 1.0)
                        self.logger.record("success/success_rate_10ep", success_rate_10)
                        
                        if len(self.episode_successes) >= 100:
                            success_rate_100 = mean_success_100 / max(mean_length_100, 1.0)
                            self.logger.record("success/success_rate_100ep", success_rate_100)
                    
                    # 记录效率指标
                    reward_per_step = mean_reward_10 / max(mean_length_10, 1.0)
                    self.logger.record("training/reward_per_step", reward_per_step)
                
                # 重置回合计数器
                self.current_episode_reward = 0
                self.current_episode_length = 0
                
                # 回合结束时强制刷新数据到TensorBoard
                self.logger.dump(self.num_timesteps)
        
        # 定期刷新数据（频率大幅降低）
        if self.total_steps % self.flush_freq == 0:
            self.logger.dump(self.total_steps)
        
        return True

    def _on_training_start(self):
        """训练开始时的初始化"""
        self.logger.record('training/training_started', 1)
        self.logger.dump(0)

    def _on_training_end(self):
        """训练结束时的最终统计"""
        if len(self.episode_rewards) > 0:
            self.logger.record('training/final_mean_reward', np.mean(self.episode_rewards))
            self.logger.record('training/final_mean_success', np.mean(self.episode_successes))
            self.logger.record('training/total_episodes_completed', len(self.episode_rewards))
        
        self.logger.record('training/training_completed', 1)
        self.logger.dump(self.num_timesteps)