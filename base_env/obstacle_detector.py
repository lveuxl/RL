"""
智能障碍物检测系统
为RRT运动规划提供精确的障碍物点云信息
"""

import numpy as np
import sapien
import trimesh
import torch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ sklearn未安装，将使用简化的聚类算法")
    SKLEARN_AVAILABLE = False


class ObstacleDetector:
    """智能障碍物检测器"""
    
    def __init__(
        self,
        point_density: int = 256,
        safety_margin: float = 0.02,
        cluster_objects: bool = True,
        use_bounding_boxes: bool = True,
        debug: bool = False
    ):
        self.point_density = point_density
        self.safety_margin = safety_margin
        self.cluster_objects = cluster_objects
        self.use_bounding_boxes = use_bounding_boxes
        self.debug = debug
        
        # 缓存系统
        self.obstacle_cache = {}
        self.last_update_time = 0
        
        print(f"✅ 障碍物检测器初始化完成")
        print(f"   点云密度: {self.point_density}")
        print(f"   安全边距: {self.safety_margin}m")
        print(f"   对象聚类: {'✅启用' if self.cluster_objects else '❌禁用'}")
    
    def detect_obstacles(
        self, 
        scene_objects: List[sapien.Actor],
        exclude_objects: Optional[List[int]] = None,
        target_object: Optional[sapien.Actor] = None
    ) -> np.ndarray:
        """
        检测场景中的障碍物并生成点云
        
        Args:
            scene_objects: 场景中的所有物体
            exclude_objects: 要排除的物体索引列表
            target_object: 目标物体（会被特殊处理）
            
        Returns:
            障碍物点云数组 [N, 3]
        """
        exclude_objects = exclude_objects or []
        all_points = []
        
        for idx, obj in enumerate(scene_objects):
            if idx in exclude_objects:
                continue
                
            # 跳过目标物体（给抓取留出空间）
            if target_object and obj == target_object:
                continue
            
            # 获取物体点云
            obj_points = self._extract_object_points(obj, idx)
            if obj_points is not None and len(obj_points) > 0:
                all_points.append(obj_points)
        
        if not all_points:
            return np.array([]).reshape(0, 3)
        
        # 合并所有点云
        combined_points = np.vstack(all_points)
        
        # 应用安全边距
        if self.safety_margin > 0:
            combined_points = self._apply_safety_margin(combined_points)
        
        # 聚类处理（可选）
        if self.cluster_objects and SKLEARN_AVAILABLE:
            combined_points = self._cluster_and_filter(combined_points)
        
        if self.debug:
            print(f"生成障碍物点云: {len(combined_points)} 个点")
        
        return combined_points
    
    def _extract_object_points(self, obj: sapien.Actor, obj_idx: int) -> Optional[np.ndarray]:
        """从物体提取点云"""
        try:
            # 检查缓存
            obj_id = f"{obj_idx}_{hash(str(obj.pose))}"
            if obj_id in self.obstacle_cache:
                return self.obstacle_cache[obj_id]
            
            # 获取物体位姿
            pose = obj.pose
            if isinstance(pose.p, torch.Tensor):
                position = pose.p.cpu().numpy()
                quaternion = pose.q.cpu().numpy()
            else:
                position = pose.p
                quaternion = pose.q
            
            # 尝试不同的点云生成策略
            points = None
            
            # 策略1: 使用碰撞形状
            points = self._extract_from_collision_shapes(obj, position, quaternion)
            
            # 策略2: 使用视觉形状（如果碰撞形状失败）
            if points is None or len(points) == 0:
                points = self._extract_from_visual_shapes(obj, position, quaternion)
            
            # 策略3: 使用边界框（最后的回退方案）
            if (points is None or len(points) == 0) and self.use_bounding_boxes:
                points = self._extract_from_bounding_box(obj, position, quaternion)
            
            # 缓存结果
            if points is not None:
                self.obstacle_cache[obj_id] = points
                
            return points
            
        except Exception as e:
            if self.debug:
                print(f"提取物体{obj_idx}点云失败: {e}")
            return None
    
    def _extract_from_collision_shapes(
        self, 
        obj: sapien.Actor, 
        position: np.ndarray, 
        quaternion: np.ndarray
    ) -> Optional[np.ndarray]:
        """从碰撞形状提取点云"""
        try:
            shapes = obj.get_collision_shapes()
            all_points = []
            
            for shape in shapes:
                geometry = shape.geometry
                shape_pose = shape.get_local_pose()
                
                # 组合物体位姿和形状局部位姿
                world_pose = obj.pose * shape_pose
                transform_matrix = world_pose.to_transformation_matrix()
                
                # 根据几何类型生成点云
                if hasattr(geometry, 'type'):
                    if 'box' in str(geometry.type).lower():
                        half_sizes = geometry.half_sizes
                        points = self._sample_box_points(half_sizes * 2, transform_matrix)
                    elif 'sphere' in str(geometry.type).lower():
                        radius = geometry.radius
                        points = self._sample_sphere_points(radius, transform_matrix)
                    elif 'cylinder' in str(geometry.type).lower():
                        radius = geometry.radius
                        height = geometry.half_height * 2
                        points = self._sample_cylinder_points(radius, height, transform_matrix)
                    else:
                        # 未知几何类型，使用边界框
                        points = self._sample_box_points([0.05, 0.05, 0.05], transform_matrix)
                    
                    if points is not None and len(points) > 0:
                        all_points.append(points)
            
            return np.vstack(all_points) if all_points else None
            
        except Exception as e:
            if self.debug:
                print(f"从碰撞形状提取点云失败: {e}")
            return None
    
    def _extract_from_visual_shapes(
        self, 
        obj: sapien.Actor, 
        position: np.ndarray, 
        quaternion: np.ndarray
    ) -> Optional[np.ndarray]:
        """从视觉形状提取点云"""
        try:
            # 这里可以添加从视觉mesh提取点云的代码
            # 目前作为占位符
            return None
        except:
            return None
    
    def _extract_from_bounding_box(
        self, 
        obj: sapien.Actor, 
        position: np.ndarray, 
        quaternion: np.ndarray
    ) -> Optional[np.ndarray]:
        """从边界框提取点云"""
        try:
            # 使用标准尺寸的边界框
            box_size = np.array([0.06, 0.06, 0.06])  # 6cm立方体
            
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = self._quat_to_rotation_matrix(quaternion)
            transform_matrix[:3, 3] = position
            
            return self._sample_box_points(box_size, transform_matrix)
            
        except Exception as e:
            if self.debug:
                print(f"从边界框提取点云失败: {e}")
            return None
    
    def _sample_box_points(self, size: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """采样盒子点云"""
        try:
            box = trimesh.creation.box(extents=size, transform=transform)
            points, _ = trimesh.sample.sample_surface(box, self.point_density)
            return points
        except:
            # 手动采样盒子表面
            return self._manual_box_sampling(size, transform)
    
    def _sample_sphere_points(self, radius: float, transform: np.ndarray) -> np.ndarray:
        """采样球体点云"""
        try:
            sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
            sphere.apply_transform(transform)
            points, _ = trimesh.sample.sample_surface(sphere, self.point_density)
            return points
        except:
            # 手动采样球面
            return self._manual_sphere_sampling(radius, transform)
    
    def _sample_cylinder_points(self, radius: float, height: float, transform: np.ndarray) -> np.ndarray:
        """采样圆柱体点云"""
        try:
            cylinder = trimesh.creation.cylinder(radius=radius, height=height, transform=transform)
            points, _ = trimesh.sample.sample_surface(cylinder, self.point_density)
            return points
        except:
            # 手动采样圆柱面
            return self._manual_cylinder_sampling(radius, height, transform)
    
    def _manual_box_sampling(self, size: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """手动盒子采样（fallback）"""
        half_size = size / 2
        n_points_per_face = self.point_density // 6
        
        points = []
        
        # 6个面
        faces = [
            ([-half_size[0], 0, 0], [0, half_size[1], half_size[2]]),  # -X面
            ([+half_size[0], 0, 0], [0, half_size[1], half_size[2]]),  # +X面
            ([0, -half_size[1], 0], [half_size[0], 0, half_size[2]]),  # -Y面
            ([0, +half_size[1], 0], [half_size[0], 0, half_size[2]]),  # +Y面
            ([0, 0, -half_size[2]], [half_size[0], half_size[1], 0]),  # -Z面
            ([0, 0, +half_size[2]], [half_size[0], half_size[1], 0]),  # +Z面
        ]
        
        for center, extent in faces:
            face_points = np.random.uniform(-1, 1, (n_points_per_face, 3))
            face_points = face_points * extent + center
            points.append(face_points)
        
        points = np.vstack(points)
        
        # 应用变换
        homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (transform @ homogeneous.T).T
        
        return transformed[:, :3]
    
    def _manual_sphere_sampling(self, radius: float, transform: np.ndarray) -> np.ndarray:
        """手动球面采样（fallback）"""
        # 球面均匀采样
        u = np.random.uniform(0, 1, self.point_density)
        v = np.random.uniform(0, 1, self.point_density)
        
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        points = np.column_stack([x, y, z])
        
        # 应用变换
        homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (transform @ homogeneous.T).T
        
        return transformed[:, :3]
    
    def _manual_cylinder_sampling(self, radius: float, height: float, transform: np.ndarray) -> np.ndarray:
        """手动圆柱面采样（fallback）"""
        n_side = int(self.point_density * 0.8)  # 80%在侧面
        n_cap = int(self.point_density * 0.1)   # 10%在每个端面
        
        points = []
        
        # 侧面采样
        theta = np.random.uniform(0, 2*np.pi, n_side)
        z = np.random.uniform(-height/2, height/2, n_side)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        side_points = np.column_stack([x, y, z])
        points.append(side_points)
        
        # 上端面
        r = np.random.uniform(0, radius, n_cap)
        theta = np.random.uniform(0, 2*np.pi, n_cap)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full(n_cap, height/2)
        top_points = np.column_stack([x, y, z])
        points.append(top_points)
        
        # 下端面
        z = np.full(n_cap, -height/2)
        bottom_points = np.column_stack([x, y, z])
        points.append(bottom_points)
        
        points = np.vstack(points)
        
        # 应用变换
        homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (transform @ homogeneous.T).T
        
        return transformed[:, :3]
    
    def _apply_safety_margin(self, points: np.ndarray) -> np.ndarray:
        """应用安全边距，扩展障碍物点云"""
        if self.safety_margin <= 0:
            return points
        
        # 简单策略：为每个点添加随机偏移
        n_expanded_points = int(len(points) * 1.2)  # 扩展20%的点
        
        expanded_points = []
        for i in range(n_expanded_points):
            base_point = points[i % len(points)]
            # 在安全边距内随机偏移
            offset = np.random.normal(0, self.safety_margin/3, 3)
            expanded_points.append(base_point + offset)
        
        return np.vstack([points, expanded_points])
    
    def _cluster_and_filter(self, points: np.ndarray) -> np.ndarray:
        """聚类和过滤点云"""
        if len(points) < 10 or not SKLEARN_AVAILABLE:
            return points
        
        try:
            # 使用DBSCAN聚类
            clustering = DBSCAN(eps=0.02, min_samples=5).fit(points)
            labels = clustering.labels_
            
            # 只保留主要聚类的点
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1:
                return points
            
            # 保留最大的几个聚类
            cluster_sizes = []
            for label in unique_labels:
                if label != -1:  # 排除噪声点
                    cluster_sizes.append((label, np.sum(labels == label)))
            
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            keep_labels = [x[0] for x in cluster_sizes[:5]]  # 保留前5个最大聚类
            
            mask = np.isin(labels, keep_labels)
            return points[mask]
            
        except Exception as e:
            if self.debug:
                print(f"聚类处理失败: {e}")
            return points
    
    def add_table_obstacle(
        self, 
        table_pose: sapien.Pose, 
        table_size: np.ndarray
    ) -> np.ndarray:
        """添加桌面障碍物"""
        transform = table_pose.to_transformation_matrix()
        return self._sample_box_points(table_size, transform)
    
    def add_wall_obstacles(
        self, 
        workspace_bounds: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """添加工作空间边界墙壁"""
        wall_points = []
        wall_thickness = 0.01
        wall_height = 0.5
        
        x_min, x_max = workspace_bounds.get('x', (-0.5, 0.5))
        y_min, y_max = workspace_bounds.get('y', (-0.5, 0.5))
        z_min, z_max = workspace_bounds.get('z', (0, 0.5))
        
        # 创建4面墙
        walls = [
            # 前墙 (x_min)
            (x_min - wall_thickness/2, (y_min + y_max)/2, (z_min + z_max)/2, 
             wall_thickness, y_max - y_min, wall_height),
            # 后墙 (x_max)  
            (x_max + wall_thickness/2, (y_min + y_max)/2, (z_min + z_max)/2,
             wall_thickness, y_max - y_min, wall_height),
            # 左墙 (y_min)
            ((x_min + x_max)/2, y_min - wall_thickness/2, (z_min + z_max)/2,
             x_max - x_min, wall_thickness, wall_height),
            # 右墙 (y_max)
            ((x_min + x_max)/2, y_max + wall_thickness/2, (z_min + z_max)/2,
             x_max - x_min, wall_thickness, wall_height),
        ]
        
        for x, y, z, w, d, h in walls:
            wall_pose = sapien.Pose(p=[x, y, z])
            wall_size = np.array([w, d, h])
            points = self.add_table_obstacle(wall_pose, wall_size)
            if points is not None and len(points) > 0:
                wall_points.append(points)
        
        return np.vstack(wall_points) if wall_points else np.array([]).reshape(0, 3)
    
    def clear_cache(self):
        """清除缓存"""
        self.obstacle_cache.clear()
    
    @staticmethod
    def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        w, x, y, z = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]
        ])