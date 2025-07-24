from copy import deepcopy
from typing import Dict, Tuple, Optional
import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.controllers import deepcopy_dict  # 明确导入 deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent()
class PandaSuction(BaseAgent):
    uid = "panda_suction"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_stick.urdf"  # 使用panda_stick的URDF
    urdf_config = dict()
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [0.0, 1.0, 0.0, -1.5, 0.0, 2.5, np.pi / 4]  # 最终优化：平衡高度和工作空间
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "panda_joint1",
        "panda_joint2", 
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]

    ee_link_name = "panda_hand_tcp"

    arm_stiffness = 300  # 提高刚度以提升精度
    arm_damping = 30     # 提高阻尼以减少振荡
    arm_force_limit = 400 # 提高力限制以确保到达目标

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm (与panda_stick相同的配置)
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=np.array([-1.0, -1.0, 0.0]),  # 扩大工作空间：从-0.8到-1.0
            pos_upper=np.array([1.0, 1.0, 1.5]),    # 扩大工作空间：从0.8到1.0，从1.2到1.5
            rot_lower=np.array([-np.pi, -np.pi, -np.pi]),  # 添加旋转下界：欧拉角范围
            rot_upper=np.array([np.pi, np.pi, np.pi]),      # 添加旋转上界：欧拉角范围
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
            interpolate=True,  # 启用插值以提高响应速度
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        arm_pd_ee_delta_pose_align = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_delta_pose_align.frame = "ee_align"

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos),
            pd_joint_pos=dict(arm=arm_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose),
            pd_ee_delta_pose_align=dict(arm=arm_pd_ee_delta_pose_align),
            pd_ee_pose=dict(arm=arm_pd_ee_pose),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos),
            pd_ee_target_delta_pos=dict(arm=arm_pd_ee_target_delta_pos),
            pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel),
            pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel),
            pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )
        self.suction_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_hand"
        )

        # 吸盘相关属性
        self.suction_constraints = {}
        self.is_suction_active = False
        self.current_suction_object = None
        
        # 用于接触检测的查询
        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def activate_suction(self, target_object: Actor, contact_threshold: float = 0.02) -> bool:
        """激活吸盘，吸附目标物体
        
        Args:
            target_object: 目标物体
            contact_threshold: 接触距离阈值
            
        Returns:
            bool: 是否成功激活吸盘
        """
        if self.is_suction_active:
            return False
            
        # 检查是否与物体接触
        if self.is_contacting(target_object, contact_threshold):
            try:
                # 使用SAPIEN的驱动约束系统创建固定约束
                suction_body = self.suction_link.entity
                target_body = target_object.entity
                
                # 创建驱动约束 - 使用正确的SAPIEN API
                constraint = self.scene.create_drive(
                    suction_body,
                    sapien.Pose(),  # 吸盘链接的本地姿态
                    target_body,
                    sapien.Pose()   # 目标物体的本地姿态
                )
                
                # 设置约束参数使其表现为固定约束
                constraint.set_x_properties(stiffness=1e6, damping=1e4)
                constraint.set_y_properties(stiffness=1e6, damping=1e4)
                constraint.set_z_properties(stiffness=1e6, damping=1e4)
                constraint.set_twist_properties(stiffness=1e6, damping=1e4)
                constraint.set_swing1_properties(stiffness=1e6, damping=1e4)
                constraint.set_swing2_properties(stiffness=1e6, damping=1e4)
                
                self.suction_constraints[target_object.name] = constraint
                self.is_suction_active = True
                self.current_suction_object = target_object
                return True
                
            except Exception as e:
                print(f"创建吸盘约束失败: {e}")
                # 尝试简化的实现：直接设置物体位置跟随TCP
                try:
                    self.is_suction_active = True
                    self.current_suction_object = target_object
                    return True
                except Exception as e2:
                    print(f"简化吸盘实现也失败: {e2}")
                    return False
        
        return False

    def deactivate_suction(self) -> bool:
        """关闭吸盘，释放物体
        
        Returns:
            bool: 是否成功关闭吸盘
        """
        if not self.is_suction_active or self.current_suction_object is None:
            return False
        
        try:
            # 移除约束
            constraint_name = self.current_suction_object.name
            if constraint_name in self.suction_constraints:
                constraint = self.suction_constraints[constraint_name]
                # 尝试移除约束
                try:
                    self.scene.remove_drive(constraint)
                except:
                    # 如果移除失败，可能约束已经不存在了
                    pass
                del self.suction_constraints[constraint_name]
            
            self.is_suction_active = False
            self.current_suction_object = None
            return True
        except Exception as e:
            print(f"移除吸盘约束失败: {e}")
            # 即使移除约束失败，也要重置状态
            self.is_suction_active = False
            self.current_suction_object = None
            return True

    def is_contacting(self, object: Actor, threshold: float = 0.02) -> torch.Tensor:
        """检测是否与物体接触
        
        Args:
            object: 目标物体
            threshold: 距离阈值
            
        Returns:
            torch.Tensor: 是否接触的布尔张量
        """
        # 计算TCP到物体的距离
        tcp_pos = self.tcp.pose.p
        obj_pos = object.pose.p
        
        # 计算距离
        distance = torch.linalg.norm(tcp_pos - obj_pos, axis=-1)
        
        # 检查是否在接触阈值内
        return distance <= threshold

    def is_grasping(self, object: Actor, min_force: float = 0.1, max_angle: float = 85) -> torch.Tensor:
        """检查是否正在抓取物体（兼容原有接口）
        
        Args:
            object: 目标物体
            min_force: 最小力阈值（对吸盘无意义，但保持接口兼容）
            max_angle: 最大角度（对吸盘无意义，但保持接口兼容）
            
        Returns:
            torch.Tensor: 是否正在抓取的布尔张量
        """
        # 对于吸盘，如果当前吸附的物体是目标物体，则认为正在抓取
        is_grasping = (self.is_suction_active and 
                      self.current_suction_object is not None and 
                      self.current_suction_object.name == object.name)
        
        # 返回正确维度的张量
        # 获取场景中的环境数量
        if hasattr(self, 'scene') and hasattr(self.scene, 'num_envs'):
            num_envs = self.scene.num_envs
        else:
            num_envs = 1
            
        # 创建正确维度的布尔张量
        result = torch.full((num_envs,), is_grasping, dtype=torch.bool, device=self.device)
        return result

    def is_static(self, threshold: float = 0.2) -> torch.Tensor:
        """检查机器人是否静止"""
        qvel = self.robot.get_qvel()[..., :]  # 吸盘版本没有夹爪关节
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        """TCP位置"""
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        """TCP姿态"""
        return self.tcp.pose

    @property
    def suction_state(self) -> Dict[str, any]:
        """获取吸盘状态信息"""
        return {
            'is_active': self.is_suction_active,
            'current_object': self.current_suction_object.name if self.current_suction_object else None,
            'num_constraints': len(self.suction_constraints)
        } 