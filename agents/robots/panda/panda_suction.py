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
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4]
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

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

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
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
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
                # 创建固定约束来模拟吸盘效果
                # 使用SAPIEN的约束系统
                constraint = self.scene.create_drive(
                    self.suction_link,
                    target_object,
                    constraint_type="fixed"
                )
                
                self.suction_constraints[target_object.name] = constraint
                self.is_suction_active = True
                self.current_suction_object = target_object
                return True
            except Exception as e:
                print(f"创建吸盘约束失败: {e}")
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
                self.scene.remove_drive(constraint)
                del self.suction_constraints[constraint_name]
            
            self.is_suction_active = False
            self.current_suction_object = None
            return True
        except Exception as e:
            print(f"移除吸盘约束失败: {e}")
            return False

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