import pybullet as p
import time
import math
import random
from config import ENV_CONFIG, PHYSICS_CONFIG, TABLE_CONFIG, ROBOT_CONFIG
from utils import random_objects, create_invisible_wall

def setup_scene():
    """设置场景，包括加载物理对象和设置物理参数"""
    # 加载URDF文件
    plane_id = p.loadURDF("plane/plane.urdf",useFixedBase=True)
    # plane_id = p.loadURDF("floor/plane.urdf",useFixedBase=True)
    table_id = p.loadURDF("table/table.urdf", basePosition=[1.0, -0.2, 0.0], baseOrientation=[0, 0, 0.7071, 0.7071], useFixedBase=True, globalScaling=1)
    table_id_2 = p.loadURDF("table/table.urdf", basePosition=[0, -0.2, 0.0], baseOrientation=[0, 0, 0.7071, 0.7071], useFixedBase=True, globalScaling=1)
    table_id_3 = p.loadURDF("table/table.urdf", basePosition=[1.0, -1.3, 0.0], baseOrientation=[0, 0, 0.7071, 0.7071], useFixedBase=True, globalScaling=1)
    table_id_4 = p.loadURDF("table/table.urdf", basePosition=[0, -1.3, 0.0], baseOrientation=[0, 0, 0.7071, 0.7071], useFixedBase=True, globalScaling=1)
    # table_id = p.loadURDF("table_new/table.urdf", basePosition=[1.0, -0.2, 0.0], baseOrientation=[0, 0, 0.7071, 0.7071], useFixedBase = True, globalScaling = 6)
    
    # 在GUI模式下加载纹理(避免在DIRECT模式下出错)
    if p.getConnectionInfo()['connectionMethod'] == p.GUI:
        try:
            texture1 = p.loadTexture('floor.jpg')
            texture2 = p.loadTexture('wood.jpg')
            p.changeVisualShape(plane_id, -1, textureUniqueId=texture1)
            p.changeVisualShape(table_id, -1, textureUniqueId=texture2)
            p.changeVisualShape(table_id_2, -1, textureUniqueId=texture2)
            p.changeVisualShape(table_id_3, -1, textureUniqueId=texture2)
            p.changeVisualShape(table_id_4, -1, textureUniqueId=texture2)
        except Exception as e:
            print(f"加载纹理时出现错误，忽略: {e}")
    
    # 创建边界墙
    wall_ids = setup_walls()
    
    # 设置重力
    p.setGravity(0, 0, -50)
    
    # 初始化机械臂
    robot_id, num_joints, kuka_end_effector_idx, kuka_cid = setup_robot()
    
    return {
        "plane_id": plane_id,
        "table_id": table_id,
        "table_id_2": table_id_2,
        "table_id_3": table_id_3,
        "table_id_4": table_id_4,
        "wall_ids": wall_ids,
        "robot_id": robot_id,
        "num_joints": num_joints,
        "kuka_end_effector_idx": kuka_end_effector_idx,
        "kuka_cid": kuka_cid
    }

def setup_walls():
    """创建墙壁"""
    table_length = TABLE_CONFIG["length"]
    table_width = TABLE_CONFIG["width"]
    wall_height = TABLE_CONFIG["wall_height"]
    wall_thickness = TABLE_CONFIG["wall_thickness"]
    
    # 创建四个边界墙
    # wall_id1 = create_invisible_wall(
    #     position=[1.0 - table_length/2 - wall_thickness, -0.2, wall_height/2],  # 左侧
    #     half_extents=[wall_thickness, table_width/2 + wall_thickness, wall_height/2]
    # )

    # 创建墙壁
    center = [0.75, 0.15, 0.4]
    half_extents = [0.01, 0.25, wall_height/2]
    wall_id1 = create_invisible_wall(center, half_extents)
    
    wall_id2 = create_invisible_wall(
        position=[0.8 + table_length/2 + wall_thickness, -0.2, wall_height/2],
        half_extents=[wall_thickness, table_width/2 + wall_thickness, wall_height/2]
    )
    
    wall_id3 = create_invisible_wall(
        position=[1.0, -0.2 - table_width/2 - wall_thickness, wall_height/2],  # 前侧
        half_extents=[table_length/2 + wall_thickness, wall_thickness, wall_height/2]
    )

    wall_id4 = create_invisible_wall(
        position=[1.0, -0.2 + table_width/2 + wall_thickness, wall_height/2],  # 后侧
        half_extents=[table_length/2 + wall_thickness, wall_thickness, wall_height/2]
    )

    wall_id5 = create_invisible_wall(
        position=[0.3, -0.2 - table_width/2 - wall_thickness, wall_height/2],  # 前侧
        half_extents=[table_length/2 + wall_thickness, wall_thickness, wall_height/2]
    )

    wall_id6 = create_invisible_wall(
        position=[0.3 - table_length/2 - wall_thickness, -0.2, wall_height/2],  # 左侧
        half_extents=[wall_thickness, table_width/2 + wall_thickness, wall_height/2]
    )

    wall_id7 = create_invisible_wall(
        position=[0.3, -0.2 + table_width/2 + wall_thickness, wall_height/2],  # 后侧
        half_extents=[table_length/2 + wall_thickness, wall_thickness, wall_height/2]
    )

    # 返回一个字典，包含所有墙壁ID
    return {
        "wall_id1": wall_id1,
        "wall_id2": wall_id2,
        "wall_id3": wall_id3,
        "wall_id4": wall_id4,
        "wall_id5": wall_id5,
        "wall_id6": wall_id6,
        "wall_id7": wall_id7
    }

def setup_robot():
    """初始化机械臂"""
    robot_id = p.loadURDF("panda_bullet/panda_suction.urdf", 
                         ROBOT_CONFIG["base_position"], 
                         ROBOT_CONFIG["base_orientation"], 
                         useFixedBase=True, globalScaling=0.8)
    
    # 设置初始关节位置
    jointPositions = ROBOT_CONFIG["joint_positions"]
    num_joints = p.getNumJoints(robot_id)
    for jointIndex in range(num_joints):
        p.resetJointState(robot_id, jointIndex, jointPositions[jointIndex])
        p.setJointMotorControl2(robot_id, jointIndex, p.POSITION_CONTROL, jointPositions[jointIndex], 0)
    
    # 指定末端关节索引和吸盘状态
    kuka_end_effector_idx = ROBOT_CONFIG["end_effector_idx"]
    kuka_cid = None
    
    return robot_id, num_joints, kuka_end_effector_idx, kuka_cid

def generate_objects(obj_num):
    """生成物体并处理超出范围的对象"""
    obj_ids = []
    
    # 生成物体
    for _ in range(int(obj_num / 8)):
        object_ids = random_objects()
        for _ in range(180):
            p.stepSimulation()
            time.sleep(1 / 240)
        obj_ids.extend(object_ids)
    
    # 移除超出范围的物体

    # 定义圆的参数：圆心和半径
    circle_center = ENV_CONFIG["circle_center"]
    radius_max = ENV_CONFIG["radius_max"]
    radius_min = ENV_CONFIG["radius_min"]
    
    for obj in obj_ids:
        obj_id = obj['id']
        obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
        x, y, _ = obj_pos
        dx = x - circle_center[0]
        dy = y - circle_center[1]
        distance = math.sqrt(dx**2 + dy**2)
        # 如果物体在圆外，则删除
        if distance > radius_max or distance < radius_min:
            #print(f"物体{obj_id}位置为({x:.2f}, {y:.2f}),已超过机械臂抓取范围，移除")
            p.removeBody(obj_id)
            obj_ids.remove(obj)
    
    # 补充物体
    object_ids = random_objects(obj_num - len(obj_ids))
    obj_ids.extend(object_ids)
    for _ in range(120): # 240步（大约4秒时间）
        p.stepSimulation()
        time.sleep(1 / 240) # Time in seconds. 仿真引擎暂停1/240秒，然后再进入下一个时间步，不影响仿真结果。
    
    
    return obj_ids