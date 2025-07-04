import pybullet as p
import pybullet_data
import random
import numpy as np
import time
import cv2
import os
import json
import math
import csv
import sys
from config import ENV_CONFIG, PHYSICS_CONFIG, ROBOT_CONFIG


def random_objects(num_objects=8):
    category_map = ENV_CONFIG["category_map"]

    object_ids = []
    # 机械臂基座在xy平面上的位置
    center_x, center_y = 0.5, -0.2
    # 定义弧形区域的角度范围（单位：弧度）
    theta_min = math.radians(0)   # -60°
    theta_max = math.radians(51.7)   # 约51.7°
    # 定义半径范围（根据测试数据调整）
    r_min = 0.55
    r_max = 0.63

    for _ in range(num_objects):
        # 随机选择物体，目前有cube系列
        #obj = random.choice(["CrackerBox", "GelatinBox", "PottedMeatCan", "MustardBottle"])
        # 随机选择物体，确保只选择category_map中存在的类别
        obj = random.choice(list(category_map.keys()))
        # 根据面积均匀采样半径
        u = random.random()
        r = math.sqrt(u * (r_max**2 - r_min**2) + r_min**2)
        
        # 随机生成角度
        theta = random.uniform(theta_min, theta_max)
        
        # 将极坐标转换为笛卡尔坐标
        x_pos = center_x + r * math.cos(theta)
        y_pos = center_y + r * math.sin(theta)
        z_pos = 0.75  # 固定的z高度

        # 根据物体类型设定缩放比例
        scale = 0.1 if obj in ["SugarBox", "PuddingBox"] else 1
        
        # 加载URDF模型
        obj_id = p.loadURDF(f"ycb_objects/Ycb{obj}/model.urdf", basePosition=[x_pos, y_pos, z_pos], globalScaling=scale)
        object_ids.append({
            "id": obj_id,
            "category": obj,
            "skip": False,
            "suck_state": False,
            "suck_times": 0
        })
        
    return object_ids


# def save_data(object_ids, camera_data, random_seed, rgbImg, depthImg, segImg, rgb_dir, depth_dir):
#     if segImg is not None:
#         # 假设 segImg 是一个二维数组，值代表不同的类别或对象
#         # 转换为 uint8 类型
#         segImg_uint8 = segImg.astype(np.uint8)

#         # 如果有多个对象，可以遍历每个唯一的标签
#         unique_labels = np.unique(segImg_uint8)
#         # 移除背景标签(plane_id),桌子标签(table_id)
        
#         unique_labels = unique_labels[(unique_labels != plane_id) & (unique_labels != table_id) & (unique_labels != table_id_2)
#                                       & (unique_labels != table_id_3) &(unique_labels != table_id_4) &(unique_labels != wall_id1) 
#                                       & (unique_labels != wall_id2)
#                                       & (unique_labels != wall_id3) & (unique_labels != wall_id4) & (unique_labels != wall_id5) 
#                                       & (unique_labels != wall_id6) & (unique_labels != wall_id7)
#                                       & (unique_labels != robot_id)]
        
#         obj_id = 0
#         for label in unique_labels:

#             #=============目前还没有加入机械臂======================
#             for obj in object_ids:
#                 if obj["id"] == label:
#                     category_id = category_map[obj["category"]]
#                     break
#             #===================================================
            
#             obj_id += 1
#             # 创建二值掩码
#             mask = np.uint8(segImg_uint8 == label) * 255
#             # 寻找轮廓
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             for contour in contours:    #这里的循环是为了避免一个物体有多个轮廓，其实在我们的场景下不会发生
#                 # 将轮廓坐标转换为列表
#                 contour_list = contour.flatten().tolist()
#                 # 过滤掉不闭合的轮廓
#                 if len(contour_list) >= 6:
#                      #得到bounding box信息
#                     contour_np = np.array(contour).reshape(-1, 2)
#                     x, y, w, h = cv2.boundingRect(contour_np)
#                     anno_dic = {"id":obj_id, 
#                                 "image_id":random_seed, 
#                                 "category_id":category_id, 
#                                 "segmentation":contour_list, 
#                                 "bbox":[x, y, w, h]}        # bbox x,y是左下角的点，w,h是宽度和高度
#                     camera_data['annotations'].append(anno_dic)

#         if rgbImg is not None:
#             rgb= rgbImg[:,:,0:3]                    # 舍弃第四个通道
#             rgb[..., [0, 2]] = rgb[..., [2, 0]]     # 更改颜色通道为正常颜色
#             timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())    # 增加时间戳
#             cv2.imwrite(os.path.join(rgb_dir, f'rgb_camera_{timestamp}.png'), rgb)  #保存图片
#             image_dic = {"id": random_seed, "width": rgb.shape[1], "height": rgb.shape[0], "file_name": f'rgb_camera_{timestamp}.png'}
#             camera_data['images'].append(image_dic)

#         if depthImg is not None:
#             # depth_normalized = cv2.normalize(depthImg, None, 0, 255, cv2.NORM_MINMAX)
#             # depth_normalized = depth_normalized.astype('uint8')
#             depthImg = far * near / (far - (far - near) * depthImg)
#             depthImg = np.asanyarray(depthImg).astype(np.float32) * 1000.
#             cv2.imwrite(os.path.join(depth_dir, f'depth_img_{timestamp}.png'), (depthImg).astype(np.uint16))

def create_invisible_wall(position, half_extents):
    collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[255, 0, 0, 1])  # 完全透明
    return p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=collision_id,
                            baseVisualShapeIndex=visual_id,
                            basePosition=position)

def pick(obj_ids, initial_positions, kuka_cid=None, robot_id=None, num_joints=None, i=None, obj_number=None):
    #=================控制机械臂进行吸取操作==================
    dist_all = 0      # 所有物体的总位移
    start_time = time.time()
    remain = 0        # 剩余物体数量
    suck_time = 0     # 吸取尝试次数
    fail = 0          # 失败次数
    circle_center = ENV_CONFIG["circle_center"]
    kuka_end_effector_idx = 11
    
    # 创建结果记录列表
    results = {
        "objnumlist": [],
        "sucktimelist": [],
        "successobjlist": [],
        "failobjlist": [],
        "remainobjlist": [],
        "successratelist": [],
        "totaldistancelist": [],
        "averagedistancelist": [],
        "timelist": []
    }

    for obj in obj_ids:
        #===========存储每个对象的位置和旋转四元数信息(拿走物体可能导致其他物体位置发生变换，所以每次都要更新那个抓取的物体)==============
        # obj['pos'], obj['ang'] = p.getBasePositionAndOrientation(obj_id)
        obj_id = obj['id']
        obj['pos'] = initial_positions[obj_id]
        obj["skip"] = False
        obj['suck_state'] = False
        
        if obj['suck_times'] >= 5:
            print(f"物体{obj_id-6}吸取次数超过5次,放弃吸取")
            remain += 1
            continue

        dist_obj = 0
        suck_time += 1
        current_state = 0

        while current_state < 8:
            if current_state == 0:
                # 机械臂上升，防止碰到物体
                roblist = list(obj['pos'])
                roblist[2] += 0.3
                target_pos, gripper_val = roblist, 0
            # 吸取物体
            elif current_state == 1:
                # 机械臂下降到物体上方0.1，准备吸取物体
                roblist[2] = obj['pos'][2] + 0.05
                target_pos, gripper_val = roblist, 0
            # 物体上升
            elif current_state == 2:
                target_pos, gripper_val = roblist, 1
            elif current_state == 3:
                roblist[2] += 0.5
                target_pos, gripper_val = roblist, 1
            # 移动到下降位置
            elif current_state == 4:
                roblist[0] = 0.7
                roblist[1] = -0.6
                target_pos, gripper_val = roblist, 1
            elif current_state == 5:
                roblist[2] -= 0.3
                target_pos, gripper_val = roblist, 1
            # 放下物体
            elif current_state == 6:
                target_pos, gripper_val = roblist, 0
            # 回到初始位置
            elif current_state == 7:
                roblist[0] = 0.5
                roblist[1] = -0.6
                roblist[2] += 0.4
                target_pos, gripper_val = roblist, 0

            # if current_state < 2:
            #     target_orn = p.getQuaternionFromEuler(new_euler_angles)
            # else:
            #     target_orn = p.getQuaternionFromEuler([0, 1.01*math.pi, 0])
            target_orn = p.getQuaternionFromEuler([0, 1.01*math.pi, 0])
            joint_poses = p.calculateInverseKinematics(robot_id, kuka_end_effector_idx, target_pos, target_orn)

            for j in range(num_joints):
                if j > 6:
                    pass
                else:
                    tp = joint_poses[j]      
                p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=tp)
            
            for _ in range(30):  # 240步（大约4秒时间）
                p.stepSimulation()
                time.sleep(1 / 240) # Time in seconds. 仿真引擎暂停1/240秒，然后再进入下一个时间步，不影响仿真结果。
            
            # 获取末端执行器的位置和姿态
            link_state = p.getLinkState(robot_id, kuka_end_effector_idx)

            # link_state 返回一个包含位置和方向的元组，位置是一个三元组
            end_effector_pos = link_state[0]

            if 0 < current_state <7 and kuka_cid == None and (end_effector_pos[0] > target_pos[0] + 0.1 or end_effector_pos[0] < target_pos[0] - 0.1 or end_effector_pos[1] < target_pos[1] - 0.1 or end_effector_pos[1] > target_pos[1] + 0.1):
                if current_state == 2:
                    x, y, _ = obj['pos']  # 只关心 x, y 坐标
                    dx = x - circle_center[0]
                    dy = y - circle_center[1]
                    distance = math.sqrt(dx**2 + dy**2)
                    dist = math.sqrt((end_effector_pos[0] - x)**2 + (end_effector_pos[1] - y)**2)
                    print(f"物体{obj_id-6}已经超过机械臂最大吸取范围，距离为{distance}吸取失败,距离机械臂为{dist}")
                    obj["skip"] = True
                    remain += 1
                    fail += 1
            elif 0 < current_state < 7 and kuka_cid == None:
                # 使用射线检测判断吸盘和物体A之间是否存在遮挡
                # ray_start = end_effector_pos
                ray_start_1 = [list(obj['pos'])[0]-0.03, list(obj['pos'])[1], list(obj['pos'])[2] + 0.5]
                ray_end_1 = [list(obj['pos'])[0]-0.03, list(obj['pos'])[1], list(obj['pos'])[2]]
                ray_start_2 = [list(obj['pos'])[0], list(obj['pos'])[1]-0.03, list(obj['pos'])[2] + 0.5]
                ray_end_2 = [list(obj['pos'])[0], list(obj['pos'])[1]-0.03, list(obj['pos'])[2]]
                ray_start_3 = [list(obj['pos'])[0]+0.03, list(obj['pos'])[1], list(obj['pos'])[2] + 0.5]
                ray_end_3 = [list(obj['pos'])[0]+0.03, list(obj['pos'])[1], list(obj['pos'])[2]]
                ray_start_4 = [list(obj['pos'])[0], list(obj['pos'])[1]+0.03, list(obj['pos'])[2] + 0.5]
                ray_end_4 = [list(obj['pos'])[0], list(obj['pos'])[1]+0.03, list(obj['pos'])[2]]
                ray_results = [p.rayTest(ray_start, ray_end) for ray_start, ray_end in [(ray_start_1, ray_end_1), (ray_start_2, ray_end_2), (ray_start_3, ray_end_3), (ray_start_4, ray_end_4)]]
                
                
                if (ray_results[0][0][0] != obj_id and ray_results[0][0][0] > 13) or (ray_results[1][0][0] != obj_id and ray_results[1][0][0] > 13) or (ray_results[2][0][0] != obj_id and ray_results[2][0][0] > 13) or (ray_results[3][0][0] != obj_id and ray_results[3][0][0] > 13):
                    if current_state == 2:
                        print(f"物体{obj_id-6}被遮挡，吸取失败，等待后续抓取")
                        obj["skip"] = True
                        obj["suck_times"] += 1
                        fail += 1
                        #将物体往后放，重新等待抓取
                        obj_ids.append(obj)

            elif current_state == 6 and kuka_cid != None:
                print(f"物体{obj_id-6}已成功吸取，类别为{obj['category']}")
                obj['suck_state'] = True
                

            if gripper_val == 0 and kuka_cid != None:
                p.removeConstraint(kuka_cid)
                kuka_cid = None
                for _ in range(60):  # 240步（大约4秒时间）
                    p.stepSimulation()
                    time.sleep(1 / 240) # Time in seconds. 仿真引擎暂停1/240秒，然后再进入下一个时间步，不影响仿真结果。
                
            if gripper_val == 1 and kuka_cid == None and obj['skip'] == False:
                cube_orn = p.getQuaternionFromEuler([0, math.pi, 0])    #返回欧拉角
                # 实现吸取操作的关键，加一个限制
                kuka_cid = p.createConstraint(robot_id, 11, obj['id'], -1, p.JOINT_FIXED, [0, 0, 0], [0 ,0, 0.01], [0, 0, 0], childFrameOrientation=cube_orn)
            
            #=================计算抓取结束后，所有物体的位移==========================
            if current_state == 6 and obj['suck_state'] == True:
                p.removeBody(obj['id'])
                for obj_1 in obj_ids:
                    if obj_1['suck_state'] == False:
                        obj_1['pos'] = p.getBasePositionAndOrientation(obj_1['id'])[0]
                        dist = math.sqrt(sum((initial_positions[obj_1['id']][i]-obj_1['pos'][i])**2 for i in range(3)))
                        initial_positions[obj_1['id']] = obj_1['pos']
                        dist_obj += dist
                print(f"物体{obj_id-6}成功吸取后所有物体的总位移为{dist_obj}")
                dist_all += dist_obj
            #====================================================================
            current_state += 1
    #=====================================================
    end_time = time.time()
    success_rate = round(((suck_time-fail)/suck_time)*100, 3)
    total_distance = round(dist_all, 3)
    average_distance = round(dist_all/(suck_time-fail), 3)
    consume_time = round(end_time-start_time, 3)
    print("============================================")
    print(f"现在统计第{i+1}个场景吸取结果")
    print(f"第{i+1}个环境共有{obj_number}个物体,共吸取次数{suck_time},成功吸取{suck_time-fail}个物体,抓取失败{fail}次,残留{remain}个物体,吸取成功率为{success_rate}%,物体位移总距离为{total_distance}，物体位移平均距离为{average_distance},抓取时间为{consume_time}s")
    results["objnumlist"].append(obj_number)
    results["sucktimelist"].append(suck_time)
    results["successobjlist"].append(suck_time-fail)
    results["failobjlist"].append(fail)
    results["remainobjlist"].append(remain)
    results["successratelist"].append(success_rate)
    results["totaldistancelist"].append(total_distance)
    results["averagedistancelist"].append(average_distance)
    results["timelist"].append(consume_time)
    # 清空当前场景
    p.resetSimulation()

    # 返回结果字典
    return results