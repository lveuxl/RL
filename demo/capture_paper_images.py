#!/usr/bin/env python3
"""
论文展示场景高清图像捕获脚本
保存两个视角的高清图像：俯视图和侧面45度平视图
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import cv2
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/linux/jzh/RL_Robot/demo')

# 动态导入环境
import importlib.util
spec = importlib.util.spec_from_file_location(
    "env_clutter_copy", 
    "/home/linux/jzh/RL_Robot/demo/env_clutter copy.py"
)
env_clutter_copy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_clutter_copy)
PaperStackingSceneEnv = env_clutter_copy.PaperStackingSceneEnv

from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig


class PaperImageCaptureEnv(PaperStackingSceneEnv):
    """
    扩展论文场景环境，支持多相机视角的高清图像捕获
    """
    
    @property
    def _default_sensor_configs(self):
        """配置多个相机视角"""
        configs = []
        
        # 1. 俯视相机（从上往下看托盘）
        top_view_pose = sapien_utils.look_at(
            eye=[0.0, 0.0, 0.8],      # 相机位置：托盘正上方80cm
            target=[-0.2, 0.0, 0.1]   # 看向托盘中心稍上方
        )
        configs.append(CameraConfig(
            "top_view_camera",
            pose=top_view_pose,
            width=4096,   # 超高清分辨率 4K
            height=2160,  # 4K UHD
            fov=np.pi/4,  # 45度视野角
            near=0.01,
            far=100,
        ))
        
        # 2. 侧面45度相机（斜向平视物体）
        side_view_pose = sapien_utils.look_at(
            eye=[0.4, 0.4, 0.3],      # 相机位置：托盘斜前方
            target=[-0.2, 0.0, 0.1]   # 看向托盘中心
        )
        configs.append(CameraConfig(
            "side_view_camera", 
            pose=side_view_pose,
            width=4096,   # 超高清分辨率 4K
            height=2160,  # 4K UHD
            fov=np.pi/3,  # 60度视野角，看得更广
            near=0.01,
            far=100,
        ))
        
        # 3. 保留默认相机作为备用
        default_pose = sapien_utils.look_at(
            eye=[0.5, 0.5, 0.6], 
            target=[-0.15, 0.0, 0.15]
        )
        configs.append(CameraConfig(
            "base_camera",
            pose=default_pose,
            width=1280,
            height=960,
            fov=np.pi/3,
            near=0.01,
            far=100,
        ))
        
        return configs


def capture_high_quality_images(scene_config="balanced", output_dir="paper_images"):
    """
    捕获论文展示场景的高清图像
    
    Args:
        scene_config: 场景配置名称
        output_dir: 输出目录
    """
    print("=" * 50)
    print("🎨 论文展示场景高清图像捕获")
    print("=" * 50)
    print(f"📷 场景配置: {scene_config}")
    print(f"📁 输出目录: {output_dir}")
    
    # 创建输出目录
    timestamp = int(time.time())
    full_output_dir = f"{output_dir}_{scene_config}_{timestamp}"
    Path(full_output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📂 完整输出路径: {full_output_dir}")
    
    try:
        # 1. 创建环境
        print("\n🏗️  创建高清图像捕获环境...")
        env = PaperImageCaptureEnv(
            render_mode="rgb_array",
            obs_mode="state", 
            control_mode="pd_ee_delta_pose",
            num_envs=1,
            scene_config=scene_config,
        )
        print("✅ 环境创建成功")
        
        # 2. 重置环境，构建场景
        print("\n🎬 初始化场景...")
        obs, info = env.reset()
        print("✅ 场景重置完成，堆叠结构已创建")
        
        # 3. 等待场景稳定
        print("\n⏳ 等待场景完全稳定...")
        for i in range(30):  # 额外稳定步骤
            env.step(np.zeros(7))  # 无动作，让场景继续稳定
            if i % 10 == 0:
                print(f"  稳定中... {i+1}/30")
        print("✅ 场景稳定完成")
        
        # 4. 捕获不同视角的图像
        print("\n📸 开始捕获高清图像...")
        
        # 获取所有相机的观测
        env.scene.update_render(update_sensors=True)
        env.capture_sensor_data()
        
        camera_names = ["top_view_camera", "side_view_camera", "base_camera"]
        view_descriptions = ["俯视图", "侧面45度视图", "默认视图"]
        
        saved_images = []
        
        for i, (camera_name, description) in enumerate(zip(camera_names, view_descriptions)):
            if camera_name in env._sensors:
                print(f"  📷 捕获{description} ({camera_name})...")
                
                # 获取RGB图像
                camera = env._sensors[camera_name]
                obs_data = camera.get_obs(rgb=True, depth=False, segmentation=False)
                rgb_image = obs_data["rgb"]  # [1, H, W, 3] tensor
                
                # 转换为numpy数组
                if isinstance(rgb_image, torch.Tensor):
                    rgb_array = rgb_image[0].cpu().numpy()  # 取第一个环境 [H, W, 3]
                else:
                    rgb_array = rgb_image[0]
                
                # 确保数据格式正确
                if rgb_array.dtype == np.float32 or rgb_array.dtype == np.float64:
                    if rgb_array.max() <= 1.0:
                        rgb_array = (rgb_array * 255).astype(np.uint8)
                    else:
                        rgb_array = rgb_array.astype(np.uint8)
                
                # 转换BGR格式用于OpenCV保存
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                
                # 保存图像
                image_filename = f"{full_output_dir}/{camera_name}_{scene_config}.png"
                success = cv2.imwrite(image_filename, bgr_array)
                
                if success:
                    print(f"    ✅ {description}已保存: {image_filename}")
                    print(f"       分辨率: {rgb_array.shape[1]}x{rgb_array.shape[0]}")
                    saved_images.append({
                        'filename': image_filename,
                        'description': description,
                        'camera': camera_name,
                        'resolution': f"{rgb_array.shape[1]}x{rgb_array.shape[0]}"
                    })
                else:
                    print(f"    ❌ {description}保存失败")
            else:
                print(f"  ⚠️  未找到相机: {camera_name}")
        
        # 5. 生成场景信息报告
        print("\n📋 生成场景信息报告...")
        report_filename = f"{full_output_dir}/scene_report.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("论文展示场景图像捕获报告\n")
            f.write("=" * 40 + "\n")
            f.write(f"捕获时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"场景配置: {scene_config}\n")
            f.write(f"物体总数: {len(env.all_objects)}\n")
            if hasattr(env, 'target_object') and env.target_object:
                f.write(f"目标物体: {env.target_object.name}\n")
            f.write(f"\n图像信息:\n")
            f.write("-" * 20 + "\n")
            
            for img_info in saved_images:
                f.write(f"文件名: {os.path.basename(img_info['filename'])}\n")
                f.write(f"描述: {img_info['description']}\n")
                f.write(f"相机: {img_info['camera']}\n")
                f.write(f"分辨率: {img_info['resolution']}\n")
                f.write(f"路径: {img_info['filename']}\n")
                f.write("\n")
            
            # 添加物体列表
            f.write("场景物体列表:\n")
            f.write("-" * 20 + "\n")
            for i, obj in enumerate(env.all_objects):
                f.write(f"{i+1:2d}. {obj.name}\n")
        
        print(f"✅ 场景报告已保存: {report_filename}")
        
        # 6. 清理
        env.close()
        
        # 7. 总结
        print("\n" + "=" * 50)
        print("🎉 高清图像捕获完成！")
        print("=" * 50)
        print(f"📂 输出目录: {full_output_dir}")
        print(f"📷 成功捕获: {len(saved_images)} 张图像")
        
        for img_info in saved_images:
            print(f"  • {img_info['description']}: {os.path.basename(img_info['filename'])}")
        
        print(f"📋 场景报告: {os.path.basename(report_filename)}")
        print("\n💡 建议:")
        print("  1. 检查图像质量和视角是否满足论文需求")
        print("  2. 可调整相机位置重新捕获")
        print("  3. 图像已为高清分辨率，适合论文使用")
        
        return full_output_dir, saved_images
        
    except Exception as e:
        print(f"\n❌ 图像捕获过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None, []


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="论文展示场景高清图像捕获")
    parser.add_argument("--config", type=str, default="balanced", 
                       help="场景配置名称 (默认: balanced)")
    parser.add_argument("--output", type=str, default="paper_images",
                       help="输出目录前缀 (默认: paper_images)")
    
    args = parser.parse_args()
    
    # 执行图像捕获
    output_dir, saved_images = capture_high_quality_images(
        scene_config=args.config,
        output_dir=args.output
    )
    
    if output_dir and saved_images:
        print(f"\n🔚 任务完成，共保存 {len(saved_images)} 张高清图像")
    else:
        print("\n🔚 任务失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
