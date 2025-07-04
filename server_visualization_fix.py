#!/usr/bin/env python3
"""
服务器可视化问题解决方案
解决ManiSkill在服务器上直接运行时的黑屏问题
"""

import os
import sys
import subprocess
import time
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import mani_skill.envs

def check_display_environment():
    """检查显示环境配置"""
    print("=== 显示环境检查 ===")
    
    # 检查DISPLAY环境变量
    display = os.environ.get('DISPLAY')
    print(f"DISPLAY环境变量: {display}")
    
    # 检查是否有X11转发
    if display:
        print("✅ DISPLAY已设置")
    else:
        print("❌ DISPLAY未设置")
        print("建议设置: export DISPLAY=:0")
    
    # 检查xvfb是否可用
    try:
        result = subprocess.run(['which', 'xvfb-run'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ xvfb-run可用")
        else:
            print("❌ xvfb-run不可用")
            print("安装命令: sudo apt-get install xvfb")
    except:
        print("❌ 无法检查xvfb-run")
    
    # 检查当前用户权限
    print(f"当前用户: {os.environ.get('USER', 'unknown')}")
    
    # 检查GPU相关
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA不可用")

def setup_virtual_display():
    """设置虚拟显示器"""
    print("\n=== 设置虚拟显示器 ===")
    
    # 方法1: 设置DISPLAY环境变量
    if not os.environ.get('DISPLAY'):
        os.environ['DISPLAY'] = ':99'
        print("设置DISPLAY=:99")
    
    # 方法2: 启动虚拟显示器
    try:
        # 检查是否已经有Xvfb在运行
        result = subprocess.run(['pgrep', 'Xvfb'], capture_output=True)
        if result.returncode != 0:
            print("启动Xvfb虚拟显示器...")
            subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)  # 等待启动
            print("✅ Xvfb已启动")
        else:
            print("✅ Xvfb已在运行")
    except Exception as e:
        print(f"❌ 启动Xvfb失败: {e}")
        print("尝试手动启动: Xvfb :99 -screen 0 1024x768x24 &")

def demo_with_virtual_display():
    """使用虚拟显示器进行演示"""
    print("\n=== 虚拟显示器演示 ===")
    
    # 设置虚拟显示器
    setup_virtual_display()
    
    try:
        # 创建环境 - 尝试human模式
        print("尝试创建human模式环境...")
        env = gym.make(
            "PickCube-v1",
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            render_mode="human",
            robot_uids="panda",
        )
        
        print("✅ human模式环境创建成功")
        
        # 重置环境
        obs, info = env.reset()
        print(f"环境重置成功，观察空间: {obs.shape}")
        
        # 执行几步动作
        print("执行演示动作...")
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 尝试渲染
            try:
                env.render()
                print(f"步数 {step + 1}: 渲染成功")
            except Exception as e:
                print(f"步数 {step + 1}: 渲染失败 - {e}")
            
            if terminated or truncated:
                print("Episode结束")
                break
            
            time.sleep(0.1)
        
        env.close()
        print("✅ 虚拟显示器演示完成")
        
    except Exception as e:
        print(f"❌ 虚拟显示器演示失败: {e}")
        print("回退到rgb_array模式...")
        demo_rgb_array_mode()

def demo_rgb_array_mode():
    """RGB数组模式演示（保存图像）"""
    print("\n=== RGB数组模式演示 ===")
    
    try:
        env = gym.make(
            "PickCube-v1",
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            render_mode="rgb_array",
            robot_uids="panda",
        )
        
        print("✅ rgb_array模式环境创建成功")
        
        # 创建输出目录
        output_dir = "visualization_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # 重置环境
        obs, info = env.reset()
        
        # 保存初始图像
        img = env.render()
        if img is not None:
            # 转换为numpy数组并保存
            if isinstance(img, torch.Tensor):
                img_np = img.squeeze().cpu().numpy()
            else:
                img_np = img
            
            # 保存为图像文件
            try:
                from PIL import Image
                if len(img_np.shape) == 3:
                    # 确保数据类型正确
                    if img_np.dtype != np.uint8:
                        img_np = (img_np * 255).astype(np.uint8)
                    
                    img_pil = Image.fromarray(img_np)
                    img_pil.save(f"{output_dir}/initial_state.png")
                    print(f"✅ 初始状态图像已保存: {output_dir}/initial_state.png")
            except ImportError:
                print("❌ PIL不可用，无法保存图像")
                print("安装命令: pip install Pillow")
        
        # 执行动作并保存关键帧
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 渲染并保存
            img = env.render()
            if img is not None:
                try:
                    from PIL import Image
                    if isinstance(img, torch.Tensor):
                        img_np = img.squeeze().cpu().numpy()
                    else:
                        img_np = img
                    
                    if img_np.dtype != np.uint8:
                        img_np = (img_np * 255).astype(np.uint8)
                    
                    img_pil = Image.fromarray(img_np)
                    img_pil.save(f"{output_dir}/step_{step + 1}.png")
                    print(f"✅ 步数 {step + 1} 图像已保存")
                except:
                    print(f"❌ 步数 {step + 1} 图像保存失败")
            
            if terminated or truncated:
                break
        
        env.close()
        print(f"✅ 图像已保存到 {output_dir}/ 目录")
        
    except Exception as e:
        print(f"❌ RGB数组模式演示失败: {e}")

def demo_headless_training():
    """无头模式训练演示"""
    print("\n=== 无头模式训练演示 ===")
    
    # 设置环境变量强制使用CPU渲染
    os.environ['MUJOCO_GL'] = 'osmesa'  # 或者 'egl'
    
    try:
        # 创建环境
        def make_env():
            return gym.make(
                "PickCube-v1",
                obs_mode="state",
                control_mode="pd_joint_delta_pos",
                render_mode="rgb_array",
                robot_uids="panda",
            )
        
        env = DummyVecEnv([make_env])
        
        # 创建简单的PPO模型
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=64,
            batch_size=32,
            learning_rate=3e-4,
            policy_kwargs={"net_arch": [64, 64]},
        )
        
        print("开始无头模式训练...")
        model.learn(total_timesteps=500, progress_bar=True)
        
        print("✅ 无头模式训练完成")
        
        # 保存模型
        model.save("headless_demo_model")
        print("✅ 模型已保存")
        
        env.close()
        
    except Exception as e:
        print(f"❌ 无头模式训练失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='服务器可视化问题解决方案')
    parser.add_argument('--mode', type=str, 
                       choices=['check', 'virtual', 'rgb', 'headless', 'all'], 
                       default='all',
                       help='选择运行模式')
    
    args = parser.parse_args()
    
    print("ManiSkill服务器可视化问题解决方案")
    print("=" * 50)
    
    if args.mode == 'check' or args.mode == 'all':
        check_display_environment()
    
    if args.mode == 'virtual' or args.mode == 'all':
        demo_with_virtual_display()
    
    if args.mode == 'rgb' or args.mode == 'all':
        demo_rgb_array_mode()
    
    if args.mode == 'headless' or args.mode == 'all':
        demo_headless_training()
    
    print("\n" + "=" * 50)
    print("解决方案总结:")
    print("1. 检查显示环境配置")
    print("2. 使用虚拟显示器 (Xvfb)")
    print("3. 使用RGB数组模式保存图像")
    print("4. 使用无头模式训练")
    print("5. 设置MUJOCO_GL环境变量")

if __name__ == "__main__":
    main() 