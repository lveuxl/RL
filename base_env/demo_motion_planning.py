#!/usr/bin/env python3
"""
🎯 EnvClutter Motion Planning 一键演示脚本

直接运行此脚本即可体验智能抓取系统：
python demo_motion_planning.py

功能：自动分析堆叠场景，规划最优抓取序列，执行智能机器人操作
"""

import os
import sys
import time
from pathlib import Path

# 设置项目路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

print("🤖 EnvClutter Motion Planning 智能抓取系统")
print("=" * 50)
print("正在初始化系统...")

try:
    import gymnasium as gym
    import numpy as np
    
    # 尝试导入必要模块
    print("📦 导入Motion Planning模块...")
    from motionplanning.env_clutter_solver import solve_env_clutter
    
    # 导入环境（确保已注册）
    from env_clutter import EnvClutterOptimizedEnv
    
    # 设置环境变量
    os.environ["MANI_SKILL_ASSET_DIR"] = str(project_root / "data")
    
    print("✅ 系统初始化完成！")
    print()
    
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请确保已安装相关依赖：mani_skill, mplib, sapien等")
    sys.exit(1)

def quick_demo():
    """快速演示功能"""
    print("🚀 启动智能抓取演示...")
    
    try:
        # 创建环境
        print("🏗️  创建仿真环境...")
        env = gym.make(
            "EnvClutterOptimized-v1",
            robot_uids="panda",
            control_mode="pd_joint_pos",
            render_mode="human",  # 开启可视化
            sim_backend="auto",
        )
        
        print("🎯 开始智能抓取任务...")
        print("   - 自动场景分析")
        print("   - 物体层次识别")
        print("   - 最优抓取规划")
        print("   - 防碰撞路径执行")
        print()
        
        # 运行Motion Planning
        start_time = time.time()
        
        result = solve_env_clutter(
            env,
            seed=42,  # 固定种子保证可复现
            debug=False,  # 非调试模式，自动执行
            vis=True,  # 开启可视化
            max_objects=2  # 抓取2个物体作为演示
        )
        
        execution_time = time.time() - start_time
        
        # 显示结果
        print("📊 任务执行结果:")
        print("=" * 30)
        
        if result.get("success", False):
            print("✅ 任务执行成功！")
            print(f"   成功率: {result.get('success_rate', 0):.1%}")
            print(f"   抓取物体: {result.get('grasped_objects', 0)} 个")
            print(f"   执行步数: {result.get('total_steps', 0)} 步")
            print(f"   总用时: {execution_time:.1f} 秒")
        else:
            print("❌ 任务执行失败")
            if 'error' in result:
                print(f"   错误信息: {result['error']}")
        
        env.close()
        print("\n🎉 演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        print("这可能是由于环境配置或依赖问题导致的")
        return False
    
    return True

def show_advanced_usage():
    """显示高级用法"""
    print("\n🔧 高级用法:")
    print("-" * 20)
    print("1. 完整演示（多回合）:")
    print("   python motionplanning/run_env_clutter.py --vis --episodes 5")
    print()
    print("2. 调试模式（手动确认每步）:")
    print("   python motionplanning/run_env_clutter.py --vis --debug")
    print()
    print("3. 性能测试（无可视化）:")
    print("   python motionplanning/run_env_clutter.py --episodes 10 --max-objects 5")
    print()
    print("4. 自定义配置:")
    print("   python motionplanning/run_env_clutter.py --robot fetch --control-mode pd_ee_delta_pose")

def main():
    """主函数"""
    try:
        # 运行快速演示
        success = quick_demo()
        
        if success:
            # 显示高级用法
            show_advanced_usage()
        
        # 提示用户
        print("\n💡 小贴士:")
        print("   - 如遇到窗口显示问题，请确保X11转发正常")
        print("   - 调整 --joint-speed 参数可改变机器人运动速度")
        print("   - 使用 --save-stats 保存详细执行数据")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断演示")
    
    print("\n👋 感谢使用 EnvClutter Motion Planning 系统！")

if __name__ == "__main__":
    main()
