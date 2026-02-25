"""
EnvClutter Motion Planning 包

提供针对复杂堆叠抓取环境的智能运动规划解决方案

主要组件:
- EnvClutterMotionPlanner: 核心运动规划器
- solve_env_clutter: 高级求解接口
- run_env_clutter: 演示和评估脚本
"""

from .env_clutter_solver import EnvClutterMotionPlanner, solve_env_clutter

__version__ = "1.0.0"
__author__ = "RL_Robot Team"
