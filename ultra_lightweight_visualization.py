#!/usr/bin/env python3
"""
超轻量级ManiSkill可视化回调 - 彻底解决卡顿问题
针对单环境可视化的极致优化方案
"""

import os
import time
import threading
import queue
from typing import Optional, Any

import numpy as np
import cv2
import torch
from stable_baselines3.common.callbacks import BaseCallback


class UltraLightweightVisualization(BaseCallback):
    """超轻量级可视化回调 - 专门解决ManiSkill卡顿问题"""
    
    def __init__(self, 
                 render_freq=500,          # 大幅降低渲染频率
                 enable_display=True,      # 是否显示窗口
                 max_fps=5,               # 限制最大FPS
                 skip_frames=True,        # 跳帧处理
                 verbose=1):
        super().__init__(verbose)
        
        # 基础配置
        self.render_freq = render_freq
        self.enable_display = enable_display
        self.max_fps = max_fps
        self.skip_frames = skip_frames
        
        # 状态变量
        self.step_count = 0
        self.last_render_time = 0
        self.min_render_interval = 1.0 / max_fps  # 最小渲染间隔
        
        # 性能统计
        self.render_times = []
        self.total_renders = 0
        
        # 窗口设置
        self.window_name = "ManiSkill (Ultra Lightweight)"
        self.window_created = False
        
        print(f"超轻量级可视化已创建:")
        print(f"  渲染频率: 每{render_freq}步")
        print(f"  最大FPS: {max_fps}")
        print(f"  跳帧处理: {'启用' if skip_frames else '禁用'}")
    
    def _on_training_start(self):
        """训练开始时的初始化"""
        if self.verbose > 0:
            print("超轻量级可视化系统启动")
    
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # 检查渲染条件
        if not self._should_render():
            return True
        
        # 执行轻量级渲染
        self._lightweight_render()
        
        return True
    
    def _should_render(self) -> bool:
        """判断是否应该渲染"""
        # 频率检查
        if self.step_count % self.render_freq != 0:
            return False
        
        # FPS限制检查
        current_time = time.time()
        if current_time - self.last_render_time < self.min_render_interval:
            return False
        
        return True
    
    def _lightweight_render(self):
        """超轻量级渲染实现"""
        start_time = time.time()
        
        try:
            # 获取单个环境的渲染图像
            image = self._get_single_env_image()
            if image is None:
                return
            
            # 快速处理和显示
            if self.enable_display:
                self._fast_display(image)
            
            # 更新统计
            self.last_render_time = time.time()
            render_time = self.last_render_time - start_time
            self.render_times.append(render_time)
            self.total_renders += 1
            
            # 定期输出性能统计
            if self.total_renders % 10 == 0:
                avg_time = np.mean(self.render_times[-10:])
                print(f"渲染性能: {avg_time*1000:.1f}ms/帧, 总计{self.total_renders}帧")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"轻量级渲染错误: {e}")
    
    def _get_single_env_image(self) -> Optional[np.ndarray]:
        """获取单个环境的图像（优化版本）"""
        try:
            # 优先使用render_human方法（官方推荐的并行化渲染）
            if hasattr(self.training_env, 'render_human'):
                try:
                    rendered = self.training_env.render_human()
                    if rendered is not None:
                        return self._process_rendered_output(rendered)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"render_human失败，回退到render: {e}")
            
            # 回退到标准render方法
            if hasattr(self.training_env, 'render'):
                rendered = self.training_env.render()
                if rendered is not None:
                    return self._process_rendered_output(rendered)
            
            return None
            
        except Exception as e:
            if self.verbose > 0:
                print(f"获取图像失败: {e}")
            return None
    
    def _process_rendered_output(self, rendered) -> Optional[np.ndarray]:
        """处理渲染输出"""
        # 处理不同的渲染输出格式
        if isinstance(rendered, np.ndarray):
            if len(rendered.shape) == 4:  # 批量图像
                return rendered[0]  # 只取第一个环境
            else:  # 单个图像
                return rendered
        elif isinstance(rendered, list) and len(rendered) > 0:
            return rendered[0]  # 只取第一个环境
        
        return None
    
    def _fast_display(self, image: np.ndarray):
        """快速显示图像（无额外处理）"""
        try:
            # 确保图像格式正确
            if len(image.shape) != 3 or image.shape[2] != 3:
                return
            
            # 快速尺寸调整（如果图像太大）
            if image.shape[0] > 480 or image.shape[1] > 640:
                image = cv2.resize(image, (640, 480))
            
            # 创建窗口（仅一次）
            if not self.window_created:
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                self.window_created = True
            
            # 直接显示，RGB转BGR
            cv2.imshow(self.window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # 非阻塞等待按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户按下 'q' 键，停止可视化")
                return False
                
        except Exception as e:
            if self.verbose > 0:
                print(f"显示图像失败: {e}")
    
    def _on_training_end(self):
        """训练结束时的清理"""
        cv2.destroyAllWindows()
        
        if self.render_times:
            avg_time = np.mean(self.render_times)
            total_time = sum(self.render_times)
            print(f"\n可视化性能统计:")
            print(f"  总渲染次数: {self.total_renders}")
            print(f"  平均渲染时间: {avg_time*1000:.1f}ms")
            print(f"  总渲染时间: {total_time:.2f}s")
            print(f"  渲染开销占比: {total_time/(time.time()-self.last_render_time)*100:.1f}%")
        
        if self.verbose > 0:
            print("超轻量级可视化系统已关闭")


class MinimalVisualizationCallback(BaseCallback):
    """最小化可视化回调 - 仅用于验证环境运行"""
    
    def __init__(self, render_freq=1000, verbose=1):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        if self.step_count % self.render_freq == 0:
            print(f"步数: {self.step_count} - 环境运行正常")
            
            # 仅获取渲染数据，不显示
            try:
                if hasattr(self.training_env, 'render'):
                    rendered = self.training_env.render()
                    if rendered is not None:
                        if isinstance(rendered, np.ndarray):
                            shape = rendered.shape
                        elif isinstance(rendered, list):
                            shape = f"list[{len(rendered)}]"
                        else:
                            shape = type(rendered)
                        print(f"  渲染数据形状: {shape}")
            except Exception as e:
                print(f"  渲染测试失败: {e}")
        
        return True


def create_ultra_lightweight_callback(
    render_freq=500,
    max_fps=5,
    enable_display=True
) -> UltraLightweightVisualization:
    """创建超轻量级可视化回调"""
    return UltraLightweightVisualization(
        render_freq=render_freq,
        enable_display=enable_display,
        max_fps=max_fps,
        skip_frames=True,
        verbose=1
    )


def create_minimal_callback(render_freq=1000) -> MinimalVisualizationCallback:
    """创建最小化可视化回调（仅用于测试）"""
    return MinimalVisualizationCallback(
        render_freq=render_freq,
        verbose=1
    )


# 使用示例和测试
if __name__ == "__main__":
    print("超轻量级ManiSkill可视化回调")
    print("特性:")
    print("1. 极低渲染频率 - 每500步渲染一次")
    print("2. FPS限制 - 最大5FPS")
    print("3. 跳帧处理 - 避免渲染堆积")
    print("4. 零异步处理 - 消除线程开销")
    print("5. 最小化图像处理 - 直接显示")
    print("6. 性能监控 - 实时统计") 