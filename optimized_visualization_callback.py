#!/usr/bin/env python3
"""
优化的ManiSkill可视化回调 - 解决卡顿问题
主要优化点：
1. 异步图像处理
2. 减少磁盘I/O
3. 降低渲染频率
4. 使用内存缓冲
"""

import os
import time
import threading
import queue
from collections import deque
from typing import Optional, List, Dict, Any

import numpy as np
import cv2
import torch
from stable_baselines3.common.callbacks import BaseCallback


class AsyncImageBuffer:
    """异步图像缓冲器 - 避免主线程阻塞"""
    
    def __init__(self, max_size=5):
        self.buffer = queue.Queue(maxsize=max_size)
        self.display_thread = None
        self.running = False
        
    def start(self):
        """启动异步处理线程"""
        self.running = True
        self.display_thread = threading.Thread(target=self._display_worker, daemon=True)
        self.display_thread.start()
        print("异步图像处理线程已启动")
    
    def stop(self):
        """停止异步处理"""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("异步图像处理线程已停止")
    
    def add_image(self, image: np.ndarray, step: int):
        """添加图像到缓冲区（非阻塞）"""
        try:
            # 非阻塞添加，如果缓冲区满了就跳过
            self.buffer.put_nowait((image.copy(), step))
        except queue.Full:
            # 缓冲区满了，跳过这一帧
            pass
    
    def _display_worker(self):
        """异步显示工作线程"""
        window_name = "ManiSkill Training (Optimized)"
        
        while self.running:
            try:
                # 等待图像数据，超时1秒
                image, step = self.buffer.get(timeout=1.0)
                
                # 显示图像
                cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                # 设置窗口标题显示步数
                cv2.setWindowTitle(window_name, f"ManiSkill Training - Step {step}")
                
                # 非阻塞等待按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户按下 'q' 键，停止可视化")
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"显示线程错误: {e}")
                continue


class OptimizedVisualizationCallback(BaseCallback):
    """优化的可视化回调 - 解决卡顿问题"""
    
    def __init__(self, 
                 render_freq=200,      # 降低渲染频率
                 max_envs_display=4,   # 限制显示环境数
                 enable_async=True,    # 启用异步处理
                 enable_disk_save=False,  # 默认不保存到磁盘
                 verbose=1):
        super().__init__(verbose)
        
        # 配置参数
        self.render_freq = render_freq
        self.max_envs_display = max_envs_display
        self.enable_async = enable_async
        self.enable_disk_save = enable_disk_save
        
        # 状态变量
        self.step_count = 0
        self.last_render_time = 0
        self.frame_times = deque(maxlen=100)  # 用于计算FPS
        
        # 异步处理组件
        self.image_buffer = None
        if enable_async:
            self.image_buffer = AsyncImageBuffer(max_size=3)
        
        # 磁盘保存设置（如果启用）
        if enable_disk_save:
            self.save_dir = "visualization_images"
            os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"优化可视化回调已创建:")
        print(f"  渲染频率: 每{render_freq}步")
        print(f"  最大显示环境: {max_envs_display}")
        print(f"  异步处理: {'启用' if enable_async else '禁用'}")
        print(f"  磁盘保存: {'启用' if enable_disk_save else '禁用'}")
    
    def _on_training_start(self):
        """训练开始时启动异步处理"""
        if self.image_buffer:
            self.image_buffer.start()
        
        if self.verbose > 0:
            print("优化可视化系统已启动")
    
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # 检查是否需要渲染
        if self.step_count % self.render_freq != 0:
            return True
        
        # 记录渲染开始时间
        start_time = time.time()
        
        try:
            self._handle_rendering()
            
            # 计算渲染耗时
            render_time = time.time() - start_time
            self.frame_times.append(render_time)
            
            # 定期输出性能统计
            if self.step_count % (self.render_freq * 10) == 0:
                avg_time = np.mean(self.frame_times) if self.frame_times else 0
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"渲染性能 - 平均耗时: {avg_time*1000:.1f}ms, FPS: {fps:.1f}")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"渲染处理错误: {e}")
        
        return True
    
    def _handle_rendering(self):
        """处理渲染逻辑（优化版本）"""
        # 获取环境渲染图像
        images = self._get_env_images()
        if not images:
            return
        
        # 限制显示的环境数量
        images = images[:self.max_envs_display]
        
        # 快速组合图像
        combined_img = self._fast_combine_images(images)
        if combined_img is None:
            return
        
        # 异步显示或同步显示
        if self.image_buffer:
            # 异步显示（推荐）
            self.image_buffer.add_image(combined_img, self.step_count)
        else:
            # 同步显示
            self._sync_display_image(combined_img)
        
        # 可选：保存到磁盘
        if self.enable_disk_save:
            self._save_image_to_disk(combined_img)
    
    def _get_env_images(self) -> List[np.ndarray]:
        """获取环境图像（优化版本）"""
        images = []
        
        try:
            # 优先使用render_human方法（官方推荐的并行化渲染）
            if hasattr(self.training_env, 'render_human'):
                try:
                    rendered = self.training_env.render_human()
                    if rendered is not None:
                        images = self._process_rendered_output(rendered)
                        if images:
                            return images
                except Exception as e:
                    if self.verbose > 0:
                        print(f"render_human失败，回退到render: {e}")
            
            # 回退到标准render方法
            if hasattr(self.training_env, 'render'):
                rendered = self.training_env.render()
                if rendered is not None:
                    images = self._process_rendered_output(rendered)
                        
        except Exception as e:
            if self.verbose > 0:
                print(f"获取环境图像失败: {e}")
        
        return images
    
    def _process_rendered_output(self, rendered) -> List[np.ndarray]:
        """处理渲染输出"""
        images = []
        
        if isinstance(rendered, np.ndarray):
            if len(rendered.shape) == 4:  # 批量图像
                images = [rendered[i] for i in range(rendered.shape[0])]
            else:  # 单个图像
                images = [rendered]
        elif isinstance(rendered, list):
            images = rendered
        
        return images
    
    def _fast_combine_images(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """快速组合图像（优化版本）"""
        if not images:
            return None
        
        num_images = len(images)
        
        # 单个图像直接返回
        if num_images == 1:
            return self._ensure_rgb_format(images[0])
        
        # 多个图像的快速布局
        try:
            # 确保所有图像格式一致
            processed_images = []
            target_size = None
            
            for img in images:
                # 确保RGB格式
                rgb_img = self._ensure_rgb_format(img)
                if rgb_img is None:
                    continue
                
                # 统一尺寸
                if target_size is None:
                    target_size = rgb_img.shape[:2]
                elif rgb_img.shape[:2] != target_size:
                    rgb_img = cv2.resize(rgb_img, (target_size[1], target_size[0]))
                
                processed_images.append(rgb_img)
            
            if not processed_images:
                return None
            
            # 快速网格布局
            return self._create_grid_layout(processed_images)
            
        except Exception as e:
            if self.verbose > 0:
                print(f"图像组合失败: {e}")
            return None
    
    def _ensure_rgb_format(self, img: np.ndarray) -> Optional[np.ndarray]:
        """确保图像是RGB格式"""
        if img is None or len(img.shape) < 2:
            return None
        
        if len(img.shape) == 2:  # 灰度图
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB图
            return img
        else:
            return None
    
    def _create_grid_layout(self, images: List[np.ndarray]) -> np.ndarray:
        """创建网格布局（快速版本）"""
        num_images = len(images)
        
        if num_images <= 2:
            # 水平排列
            return np.hstack(images)
        else:
            # 2x2网格
            if num_images == 3:
                # 补充一个空白图像
                empty_img = np.zeros_like(images[0])
                images.append(empty_img)
            
            # 创建2x2网格
            top_row = np.hstack(images[:2])
            bottom_row = np.hstack(images[2:4])
            combined = np.vstack([top_row, bottom_row])
            
            # 添加环境标签
            self._add_env_labels(combined, min(num_images, 4))
            
            return combined
    
    def _add_env_labels(self, image: np.ndarray, num_envs: int):
        """添加环境标签"""
        h, w = image.shape[:2]
        
        # 计算每个环境区域的位置
        positions = [
            (10, 30),                    # 左上
            (w//2 + 10, 30),            # 右上
            (10, h//2 + 30),            # 左下
            (w//2 + 10, h//2 + 30)      # 右下
        ]
        
        for i in range(min(num_envs, 4)):
            cv2.putText(image, f'Env {i}', positions[i], 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _sync_display_image(self, image: np.ndarray):
        """同步显示图像"""
        try:
            cv2.imshow("ManiSkill Training", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)  # 非阻塞
        except Exception as e:
            if self.verbose > 0:
                print(f"同步显示失败: {e}")
    
    def _save_image_to_disk(self, image: np.ndarray):
        """保存图像到磁盘（可选功能）"""
        try:
            filename = f"{self.save_dir}/step_{self.step_count:06d}.png"
            cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # 清理旧文件（保持最新5张）
            self._cleanup_old_images(keep_count=5)
            
        except Exception as e:
            if self.verbose > 0:
                print(f"保存图像失败: {e}")
    
    def _cleanup_old_images(self, keep_count=5):
        """清理旧的图像文件"""
        try:
            import glob
            pattern = f"{self.save_dir}/step_*.png"
            files = sorted(glob.glob(pattern))
            
            if len(files) > keep_count:
                for old_file in files[:-keep_count]:
                    os.remove(old_file)
                    
        except Exception as e:
            if self.verbose > 0:
                print(f"清理文件失败: {e}")
    
    def _on_training_end(self):
        """训练结束时的清理"""
        if self.image_buffer:
            self.image_buffer.stop()
        
        cv2.destroyAllWindows()
        
        if self.verbose > 0:
            print("优化可视化系统已关闭")


def create_optimized_callback(
    render_freq=200,
    max_envs=4,
    enable_async=True,
    save_to_disk=False
) -> OptimizedVisualizationCallback:
    """创建优化的可视化回调（便捷函数）"""
    return OptimizedVisualizationCallback(
        render_freq=render_freq,
        max_envs_display=max_envs,
        enable_async=enable_async,
        enable_disk_save=save_to_disk,
        verbose=1
    )


# 使用示例
if __name__ == "__main__":
    print("优化可视化回调模块")
    print("主要特性:")
    print("1. 异步图像处理 - 避免训练阻塞")
    print("2. 降低渲染频率 - 减少性能开销")
    print("3. 限制显示环境数 - 优化资源使用")
    print("4. 可选磁盘保存 - 减少I/O操作")
    print("5. 性能监控 - 实时FPS统计") 