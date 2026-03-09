"""
visualize_for_paper.py — 生成适合论文的高质量可视化

特点：
1. 3D 塔视图，显示所有积木
2. 决策顺序标注（1, 2, 3...）
3. 颜色编码（成功/失败）
4. 高分辨率，适合论文
5. 多种视角（正视图、侧视图、俯视图）

Usage:
    python visualize_for_paper.py --decisions results/rl.json \
        --episode 0 --output paper_figures/
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 设置论文质量的绘图参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300


# ════════════════════════════════════════════════════
#  Jenga 塔参数
# ════════════════════════════════════════════════════

BLOCK_LENGTH = 0.15  # 15 cm
BLOCK_WIDTH = 0.05   # 5 cm
BLOCK_HEIGHT = 0.03  # 3 cm
NUM_LEVELS = 18
BLOCKS_PER_LEVEL = 3


# ════════════════════════════════════════════════════
#  3D 可视化
# ════════════════════════════════════════════════════

def create_block_vertices(center, length, width, height, rotation=0):
    """创建积木的 8 个顶点"""
    # 局部坐标系中的顶点
    l, w, h = length/2, width/2, height/2
    vertices = np.array([
        [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],  # 底面
        [-l, -w, h],  [l, -w, h],  [l, w, h],  [-l, w, h],   # 顶面
    ])
    
    # 旋转（绕 Z 轴）
    if rotation != 0:
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        rot_matrix = np.array([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ])
        vertices = vertices @ rot_matrix.T
    
    # 平移到中心位置
    vertices += center
    
    return vertices


def create_block_faces(vertices):
    """创建积木的 6 个面"""
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
    ]
    return faces


def visualize_tower_3d(decisions, title="Jenga Tower - Decision Sequence", 
                       save_path=None, view_angle=(30, 45)):
    """
    3D 塔可视化，显示决策顺序
    
    Args:
        decisions: 决策列表
        title: 图表标题
        save_path: 保存路径
        view_angle: 视角 (elevation, azimuth)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 记录哪些积木被移除了
    removed = {d['block_id']: (d['step'], d['success']) for d in decisions}
    
    # 颜色方案
    cmap = plt.cm.viridis
    num_steps = len(decisions)
    
    # 绘制所有积木
    for level in range(NUM_LEVELS):
        is_horizontal = (level % 2 == 0)
        z_center = level * BLOCK_HEIGHT + BLOCK_HEIGHT / 2
        
        for pos in range(BLOCKS_PER_LEVEL):
            block_id = level * 3 + pos
            
            # 计算积木中心位置
            if is_horizontal:
                # 水平方向（沿 X 轴）
                x_center = (pos - 1) * BLOCK_LENGTH / 3
                y_center = 0
                length, width = BLOCK_LENGTH / 3, BLOCK_WIDTH
                rotation = 0
            else:
                # 垂直方向（沿 Y 轴）
                x_center = 0
                y_center = (pos - 1) * BLOCK_LENGTH / 3
                length, width = BLOCK_WIDTH, BLOCK_LENGTH / 3
                rotation = np.pi / 2
            
            center = np.array([x_center, y_center, z_center])
            
            # 创建积木
            vertices = create_block_vertices(center, length, width, BLOCK_HEIGHT, rotation)
            faces = create_block_faces(vertices)
            
            # 确定颜色和透明度
            if block_id in removed:
                step, success = removed[block_id]
                color = cmap(step / max(num_steps - 1, 1))
                alpha = 0.9
                edgecolor = 'darkgreen' if success else 'darkred'
                linewidth = 2.5
            else:
                color = 'lightgray'
                alpha = 0.3
                edgecolor = 'gray'
                linewidth = 0.8
            
            # 绘制积木
            poly = Poly3DCollection(faces, facecolors=color, alpha=alpha,
                                   edgecolors=edgecolor, linewidths=linewidth)
            ax.add_collection3d(poly)
            
            # 标注决策顺序
            if block_id in removed:
                step, success = removed[block_id]
                # 在积木顶部标注步数
                text_z = z_center + BLOCK_HEIGHT / 2 + 0.005
                ax.text(x_center, y_center, text_z, f'{step+1}',
                       fontsize=14, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                                edgecolor='black', linewidth=2))
    
    # 设置坐标轴
    ax.set_xlabel('X (m)', fontsize=11, labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=11, labelpad=10)
    ax.set_zlabel('Height (m)', fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 设置视角
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # 设置坐标范围
    max_range = 0.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, NUM_LEVELS * BLOCK_HEIGHT + 0.05])
    
    # 设置背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                               norm=plt.Normalize(vmin=1, vmax=num_steps))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.1, aspect=30)
    cbar.set_label('Removal Step', fontsize=11)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
               markersize=10, markeredgecolor='darkgreen', markeredgewidth=2,
               label='Success'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=10, markeredgecolor='darkred', markeredgewidth=2,
               label='Collapse'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', 
               markersize=10, markeredgecolor='gray', markeredgewidth=1,
               label='Not Removed', alpha=0.5),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_tower_2d_side(decisions, title="Jenga Tower - Side View", save_path=None):
    """
    2D 侧视图，更清晰地显示层次结构
    """
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # 记录移除的积木
    removed = {d['block_id']: (d['step'], d['success']) for d in decisions}
    
    # 颜色方案
    cmap = plt.cm.viridis
    num_steps = len(decisions)
    
    # 绘制塔
    for level in range(NUM_LEVELS):
        is_horizontal = (level % 2 == 0)
        y = level * BLOCK_HEIGHT
        
        for pos in range(BLOCKS_PER_LEVEL):
            block_id = level * 3 + pos
            
            if is_horizontal:
                # 水平方向
                x = (pos - 1) * BLOCK_LENGTH / 3 - BLOCK_LENGTH / 6
                w = BLOCK_LENGTH / 3
                h = BLOCK_HEIGHT
            else:
                # 垂直方向（显示为小矩形）
                x = -BLOCK_WIDTH / 2
                w = BLOCK_WIDTH
                h = BLOCK_HEIGHT
            
            # 确定颜色
            if block_id in removed:
                step, success = removed[block_id]
                color = cmap(step / max(num_steps - 1, 1))
                alpha = 0.9
                edgecolor = 'darkgreen' if success else 'darkred'
                linewidth = 2.5
            else:
                color = 'lightgray'
                alpha = 0.4
                edgecolor = 'gray'
                linewidth = 1
            
            # 绘制矩形
            rect = FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.002",
                                 facecolor=color, alpha=alpha,
                                 edgecolor=edgecolor, linewidth=linewidth)
            ax.add_patch(rect)
            
            # 标注步数
            if block_id in removed:
                step, success = removed[block_id]
                # 使用圆形标注
                circle = Circle((x + w/2, y + h/2), 0.008, 
                              facecolor='white', edgecolor='black', linewidth=2, zorder=10)
                ax.add_patch(circle)
                ax.text(x + w/2, y + h/2, f'{step+1}',
                       ha='center', va='center', fontsize=11, fontweight='bold', zorder=11)
    
    # 设置坐标轴
    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(-0.02, NUM_LEVELS * BLOCK_HEIGHT + 0.02)
    ax.set_aspect('equal')
    ax.set_xlabel('Position (m)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加层数标注
    for level in range(0, NUM_LEVELS, 3):
        y = level * BLOCK_HEIGHT
        ax.text(-0.13, y, f'L{level}', ha='right', va='center', 
               fontsize=9, color='gray')
    
    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                               norm=plt.Normalize(vmin=1, vmax=num_steps))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Removal Step', fontsize=11)
    
    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
               markersize=12, markeredgecolor='darkgreen', markeredgewidth=2,
               label='Success'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=12, markeredgecolor='darkred', markeredgewidth=2,
               label='Collapse'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_comparison_grid(decision_files, model_names, episode=0, save_path=None):
    """
    网格对比多个模型的决策
    """
    n_models = len(decision_files)
    fig = plt.figure(figsize=(6*n_models, 14))
    
    for idx, (filepath, name) in enumerate(zip(decision_files, model_names)):
        # 加载决策
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and "decisions" in data:
                decisions = data["decisions"][episode]
            else:
                decisions = data[episode]
        
        # 创建子图
        ax = fig.add_subplot(1, n_models, idx+1)
        
        # 记录移除的积木
        removed = {d['block_id']: (d['step'], d['success']) for d in decisions}
        
        # 颜色方案
        cmap = plt.cm.viridis
        num_steps = len(decisions)
        
        # 绘制塔
        for level in range(NUM_LEVELS):
            is_horizontal = (level % 2 == 0)
            y = level * BLOCK_HEIGHT
            
            for pos in range(BLOCKS_PER_LEVEL):
                block_id = level * 3 + pos
                
                if is_horizontal:
                    x = (pos - 1) * BLOCK_LENGTH / 3 - BLOCK_LENGTH / 6
                    w = BLOCK_LENGTH / 3
                    h = BLOCK_HEIGHT
                else:
                    x = -BLOCK_WIDTH / 2
                    w = BLOCK_WIDTH
                    h = BLOCK_HEIGHT
                
                # 颜色
                if block_id in removed:
                    step, success = removed[block_id]
                    color = cmap(step / max(num_steps - 1, 1))
                    alpha = 0.9
                    edgecolor = 'darkgreen' if success else 'darkred'
                    linewidth = 2.5
                else:
                    color = 'lightgray'
                    alpha = 0.4
                    edgecolor = 'gray'
                    linewidth = 1
                
                # 绘制
                rect = FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.002",
                                     facecolor=color, alpha=alpha,
                                     edgecolor=edgecolor, linewidth=linewidth)
                ax.add_patch(rect)
                
                # 标注
                if block_id in removed:
                    step, success = removed[block_id]
                    circle = Circle((x + w/2, y + h/2), 0.008, 
                                  facecolor='white', edgecolor='black', linewidth=2, zorder=10)
                    ax.add_patch(circle)
                    ax.text(x + w/2, y + h/2, f'{step+1}',
                           ha='center', va='center', fontsize=10, fontweight='bold', zorder=11)
        
        # 设置
        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.02, NUM_LEVELS * BLOCK_HEIGHT + 0.02)
        ax.set_aspect('equal')
        ax.set_xlabel('Position (m)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Height (m)', fontsize=11)
        ax.set_title(f'{name}\n({len(decisions)} steps)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'Decision Sequence Comparison - Episode {episode}', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ════════════════════════════════════════════════════
#  主程序
# ════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="生成论文质量的可视化")
    p.add_argument("--decisions", type=str, nargs='+', required=True,
                   help="决策记录文件（JSON）")
    p.add_argument("--names", type=str, nargs='+',
                   help="模型名称（用于对比图）")
    p.add_argument("--episode", type=int, default=0,
                   help="可视化第几个 episode")
    p.add_argument("--output", type=str, default="paper_figures",
                   help="输出目录")
    p.add_argument("--views", type=str, nargs='+', 
                   default=['3d', '2d', 'comparison'],
                   choices=['3d', '2d', 'comparison'],
                   help="生成哪些视图")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("  生成论文质量可视化")
    print("=" * 60)
    
    # 如果只有一个文件，生成单模型可视化
    if len(args.decisions) == 1:
        filepath = args.decisions[0]
        name = args.names[0] if args.names else "Model"
        
        print(f"\n加载: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and "decisions" in data:
                decisions = data["decisions"][args.episode]
            else:
                decisions = data[args.episode]
        
        print(f"  Episode {args.episode}: {len(decisions)} steps")
        
        # 3D 视图
        if '3d' in args.views:
            print("\n生成 3D 视图...")
            save_path = os.path.join(args.output, f"{name}_ep{args.episode}_3d.png")
            visualize_tower_3d(decisions, 
                             title=f"{name} - Episode {args.episode}",
                             save_path=save_path,
                             view_angle=(25, 45))
        
        # 2D 侧视图
        if '2d' in args.views:
            print("\n生成 2D 侧视图...")
            save_path = os.path.join(args.output, f"{name}_ep{args.episode}_2d.png")
            visualize_tower_2d_side(decisions,
                                   title=f"{name} - Episode {args.episode}",
                                   save_path=save_path)
    
    # 多模型对比
    if len(args.decisions) > 1 and 'comparison' in args.views:
        print("\n生成对比图...")
        if not args.names or len(args.names) != len(args.decisions):
            args.names = [f"Model {i+1}" for i in range(len(args.decisions))]
        
        save_path = os.path.join(args.output, f"comparison_ep{args.episode}.png")
        visualize_comparison_grid(args.decisions, args.names, 
                                 episode=args.episode, save_path=save_path)
    
    print("\n" + "=" * 60)
    print(f"  所有可视化已保存到: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
