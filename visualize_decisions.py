"""
visualize_decisions.py — 可视化不同模型的积木选择顺序

功能：
1. 记录每个模型在每步选择的积木
2. 生成可视化图表：
   - 积木选择的时间序列
   - 积木高度分布
   - 积木位置热力图
3. 对比不同模型的策略差异

Usage:
    python visualize_decisions.py --episodes 5 --output visualizations/
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ════════════════════════════════════════════════════
#  数据结构
# ════════════════════════════════════════════════════

class DecisionRecorder:
    """记录模型的决策序列"""
    
    def __init__(self):
        self.episodes = []
        self.current_episode = []
    
    def record_step(self, step, block_id, success):
        """记录一步决策"""
        level = block_id // 3
        position = block_id % 3
        
        self.current_episode.append({
            'step': step,
            'block_id': block_id,
            'level': level,
            'position': position,
            'success': success,
        })
    
    def end_episode(self):
        """结束当前 episode"""
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []
    
    def save(self, filepath):
        """保存到 JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.episodes, f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """从 JSON 加载"""
        recorder = cls()
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            # 兼容两种格式
            if isinstance(data, list):
                # 格式 1: 直接是 episodes 列表
                recorder.episodes = data
            elif isinstance(data, dict) and "decisions" in data:
                # 格式 2: evaluate_jenga.py 保存的格式
                recorder.episodes = data["decisions"]
            else:
                raise ValueError(f"无法识别的 JSON 格式: {filepath}")
        
        return recorder


# ════════════════════════════════════════════════════
#  可视化函数
# ════════════════════════════════════════════════════

def visualize_single_episode(decisions, title="Episode Decision Sequence", save_path=None):
    """
    可视化单个 episode 的决策序列
    
    显示：
    - X 轴：步数
    - Y 轴：积木层数
    - 颜色：成功/失败
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = [d['step'] for d in decisions]
    levels = [d['level'] for d in decisions]
    success = [d['success'] for d in decisions]
    
    # 绘制散点图
    colors = ['green' if s else 'red' for s in success]
    ax.scatter(steps, levels, c=colors, s=200, alpha=0.6, edgecolors='black', linewidth=2)
    
    # 连线
    ax.plot(steps, levels, 'b--', alpha=0.3, linewidth=1)
    
    # 标注积木 ID
    for d in decisions:
        ax.text(d['step'], d['level'], str(d['block_id']), 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Level (Height)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 18)
    
    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=10, label='Success'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Collapse'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_level_distribution(recorders, model_names, save_path=None):
    """
    可视化不同模型选择的层数分布
    
    对比不同模型倾向于选择哪些层的积木
    """
    fig, axes = plt.subplots(1, len(recorders), figsize=(5*len(recorders), 4))
    
    if len(recorders) == 1:
        axes = [axes]
    
    for ax, recorder, name in zip(axes, recorders, model_names):
        # 统计每层被选择的次数
        level_counts = defaultdict(int)
        total_steps = 0
        
        for episode in recorder.episodes:
            for decision in episode:
                if decision['success']:  # 只统计成功的
                    level_counts[decision['level']] += 1
                    total_steps += 1
        
        # 绘制柱状图
        levels = sorted(level_counts.keys())
        counts = [level_counts[l] for l in levels]
        percentages = [c / total_steps * 100 if total_steps > 0 else 0 for c in counts]
        
        bars = ax.bar(levels, percentages, color='steelblue', alpha=0.7, edgecolor='black')
        
        # 标注百分比
        for bar, pct in zip(bars, percentages):
            if pct > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Level (Height)', fontsize=11)
        ax.set_ylabel('Selection Frequency (%)', fontsize=11)
        ax.set_title(f'{name}\nLevel Distribution', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.5, 17.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_position_heatmap(recorders, model_names, save_path=None):
    """
    可视化积木位置热力图
    
    显示每个位置（层 × 位置）被选择的频率
    """
    fig, axes = plt.subplots(1, len(recorders), figsize=(4*len(recorders), 8))
    
    if len(recorders) == 1:
        axes = [axes]
    
    for ax, recorder, name in zip(axes, recorders, model_names):
        # 创建 18×3 的热力图矩阵
        heatmap = np.zeros((18, 3))
        
        for episode in recorder.episodes:
            for decision in episode:
                if decision['success']:
                    level = decision['level']
                    pos = decision['position']
                    heatmap[level, pos] += 1
        
        # 归一化
        if heatmap.sum() > 0:
            heatmap = heatmap / heatmap.sum() * 100
        
        # 绘制热力图
        im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=0, vmax=heatmap.max())
        
        # 标注数值
        for i in range(18):
            for j in range(3):
                if heatmap[i, j] > 0:
                    text = ax.text(j, i, f'{heatmap[i, j]:.1f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Left', 'Center', 'Right'])
        ax.set_yticks(range(18))
        ax.set_yticklabels([f'L{i}' for i in range(18)])
        ax.set_xlabel('Position', fontsize=11)
        ax.set_ylabel('Level', fontsize=11)
        ax.set_title(f'{name}\nPosition Heatmap (%)', fontsize=12, fontweight='bold')
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Selection Frequency (%)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_strategy_comparison(recorders, model_names, save_path=None):
    """
    对比不同模型的策略
    
    显示：
    - 平均选择高度
    - 选择的层数范围
    - 成功率
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. 平均选择高度
    avg_levels = []
    for recorder in recorders:
        levels = []
        for episode in recorder.episodes:
            for decision in episode:
                if decision['success']:
                    levels.append(decision['level'])
        avg_levels.append(np.mean(levels) if levels else 0)
    
    axes[0].bar(model_names, avg_levels, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Average Level', fontsize=11)
    axes[0].set_title('Average Selection Height', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 标注数值
    for i, val in enumerate(avg_levels):
        axes[0].text(i, val + 0.2, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 2. 层数范围（箱线图）
    level_data = []
    for recorder in recorders:
        levels = []
        for episode in recorder.episodes:
            for decision in episode:
                if decision['success']:
                    levels.append(decision['level'])
        level_data.append(levels if levels else [0])
    
    bp = axes[1].boxplot(level_data, labels=model_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Level', fontsize=11)
    axes[1].set_title('Level Distribution Range', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. 成功率
    success_rates = []
    for recorder in recorders:
        total = 0
        success = 0
        for episode in recorder.episodes:
            for decision in episode:
                total += 1
                if decision['success']:
                    success += 1
        success_rates.append(success / total * 100 if total > 0 else 0)
    
    bars = axes[2].bar(model_names, success_rates, color='green', alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Success Rate (%)', fontsize=11)
    axes[2].set_title('Step Success Rate', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # 标注数值
    for i, val in enumerate(success_rates):
        axes[2].text(i, val + 2, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_tower_view(decisions, title="Tower View", save_path=None):
    """
    可视化塔的侧视图，显示积木被移除的顺序
    
    每个积木用矩形表示，颜色表示移除顺序
    """
    fig, ax = plt.subplots(figsize=(8, 12))
    
    # Jenga 塔参数
    num_levels = 18
    blocks_per_level = 3
    block_length = 0.15
    block_width = 0.05
    block_height = 0.03
    
    # 创建颜色映射
    num_steps = len(decisions)
    colors = plt.cm.viridis(np.linspace(0, 1, num_steps))
    
    # 记录哪些积木被移除了
    removed = {d['block_id']: (d['step'], colors[d['step']]) for d in decisions}
    
    # 绘制塔
    for level in range(num_levels):
        # 奇数层和偶数层方向不同
        is_horizontal = (level % 2 == 0)
        
        for pos in range(blocks_per_level):
            block_id = level * 3 + pos
            
            if is_horizontal:
                # 水平方向
                x = pos * block_length / 3 - block_length / 2
                y = level * block_height
                w = block_length / 3
                h = block_height
            else:
                # 垂直方向
                x = -block_width / 2
                y = level * block_height + pos * block_length / 3
                w = block_width
                h = block_length / 3
            
            # 确定颜色
            if block_id in removed:
                step, color = removed[block_id]
                alpha = 0.8
                edgecolor = 'red'
                linewidth = 2
            else:
                color = 'lightgray'
                alpha = 0.3
                edgecolor = 'gray'
                linewidth = 1
            
            # 绘制矩形
            rect = patches.Rectangle((x, y), w, h, 
                                     facecolor=color, alpha=alpha,
                                     edgecolor=edgecolor, linewidth=linewidth)
            ax.add_patch(rect)
            
            # 标注积木 ID 和步数
            if block_id in removed:
                step, _ = removed[block_id]
                ax.text(x + w/2, y + h/2, f'{block_id}\n#{step+1}',
                       ha='center', va='center', fontsize=7, fontweight='bold')
    
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.05, num_levels * block_height + 0.05)
    ax.set_aspect('equal')
    ax.set_xlabel('Position (m)', fontsize=11)
    ax.set_ylabel('Height (m)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=0, vmax=num_steps-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Removal Step', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ════════════════════════════════════════════════════
#  主程序
# ════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="可视化模型决策序列")
    p.add_argument("--decisions", type=str, nargs='+', required=True,
                   help="决策记录文件路径（JSON）")
    p.add_argument("--names", type=str, nargs='+', required=True,
                   help="模型名称")
    p.add_argument("--output", type=str, default="visualizations",
                   help="输出目录")
    p.add_argument("--episode", type=int, default=0,
                   help="可视化第几个 episode（用于单 episode 可视化）")
    return p.parse_args()


def main():
    args = parse_args()
    
    assert len(args.decisions) == len(args.names), \
        "决策文件数量和模型名称数量必须相同"
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("  可视化模型决策序列")
    print("=" * 60)
    
    # 加载决策记录
    recorders = []
    for filepath, name in zip(args.decisions, args.names):
        print(f"\n加载: {name} <- {filepath}")
        recorder = DecisionRecorder.load(filepath)
        print(f"  Episodes: {len(recorder.episodes)}")
        recorders.append(recorder)
    
    # 1. 单 episode 可视化（每个模型）
    print(f"\n生成单 episode 可视化（Episode {args.episode}）...")
    for recorder, name in zip(recorders, args.names):
        if args.episode < len(recorder.episodes):
            decisions = recorder.episodes[args.episode]
            save_path = os.path.join(args.output, f"{name}_ep{args.episode}_sequence.png")
            visualize_single_episode(decisions, 
                                    title=f"{name} - Episode {args.episode}",
                                    save_path=save_path)
            
            # 塔视图
            save_path = os.path.join(args.output, f"{name}_ep{args.episode}_tower.png")
            visualize_tower_view(decisions,
                                title=f"{name} - Episode {args.episode} Tower View",
                                save_path=save_path)
    
    # 2. 层数分布对比
    print("\n生成层数分布对比...")
    save_path = os.path.join(args.output, "level_distribution.png")
    visualize_level_distribution(recorders, args.names, save_path)
    
    # 3. 位置热力图
    print("\n生成位置热力图...")
    save_path = os.path.join(args.output, "position_heatmap.png")
    visualize_position_heatmap(recorders, args.names, save_path)
    
    # 4. 策略对比
    print("\n生成策略对比...")
    save_path = os.path.join(args.output, "strategy_comparison.png")
    visualize_strategy_comparison(recorders, args.names, save_path)
    
    print("\n" + "=" * 60)
    print(f"  所有可视化已保存到: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
