"""
diagnose_model.py — 诊断 RL 模型性能问题

分析可能的原因:
1. 训练不充分
2. 信息不对称（完美感知 vs 点云）
3. 探索不足
4. 奖励设计问题
5. 网络容量问题
"""
import argparse
import json
import os

import numpy as np
import torch

from vp3e_modules import VP3ENetwork, PriorGuidedActorCritic
from jenga_tower import NUM_BLOCKS


def analyze_checkpoint(ckpt_path):
    """分析 checkpoint 文件"""
    print(f"\n分析 checkpoint: {ckpt_path}")
    print("=" * 70)
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # 检查包含的键
    print(f"Checkpoint 包含的键: {list(ckpt.keys())}")
    
    if "global_step" in ckpt:
        print(f"  全局步数: {ckpt['global_step']:,}")
    if "iteration" in ckpt:
        print(f"  迭代次数: {ckpt['iteration']}")
    if "target_c" in ckpt:
        print(f"  课程难度: {ckpt['target_c']:.3f}")
    if "success_ema" in ckpt:
        print(f"  成功率 EMA: {ckpt['success_ema']:.3f}")
    
    # 分析网络参数
    if "vision_net" in ckpt:
        vision_params = sum(p.numel() for p in ckpt["vision_net"].values() if isinstance(p, torch.Tensor))
        print(f"  Vision 网络参数: {vision_params:,}")
    
    if "rl_net" in ckpt:
        rl_params = sum(p.numel() for p in ckpt["rl_net"].values() if isinstance(p, torch.Tensor))
        print(f"  RL 网络参数: {rl_params:,}")
        
        # 检查 Actor 的 alpha/beta 参数
        if "alpha" in ckpt["rl_net"]:
            alpha = ckpt["rl_net"]["alpha"].item()
            print(f"  Alpha (稳定性权重): {alpha:.4f}")
        if "beta" in ckpt["rl_net"]:
            beta = ckpt["rl_net"]["beta"].item()
            print(f"  Beta (潜能权重): {beta:.4f}")


def compare_baselines():
    """对比分析"""
    print("\n" + "=" * 70)
    print("Heuristic vs RL 对比分析")
    print("=" * 70)
    
    print("\n【信息不对称问题】")
    print("  Heuristic Baseline:")
    print("    ✓ 使用完美感知（物理引擎精确位姿）")
    print("    ✓ 无噪声、无遮挡")
    print("    ✓ 毫米级精度")
    print("    ✓ 计算简单、确定性强")
    
    print("\n  RL Model:")
    print("    ✗ 使用点云视觉输入")
    print("    ✗ 有噪声、有遮挡")
    print("    ✗ 需要学习特征提取")
    print("    ✗ 需要学习物理推理")
    
    print("\n【可能的问题】")
    problems = [
        ("1. 训练不充分", 
         "280 iterations 可能不够，建议至少 1000+ iterations"),
        
        ("2. 信息不对称", 
         "Heuristic 用完美感知，RL 用点云，这是不公平的对比"),
        
        ("3. 奖励稀疏", 
         "只有成功/失败的奖励，缺少中间引导信号"),
        
        ("4. 探索不足", 
         "课程学习可能过于保守，没有探索到足够多样的场景"),
        
        ("5. 网络容量", 
         "点云处理 + 物理推理需要更大的网络"),
        
        ("6. 蒸馏权重衰减", 
         "λ_t 衰减可能过快，导致先验引导过早消失"),
    ]
    
    for title, desc in problems:
        print(f"\n  {title}")
        print(f"    {desc}")


def suggest_improvements():
    """改进建议"""
    print("\n" + "=" * 70)
    print("改进建议")
    print("=" * 70)
    
    suggestions = [
        ("1. 公平对比", [
            "创建基于点云的 Heuristic Baseline",
            "或创建使用完美感知的 RL Baseline",
            "确保两者使用相同的输入信息",
        ]),
        
        ("2. 延长训练", [
            "增加 total_steps 到 500K-1M",
            "监控训练曲线，确保收敛",
            "保存多个 checkpoint 进行对比",
        ]),
        
        ("3. 改进奖励", [
            "添加 shaped reward（如：选择稳定块 +0.1）",
            "添加 potential-based reward",
            "惩罚选择不稳定的块（即使没坍塌）",
        ]),
        
        ("4. 调整课程学习", [
            "降低 success_threshold（如 0.7）",
            "增大 complexity_step（如 0.1）",
            "更快地增加难度",
        ]),
        
        ("5. 调整蒸馏衰减", [
            "减慢 λ_t 衰减速度",
            "或使用固定的 λ（不衰减）",
            "让先验引导持续更长时间",
        ]),
        
        ("6. 数据增强", [
            "点云随机旋转",
            "点云随机采样",
            "增加训练数据多样性",
        ]),
        
        ("7. 网络架构", [
            "增加 GNN 层数（如 6-8 层）",
            "增加特征维度（如 512）",
            "使用更强的点云编码器（如 PointNet++）",
        ]),
    ]
    
    for title, items in suggestions:
        print(f"\n{title}")
        for item in items:
            print(f"  • {item}")


def create_fair_comparison_script():
    """生成公平对比的代码示例"""
    print("\n" + "=" * 70)
    print("公平对比代码示例")
    print("=" * 70)
    
    code = '''
# 方案 1: 基于点云的 Heuristic Baseline
class PointCloudHeuristicAgent:
    """使用点云估计位姿，然后应用几何规则"""
    
    def select_action(self, env, obs, info):
        # 1. 渲染点云
        pcd = render_point_cloud(env)  # [N, K, 3]
        
        # 2. 从点云估计每个块的中心位置
        centers = []
        for i in range(len(pcd)):
            if len(pcd[i]) > 0:
                center = pcd[i].mean(axis=0)  # 简单平均
                centers.append(center)
            else:
                centers.append([0, 0, 0])
        
        # 3. 应用几何规则（与原 Heuristic 相同）
        z_values = [c[2] for c in centers]
        # ... 过滤 + 选择逻辑
        
        return selected_block_id

# 方案 2: 使用完美感知的 RL Baseline
class OracleRLAgent:
    """RL 模型，但输入是完美位姿而非点云"""
    
    def select_action(self, env, obs, info):
        # 1. 获取精确位姿
        poses = []
        for actor in env.unwrapped.blocks:
            pose = actor._objs[0].get_pose()
            poses.append(pose.p)
        
        # 2. 转换为特征向量
        features = self._encode_poses(poses)
        
        # 3. RL 网络推理
        action = self.rl_net(features)
        return action
'''
    
    print(code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="runs/jenga_ppo")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RL 模型性能诊断工具")
    print("=" * 70)
    
    # 分析最终 checkpoint
    final_ckpt = os.path.join(args.checkpoint_dir, "ckpt_0280.pt")
    if os.path.exists(final_ckpt):
        analyze_checkpoint(final_ckpt)
    else:
        print(f"未找到 checkpoint: {final_ckpt}")
    
    # 对比分析
    compare_baselines()
    
    # 改进建议
    suggest_improvements()
    
    # 公平对比代码
    create_fair_comparison_script()
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)
    print("\n建议优先级:")
    print("  1. 创建公平的对比 baseline（使用相同输入）")
    print("  2. 延长训练时间（至少 500K steps）")
    print("  3. 调整蒸馏衰减速度（保持先验引导更久）")
    print("  4. 改进奖励函数（添加 shaped reward）")
    print()


if __name__ == "__main__":
    main()
