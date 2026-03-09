"""
evaluate_jenga.py — Unified Evaluation Harness for Jenga Block Extraction

统一评测脚本，支持评估多种模型:
  - Heuristic Baseline (基于几何规则 + 完美感知)
  - PointCloud Heuristic (基于几何规则 + 点云，公平对比)
  - VLM Baseline (GPT-4o/Claude 等)
  - RL Model (PPO 训练的完整模型)

评测指标:
  - Success Rate (SR): 每 episode 平均成功操作率（不导致坍塌）
  - Max Extracted Blocks (MEB): 每 episode 成功抽取的最大积木数
  - OOD Generalization: 在极端 OOD 拓扑（稀疏支撑、超高塔）上的成功率

Usage:
    # 评估 Heuristic Baseline (完美感知)
    python evaluate_jenga.py --model heuristic --num_episodes 100 --seed 42
    
    # 评估 Point Cloud Heuristic (公平对比)
    python evaluate_jenga.py --model pointcloud_heuristic --num_episodes 100 --seed 42
    
    # 评估 VLM Baseline
    python evaluate_jenga.py --model vlm --num_episodes 100 --seed 42 --api_key sk-A6WNIdgvHjFkR1vaIyqgEyoP6KnXQgld6Zy75b2Vypz1DCE0 --base_url https://www.chataiapi.com/v1 --vlm_model gemini-2.5-pro
    
    # 评估 RL Model (方式1: 指定目录)
    python evaluate_jenga.py --model rl --num_episodes 100 --seed 42 --checkpoint runs/jenga_ppo
    
    # 评估 RL Model (方式2: 分别指定)
    python evaluate_jenga.py --model rl --num_episodes 100 --seed 42 --vision_checkpoint runs/jenga_ppo/vision_final.pt --rl_checkpoint runs/jenga_ppo/rl_final.pt
    
    # OOD 评估（极端拓扑）
    python evaluate_jenga.py --model heuristic --num_episodes 100 --seed 42 --ood_mode sparse  # 或 ultra_high
"""
import argparse
import base64
import io
import json
import os
import re
import time
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont

from jenga_tower import NUM_BLOCKS, NUM_LEVELS
from jenga_ppo_wrapper import JengaPPOWrapper


# ════════════════════════════════════════════════════
#  Agent 接口定义
# ════════════════════════════════════════════════════

class BaseAgent:
    """所有 Agent 的基类"""
    
    def select_action(self, env, obs, info):
        """
        选择动作
        
        Args:
            env: Gym environment
            obs: 当前观测
            info: 环境信息字典（包含 mask 等）
        
        Returns:
            action: int, 选中的积木 ID
        """
        raise NotImplementedError
    
    def reset(self):
        """重置 Agent 状态（如果需要）"""
        pass


class PointCloudHeuristicAgent(BaseAgent):
    """基于点云的几何规则策略（公平对比版本）"""
    
    def __init__(self, tower_center=(0.0, 0.0), n_pts=256, min_points=10):
        self.tower_center = np.array(tower_center)
        self.n_pts = n_pts
        self.min_points = min_points
        
        from jenga_tower import render_point_cloud
        self.render_point_cloud = render_point_cloud
    
    def _render_pcd(self, env):
        """渲染点云"""
        uw = env.unwrapped
        cams = {k: v for k, v in uw.scene.sensors.items() if k.startswith("surround")}
        pcd_data = self.render_point_cloud(uw.scene, cams, uw.blocks)
        return pcd_data["per_block_pcd"]
    
    def _estimate_center(self, pointcloud):
        """从点云估计中心"""
        if len(pointcloud) < self.min_points:
            return None
        xyz = pointcloud[:, :3]
        return xyz.mean(axis=0)
    
    def select_action(self, env, obs, info):
        mask = info["mask"]
        valid_ids = [i for i in range(NUM_BLOCKS) if mask[i]]
        
        if not valid_ids:
            raise ValueError("没有可抽取的积木")
        
        # 渲染点云
        per_block_pcd = self._render_pcd(env)
        
        # 估计中心
        centers = {}
        for idx in valid_ids:
            center = self._estimate_center(per_block_pcd[idx])
            if center is not None:
                centers[idx] = center
        
        if not centers:
            return np.random.choice(valid_ids)
        
        visible_ids = list(centers.keys())
        positions = [(idx, centers[idx][0], centers[idx][1], centers[idx][2]) 
                     for idx in visible_ids]
        
        # 计算 Z 范围
        z_values = [p[3] for p in positions]
        z_sorted = sorted(set(z_values))
        
        if len(z_sorted) <= 3:
            non_top = [p for p in positions if p[3] < max(z_values) - 0.01]
            if non_top:
                return non_top[np.random.randint(len(non_top))][0]
            return np.random.choice(visible_ids)
        
        # 过滤
        top_two_threshold = z_sorted[-2] - 0.01
        bottom_threshold = z_sorted[0] + 0.01
        
        candidates = [
            p for p in positions
            if p[3] > bottom_threshold and p[3] < top_two_threshold
        ]
        
        if not candidates:
            non_top = [p for p in positions if p[3] < max(z_values) - 0.01]
            if non_top:
                return non_top[np.random.randint(len(non_top))][0]
            return np.random.choice(visible_ids)
        
        # 选择距离中心最近的
        distances = []
        for idx, x, y, z in candidates:
            dist_2d = np.sqrt((x - self.tower_center[0])**2 + 
                            (y - self.tower_center[1])**2)
            distances.append((idx, dist_2d))
        
        distances.sort(key=lambda x: x[1])
        return distances[0][0]


class HeuristicAgent(BaseAgent):
    """基于几何规则的启发式策略"""
    
    def __init__(self, tower_center=(0.0, 0.0)):
        self.tower_center = np.array(tower_center)
    
    def select_action(self, env, obs, info):
        mask = info["mask"]
        valid_ids = [i for i in range(NUM_BLOCKS) if mask[i]]
        
        if not valid_ids:
            raise ValueError("没有可抽取的积木")
        
        # 获取积木 actors
        uw = env.unwrapped
        actors = [blk._objs[0] for blk in uw.blocks]
        
        # 获取位姿
        positions = []
        for idx in valid_ids:
            pose = actors[idx].get_pose()
            pos = pose.p
            positions.append((idx, pos[0], pos[1], pos[2]))
        
        # 计算 Z 范围
        z_values = [p[3] for p in positions]
        z_min = min(z_values)
        z_max = max(z_values)
        z_sorted = sorted(set(z_values))
        
        # 层数太少，回退策略
        if len(z_sorted) <= 3:
            non_top = [p for p in positions if p[3] < z_max - 0.01]
            if non_top:
                return non_top[np.random.randint(len(non_top))][0]
            return np.random.choice(valid_ids)
        
        # 过滤: 排除最高两层和最底层
        top_two_threshold = z_sorted[-2] - 0.005
        bottom_threshold = z_sorted[0] + 0.005
        
        candidates = [
            p for p in positions
            if p[3] > bottom_threshold and p[3] < top_two_threshold
        ]
        
        # 候选池为空，回退
        if not candidates:
            non_top = [p for p in positions if p[3] < z_max - 0.01]
            if non_top:
                return non_top[np.random.randint(len(non_top))][0]
            return np.random.choice(valid_ids)
        
        # 选择距离塔中心最近的
        distances = []
        for idx, x, y, z in candidates:
            dist_2d = np.sqrt((x - self.tower_center[0])**2 + 
                            (y - self.tower_center[1])**2)
            distances.append((idx, dist_2d))
        
        distances.sort(key=lambda x: x[1])
        return distances[0][0]


class VLMAgent(BaseAgent):
    """基于 VLM (GPT-4o/Claude) 的策略"""
    
    def __init__(self, api_key, base_url, model="gpt-4o", camera_uids=None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.camera_uids = camera_uids or ["surround_0"]
        self.removed_ids = []
        
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except Exception:
            self.font = ImageFont.load_default()
    
    def reset(self):
        self.removed_ids = []
    
    def select_action(self, env, obs, info):
        mask = info["mask"]
        valid_ids = [i for i in range(NUM_BLOCKS) if mask[i]]
        
        # 渲染标注图像
        imgs, visible = self._render_annotated(env)
        final_img = self._composite_grid(imgs) if len(imgs) > 1 else imgs[0]
        img_b64 = self._encode_image(final_img)
        
        # 查询 VLM
        try:
            block_id, _ = self._query_vlm(img_b64, valid_ids)
        except Exception as e:
            print(f"    VLM 错误: {e}, 回退到最高层")
            block_id = None
        
        if block_id is None:
            block_id = max(valid_ids)
        
        self.removed_ids.append(block_id)
        return block_id
    
    def _render_annotated(self, env):
        """渲染并标注图像"""
        uw = env.unwrapped
        scene = uw.scene
        
        scene.update_render(update_sensors=True, update_human_render_cameras=False)
        sensors = {uid: scene.sensors[uid] for uid in self.camera_uids}
        for s in sensors.values():
            s.capture()
        
        sid_to_idx = {blk._objs[0].per_scene_id: i for i, blk in enumerate(uw.blocks)}
        
        images = []
        all_visible = set()
        
        for cam_uid in self.camera_uids:
            obs = sensors[cam_uid].get_obs(rgb=True, depth=False, segmentation=True, position=False)
            rgb = obs["rgb"][0].cpu().numpy()
            seg = obs["segmentation"][0].cpu().numpy()[:, :, 0].astype(int)
            
            rgb_u8 = rgb.astype(np.uint8) if rgb.max() > 1 else (rgb * 255).astype(np.uint8)
            img = Image.fromarray(rgb_u8)
            draw = ImageDraw.Draw(img)
            
            for sid, idx in sid_to_idx.items():
                pmask = (seg == sid)
                if pmask.sum() < 20:
                    continue
                ys, xs = np.where(pmask)
                cy, cx = int(ys.mean()), int(xs.mean())
                
                text = str(idx)
                bb = draw.textbbox((0, 0), text, font=self.font)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                draw.rectangle(
                    [cx - tw // 2 - 2, cy - th // 2 - 1,
                     cx + tw // 2 + 2, cy + th // 2 + 1],
                    fill="black",
                )
                draw.text((cx - tw // 2, cy - th // 2), text, fill="white", font=self.font)
                all_visible.add(idx)
            
            images.append(np.array(img))
        
        return images, sorted(all_visible)
    
    def _composite_grid(self, images):
        """拼接多张图"""
        if len(images) == 1:
            return images[0]
        cols = 2
        rows = (len(images) + 1) // 2
        h, w = images[0].shape[:2]
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for i, img in enumerate(images):
            r, c = divmod(i, cols)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
        return canvas
    
    def _encode_image(self, image_np):
        """Base64 编码"""
        buf = io.BytesIO()
        Image.fromarray(image_np).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    
    def _query_vlm(self, image_b64, valid_ids, retries=2):
        """查询 VLM"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        system_prompt = (
            "你是一个精通结构力学和积木物理的专家。"
            "你将看到 Jenga 塔的图像，每块积木上标注了数字 ID。"
            "你的任务是选出抽取后最不可能导致塔倒塌的积木。"
        )
        
        valid_str = ", ".join(map(str, sorted(valid_ids)))
        removed_str = ", ".join(map(str, sorted(self.removed_ids))) if self.removed_ids else "无"
        user_prompt = (
            f"图中是当前 Jenga 塔。标注数字 = 积木 ID。\n\n"
            f"当前可抽取 ID: [{valid_str}]\n"
            f"已抽走 ID: [{removed_str}]\n\n"
            f"请以 JSON 输出推荐:\n"
            f'```json\n{{"block_id": <ID>, "reasoning": "<理由>"}}\n```'
        )
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]},
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
        }
        
        for attempt in range(retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return self._parse_response(text, valid_ids), text
            except Exception as e:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
    
    def _parse_response(self, text, valid_ids):
        """解析 VLM 响应"""
        valid_set = set(valid_ids)
        
        # JSON 代码块
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                bid = int(json.loads(m.group(1))["block_id"])
                if bid in valid_set:
                    return bid
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        
        # 裸 JSON
        m = re.search(r'\{[^{}]*"block_id"\s*:\s*(\d+)[^{}]*\}', text)
        if m:
            bid = int(m.group(1))
            if bid in valid_set:
                return bid
        
        # 文本中第一个有效数字
        for m in re.finditer(r"\b(\d{1,2})\b", text):
            bid = int(m.group(1))
            if bid in valid_set:
                return bid
        
        return None


class RLAgent(BaseAgent):
    """基于 PPO 训练的 RL 策略 (VP3E + PriorGuidedActorCritic)"""
    
    def __init__(self, vision_checkpoint, rl_checkpoint, device="cpu", 
                 feat_dim=256, gnn_layers=4, n_pts=256):
        self.device = torch.device(device)
        self.n_pts = n_pts
        
        # 导入模型定义
        from vp3e_modules import VP3ENetwork, PriorGuidedActorCritic
        from jenga_tower import NUM_BLOCKS, render_point_cloud
        
        self.render_point_cloud = render_point_cloud
        self.NUM_BLOCKS = NUM_BLOCKS
        
        # 加载 Vision 网络
        self.vision_net = VP3ENetwork(
            feat_dim=feat_dim, 
            gnn_layers=gnn_layers, 
            max_blocks=NUM_BLOCKS
        ).to(self.device)
        
        if os.path.isfile(vision_checkpoint):
            self.vision_net.load_state_dict(
                torch.load(vision_checkpoint, map_location=self.device)
            )
            print(f"  Vision 网络加载: {vision_checkpoint}")
        else:
            raise FileNotFoundError(f"Vision checkpoint 不存在: {vision_checkpoint}")
        
        # 加载 RL 网络
        self.rl_net = PriorGuidedActorCritic(feat_dim=feat_dim).to(self.device)
        
        if os.path.isfile(rl_checkpoint):
            self.rl_net.load_state_dict(
                torch.load(rl_checkpoint, map_location=self.device)
            )
            print(f"  RL 网络加载: {rl_checkpoint}")
        else:
            raise FileNotFoundError(f"RL checkpoint 不存在: {rl_checkpoint}")
        
        self.vision_net.eval()
        self.rl_net.eval()
    
    def _render_vp3e_input(self, env):
        """渲染点云输入 [1, N, K, 3]"""
        uw = env.unwrapped
        cams = {k: v for k, v in uw.scene.sensors.items() if k.startswith("surround")}
        pcd_data = self.render_point_cloud(uw.scene, cams, uw.blocks)
        
        N = len(uw.blocks)
        obs = np.zeros((1, N, self.n_pts, 3), dtype=np.float32)
        for i, pc in enumerate(pcd_data["per_block_pcd"]):
            if len(pc) == 0:
                continue
            xyz = pc[:, :3]
            if len(xyz) >= self.n_pts:
                obs[0, i] = xyz[np.random.choice(len(xyz), self.n_pts, replace=False)]
            else:
                obs[0, i, :len(xyz)] = xyz
        return torch.from_numpy(obs)
    
    def select_action(self, env, obs, info):
        """使用 RL 模型选择动作"""
        mask = info["mask"]
        
        # 渲染点云
        pcd = self._render_vp3e_input(env).to(self.device)
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # VP3E 前向
            vp3e_out = self.vision_net(pcd, mask_t, use_causal=True)
            
            # Actor-Critic 选择动作
            action, _, _, _ = self.rl_net.get_action_and_value(
                vp3e_out["Z"], 
                vp3e_out["stab_hat"],
                vp3e_out["pot_hat"], 
                mask_t
            )
        
        return action.item()


# ════════════════════════════════════════════════════
#  评测逻辑
# ════════════════════════════════════════════════════

def create_env(ood_mode=None, seed=None):
    """
    创建环境
    
    Args:
        ood_mode: None (easy), "moderate" (中等), "hard" (困难), 
                  "sparse" (稀疏支撑, 等同于 hard), "ultra_high" (超高塔, 极高)
        seed: 随机种子
    """
    base_env = gym.make(
        "JengaTower-v1", 
        obs_mode="state", 
        render_mode="rgb_array",
        num_envs=1, 
        sim_backend="cpu",
    )
    env = JengaPPOWrapper(base_env, lambda_int=0.0)
    
    # 根据场景设置难度参数
    # Easy (0.0-0.3): 标准配置，塔完整稳定
    # Moderate (0.4-0.6): 中等难度，部分积木已移除
    # Hard (0.7-0.9): 高难度，稀疏支撑，塔不稳定
    # Ultra-Hard (1.0): 极高难度，塔非常不稳定
    
    if ood_mode == "moderate":
        target_c = 0.5  # 中等难度
        difficulty = "Moderate"
    elif ood_mode == "hard" or ood_mode == "sparse":
        target_c = 0.8  # 高难度（稀疏支撑）
        difficulty = "Hard (Sparse Support)"
    elif ood_mode == "ultra_high":
        target_c = 1.0  # 极高难度（超高塔）
        difficulty = "Ultra-Hard (Ultra-High Tower)"
    else:
        target_c = 0.3  # 标准难度（Easy）
        difficulty = "Easy (Standard)"
    
    if ood_mode is not None:
        print(f"  场景难度: {difficulty} (target_c={target_c})")
    
    # Reset 环境并应用难度
    if seed is not None:
        env.reset(seed=seed, target_c=target_c)
    else:
        env.reset(target_c=target_c)
    
    return env


def evaluate_agent(
    agent: BaseAgent,
    num_episodes: int,
    max_steps: int,
    seed: int,
    ood_mode: str = None,
    verbose: bool = True,
    record_decisions: bool = False,
) -> Dict:
    """
    评估单个 Agent
    
    Returns:
        results: dict 包含 SR, MEB, 每 episode 详细数据
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = create_env(ood_mode=ood_mode, seed=seed)
    
    all_meb = []
    all_success_steps = []
    all_total_steps = []
    episode_details = []
    
    # 决策记录
    decision_records = [] if record_decisions else None
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        agent.reset()
        
        ep_success = 0
        ep_total = 0
        ep_collapsed = False
        ep_decisions = [] if record_decisions else None
        
        for step in range(max_steps):
            mask = info["mask"]
            valid_ids = [i for i in range(NUM_BLOCKS) if mask[i]]
            
            if not valid_ids:
                break
            
            # 选择动作
            try:
                action = agent.select_action(env, obs, info)
            except Exception as e:
                if verbose:
                    print(f"  Episode {ep + 1}, Step {step + 1}: Agent 错误 ({e})")
                action = np.random.choice(valid_ids)
            
            # 执行
            obs, reward, done, _, info = env.step(action)
            ep_total += 1
            
            collapsed = info.get("collapsed", False)
            
            # 记录决策
            if record_decisions:
                ep_decisions.append({
                    'step': step,
                    'block_id': int(action),
                    'level': int(action // 3),
                    'position': int(action % 3),
                    'success': not collapsed,
                })
            
            if collapsed:
                ep_collapsed = True
                break
            
            ep_success += 1
            
            if done:
                break
        
        all_meb.append(ep_success)
        all_success_steps.append(ep_success)
        all_total_steps.append(ep_total)
        
        episode_details.append({
            "episode": ep + 1,
            "extracted": ep_success,
            "total_steps": ep_total,
            "collapsed": ep_collapsed,
        })
        
        if record_decisions:
            decision_records.append(ep_decisions)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"  Progress: {ep + 1}/{num_episodes} episodes")
    
    env.close()
    
    # 计算指标
    total_success = sum(all_success_steps)
    total_attempts = sum(all_total_steps)
    sr = total_success / max(total_attempts, 1)
    meb_mean = np.mean(all_meb)
    meb_std = np.std(all_meb)
    meb_max = np.max(all_meb)
    
    results = {
        "sr": sr,
        "meb_mean": meb_mean,
        "meb_std": meb_std,
        "meb_max": meb_max,
        "total_success": total_success,
        "total_attempts": total_attempts,
        "num_episodes": num_episodes,
        "episode_details": episode_details,
        "all_meb": all_meb,
    }
    
    if record_decisions:
        results["decisions"] = decision_records
    
    return results


def print_results(results: Dict, model_name: str, ood_mode: str = None):
    """打印评测结果"""
    mode_str = f" ({ood_mode.upper()})" if ood_mode else ""
    
    print("\n" + "=" * 70)
    print(f"  Evaluation Results: {model_name}{mode_str}")
    print("=" * 70)
    print(f"  Success Rate (SR):        {results['sr']:.2%}  "
          f"({results['total_success']}/{results['total_attempts']})")
    print(f"  Mean Extracted Blocks:    {results['meb_mean']:.2f} ± {results['meb_std']:.2f}")
    print(f"  Max Extracted Blocks:     {results['meb_max']}")
    print(f"  Episodes:                 {results['num_episodes']}")
    print("=" * 70)
    
    # 分布统计
    meb_counts = {}
    for meb in results['all_meb']:
        meb_counts[meb] = meb_counts.get(meb, 0) + 1
    
    print("\n  MEB Distribution:")
    for meb in sorted(meb_counts.keys()):
        count = meb_counts[meb]
        pct = count / results['num_episodes'] * 100
        bar = "█" * int(pct / 2)
        print(f"    {meb:2d} blocks: {count:3d} episodes ({pct:5.1f}%) {bar}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Unified Evaluation Harness for Jenga")
    
    # 通用参数
    p.add_argument("--model", type=str, required=True,
                   choices=["heuristic", "pointcloud_heuristic", "vlm", "rl"],
                   help="模型类型")
    p.add_argument("--num_episodes", type=int, default=100,
                   help="评测 episode 数量")
    p.add_argument("--max_steps", type=int, default=15,
                   help="每 episode 最大步数")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子")
    p.add_argument("--ood_mode", type=str, default=None,
                   choices=[None, "moderate", "hard", "sparse", "ultra_high"],
                   help="OOD 评测模式: moderate (中等难度), hard (高难度), "
                        "sparse (稀疏支撑, 等同于 hard), ultra_high (超高塔, 极高难度)")
    p.add_argument("--output", type=str, default=None,
                   help="结果保存路径 (JSON)")
    p.add_argument("--record_decisions", action="store_true",
                   help="记录每步的决策（用于可视化）")
    
    # Heuristic 参数
    p.add_argument("--tower_center_x", type=float, default=0.0)
    p.add_argument("--tower_center_y", type=float, default=0.0)
    
    # VLM 参数
    p.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--base_url", type=str, 
                   default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--vlm_model", type=str, default="gpt-4o")
    p.add_argument("--camera", type=str, default="surround_0",
                   help="相机 UID, 或 'multi' 使用全部 4 个环绕相机")
    
    # RL 参数
    p.add_argument("--vision_checkpoint", type=str, default=None,
                   help="Vision 网络 checkpoint 路径 (vision_final.pt)")
    p.add_argument("--rl_checkpoint", type=str, default=None,
                   help="RL 网络 checkpoint 路径 (rl_final.pt)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="完整 checkpoint 路径 (包含 vision_net 和 rl_net)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--n_pts", type=int, default=256)
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # 创建 Agent
    if args.model == "heuristic":
        agent = HeuristicAgent(tower_center=(args.tower_center_x, args.tower_center_y))
        model_name = "Heuristic Baseline (Oracle - 完美感知)"
    
    elif args.model == "pointcloud_heuristic":
        agent = PointCloudHeuristicAgent(
            tower_center=(args.tower_center_x, args.tower_center_y),
            n_pts=args.n_pts if hasattr(args, 'n_pts') else 256,
        )
        model_name = "Point Cloud Heuristic (公平对比 - 点云输入)"
    
    elif args.model == "vlm":
        assert args.api_key, "VLM 模式需要 --api_key"
        camera_uids = (
            [f"surround_{i}" for i in range(4)]
            if args.camera == "multi"
            else [args.camera]
        )
        agent = VLMAgent(args.api_key, args.base_url, args.vlm_model, camera_uids)
        model_name = f"VLM Baseline ({args.vlm_model})"
    
    elif args.model == "rl":
        # 支持两种加载方式:
        # 1. 分别指定 vision 和 rl checkpoint
        # 2. 使用完整 checkpoint (包含 vision_net 和 rl_net)
        if args.vision_checkpoint and args.rl_checkpoint:
            vision_ckpt = args.vision_checkpoint
            rl_ckpt = args.rl_checkpoint
        elif args.checkpoint:
            # 假设 checkpoint 是一个目录或完整 checkpoint 文件
            if os.path.isdir(args.checkpoint):
                vision_ckpt = os.path.join(args.checkpoint, "vision_final.pt")
                rl_ckpt = os.path.join(args.checkpoint, "rl_final.pt")
            else:
                # 完整 checkpoint 文件，需要解包
                ckpt = torch.load(args.checkpoint, map_location="cpu")
                if "vision_net" in ckpt and "rl_net" in ckpt:
                    # 临时保存到文件
                    import tempfile
                    tmpdir = tempfile.mkdtemp()
                    vision_ckpt = os.path.join(tmpdir, "vision.pt")
                    rl_ckpt = os.path.join(tmpdir, "rl.pt")
                    torch.save(ckpt["vision_net"], vision_ckpt)
                    torch.save(ckpt["rl_net"], rl_ckpt)
                else:
                    raise ValueError("checkpoint 文件格式不正确，需要包含 'vision_net' 和 'rl_net'")
        else:
            raise ValueError("RL 模式需要 --vision_checkpoint + --rl_checkpoint 或 --checkpoint")
        
        agent = RLAgent(
            vision_checkpoint=vision_ckpt,
            rl_checkpoint=rl_ckpt,
            device=args.device,
            feat_dim=args.feat_dim,
            gnn_layers=args.gnn_layers,
            n_pts=args.n_pts,
        )
        model_name = "RL Model (VP3E + PPO)"
    
    else:
        raise ValueError(f"未知模型类型: {args.model}")
    
    # 运行评测
    print(f"\n开始评测: {model_name}")
    print(f"Episodes: {args.num_episodes}, Seed: {args.seed}")
    if args.ood_mode:
        print(f"OOD Mode: {args.ood_mode}")
    
    t0 = time.time()
    results = evaluate_agent(
        agent=agent,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        ood_mode=args.ood_mode,
        verbose=True,
        record_decisions=args.record_decisions,
    )
    elapsed = time.time() - t0
    
    # 打印结果
    print_results(results, model_name, args.ood_mode)
    print(f"  Total Time: {elapsed:.1f}s  ({elapsed / args.num_episodes:.1f}s/episode)")
    
    # 保存结果
    if args.output:
        # 转换 numpy 类型为 Python 原生类型
        def convert_to_native(obj):
            """递归转换 numpy 类型为 Python 原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        results_to_save = {
            "model": args.model,
            "model_name": model_name,
            "ood_mode": args.ood_mode,
            "seed": args.seed,
            "num_episodes": args.num_episodes,
            "sr": float(results["sr"]),
            "meb_mean": float(results["meb_mean"]),
            "meb_std": float(results["meb_std"]),
            "meb_max": int(results["meb_max"]),
            "total_success": int(results["total_success"]),
            "total_attempts": int(results["total_attempts"]),
            "all_meb": [int(x) for x in results["all_meb"]],
            "elapsed_time": float(elapsed),
        }
        
        # 如果有决策记录，也要转换
        if "decisions" in results:
            results_to_save["decisions"] = convert_to_native(results["decisions"])
        
        with open(args.output, "w") as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
