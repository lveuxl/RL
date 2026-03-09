"""
vlm_jenga_baseline.py — VLM Zero-shot Baseline for Jenga Block Extraction

在每一步:
  1. 渲染 Jenga 塔 RGB 图像 + 分割 Mask
  2. 在每块可见积木中心标注其数字 ID (白字黑底)
  3. 发送标注图给 VLM (GPT-4o / Claude / Gemini), 请求物理稳定性推理
  4. 解析返回 JSON 中的推荐抽取 ID, 执行 env.step
  5. 记录 MEB (成功抽取数) 和 SR (单步成功率)

指标:
  MEB (Mean Extracted Blocks): 平均每 episode 成功抽取数
  SR  (Step Success Rate):     成功抽取步 / 总尝试步

Usage:
    conda activate /opt/anaconda3/envs/skill
    python vlm_jenga_baseline.py \\
        --api_key sk-xxx \\
        --base_url https://www.chataiapi.com/v1 \\
        --model claude-3-5-sonnet-20240620 \\
        --camera multi --save_images
"""
import argparse
import base64
import io
import json
import os
import re
import time

import gymnasium as gym
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont

from jenga_tower import NUM_BLOCKS, NUM_LEVELS
from jenga_ppo_wrapper import JengaPPOWrapper

# ════════════════════════════════════════════════════
#  图像渲染与标注
# ════════════════════════════════════════════════════

def render_annotated_images(env, camera_uids):
    """
    从指定相机渲染 RGB + 分割, 在每块可见积木的像素中心标注其 ID。

    利用 SAPIEN segmentation 中的 per_scene_id 定位每块积木在图像中的位置,
    无需 3D→2D 投影, 直接从 mask 质心获得标注坐标。

    Returns:
        images:      list[np.ndarray]  每个相机的标注后 RGB (H, W, 3) uint8
        visible_ids: list[int]         所有视角中可见的积木 ID (去重排序)
    """
    uw = env.unwrapped
    scene = uw.scene

    # 统一刷新渲染 → 逐相机 capture → 逐相机 get_obs (与 render_point_cloud 一致)
    scene.update_render(update_sensors=True, update_human_render_cameras=False)
    sensors = {uid: scene.sensors[uid] for uid in camera_uids}
    for s in sensors.values():
        s.capture()

    sid_to_idx = {blk._objs[0].per_scene_id: i for i, blk in enumerate(uw.blocks)}

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font = ImageFont.load_default()

    images = []
    all_visible = set()

    for cam_uid in camera_uids:
        obs = sensors[cam_uid].get_obs(
            rgb=True, depth=False, segmentation=True, position=False,
        )
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
            bb = draw.textbbox((0, 0), text, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            draw.rectangle(
                [cx - tw // 2 - 2, cy - th // 2 - 1,
                 cx + tw // 2 + 2, cy + th // 2 + 1],
                fill="black",
            )
            draw.text((cx - tw // 2, cy - th // 2), text, fill="white", font=font)
            all_visible.add(idx)

        images.append(np.array(img))

    return images, sorted(all_visible)


def composite_grid(images):
    """将多张图拼成 2 列网格"""
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


def encode_image_base64(image_np: np.ndarray) -> str:
    """JPEG 编码 (比 PNG 小 5-10 倍, 中转 API 更友好)"""
    buf = io.BytesIO()
    Image.fromarray(image_np).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ════════════════════════════════════════════════════
#  GPT-4o 交互
# ════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a Jenga expert. "
    "Your task is to select the safest block to extract from the tower without causing it to collapse."
)


def build_user_prompt(valid_ids, removed_ids):
    valid_str = ", ".join(map(str, sorted(valid_ids)))
    removed_str = ", ".join(map(str, sorted(removed_ids))) if removed_ids else "None"
    
    return (
        f"Current tower state:\n"
        f"Available blocks: [{valid_str}]\n"
        f"Already removed: [{removed_str}]\n\n"
        f"Which block should I extract? Choose the safest one.\n\n"
        f"Respond with JSON:\n"
        f'```json\n{{"block_id": <ID>, "reasoning": "<brief explanation>"}}\n```'
    )


def parse_vlm_response(text, valid_ids):
    """从 GPT-4o 响应解析 block_id, 多级回退"""
    valid_set = set(valid_ids) if not isinstance(valid_ids, set) else valid_ids

    # 策略 1: JSON 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            bid = int(json.loads(m.group(1))["block_id"])
            if bid in valid_set:
                return bid
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # 策略 2: 裸 JSON 对象
    m = re.search(r'\{[^{}]*"block_id"\s*:\s*(\d+)[^{}]*\}', text)
    if m:
        bid = int(m.group(1))
        if bid in valid_set:
            return bid

    # 策略 3: 文本中第一个有效数字
    for m in re.finditer(r"\b(\d{1,2})\b", text):
        bid = int(m.group(1))
        if bid in valid_set:
            return bid

    return None


def query_vlm(api_key, base_url, image_b64, valid_ids, removed_ids,
              model="gpt-4o", retries=2):
    """
    用 requests 直接发 HTTP 请求 (兼容中转 API)。
    返回 (block_id | None, response_text)。
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": build_user_prompt(valid_ids, removed_ids)},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    },
                },
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
            return parse_vlm_response(text, set(valid_ids)), text
        except requests.exceptions.HTTPError as e:
            # 429/503 需要更长的等待时间
            if e.response.status_code in [429, 503] and attempt < retries:
                wait_time = min(10 * (2 ** attempt), 60)  # 最多等 60 秒
                print(f"    API 限流/不可用 ({e.response.status_code}), {wait_time}s 后重试...")
                time.sleep(wait_time)
                continue
            raise
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise


# ════════════════════════════════════════════════════
#  主循环
# ════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="VLM Zero-shot Jenga Baseline")
    p.add_argument("--api_key", type=str,
                   default=os.environ.get("OPENAI_API_KEY", ""),
                   help="API Key (也可通过 OPENAI_API_KEY 环境变量设置)")
    p.add_argument("--base_url", type=str,
                   default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                   help="API Base URL (也可通过 OPENAI_BASE_URL 环境变量设置)")
    p.add_argument("--max_episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=15,
                   help="每 episode 最大抽取步数")
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--camera", type=str, default="surround_0",
                   help="相机 UID, 或 'multi' 使用全部 4 个环绕相机")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_images", action="store_true",
                   help="保存每步标注图 (调试用)")
    p.add_argument("--image_dir", type=str, default="vlm_baseline_images")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert args.api_key, "请通过 --api_key 或 OPENAI_API_KEY 环境变量提供 API Key"

    base_env = gym.make(
        "JengaTower-v1", obs_mode="state", render_mode="rgb_array",
        num_envs=1, sim_backend="cpu",
    )
    env = JengaPPOWrapper(base_env, lambda_int=0.0)

    camera_uids = (
        [f"surround_{i}" for i in range(4)]
        if args.camera == "multi"
        else [args.camera]
    )

    if args.save_images:
        os.makedirs(args.image_dir, exist_ok=True)

    all_meb = []
    total_attempts, total_success = 0, 0

    print("=" * 60)
    print("  GPT-4o VLM Zero-shot Baseline for Jenga")
    print(f"  Model: {args.model}  |  View: {args.camera}")
    print(f"  Episodes: {args.max_episodes}  |  Max steps/ep: {args.max_steps}")
    print("=" * 60)

    t0 = time.time()

    for ep in range(args.max_episodes):
        obs, info = env.reset()
        removed_ids = []
        ep_success = 0
        ep_collapsed = False

        print(f"\n── Episode {ep + 1}/{args.max_episodes} ──")

        for step in range(args.max_steps):
            mask = info["mask"]
            valid_ids = [i for i in range(NUM_BLOCKS) if mask[i]]
            if not valid_ids:
                break

            # ① 渲染 & 标注
            imgs, visible = render_annotated_images(env, camera_uids)
            final_img = composite_grid(imgs) if len(imgs) > 1 else imgs[0]

            if args.save_images:
                Image.fromarray(final_img).save(
                    os.path.join(args.image_dir, f"ep{ep:02d}_step{step:02d}.png")
                )

            # ② 请求间隔控制（避免 429）
            if step > 0 or ep > 0:
                time.sleep(2)  # 每次请求前等待 2 秒

            # ③ 查询 VLM
            img_b64 = encode_image_base64(final_img)
            try:
                block_id, resp_text = query_vlm(
                    args.api_key, args.base_url,
                    img_b64, valid_ids, removed_ids, model=args.model,
                )
            except Exception as e:
                print(f"  Step {step + 1}: API 错误 ({e}), 回退到顶层")
                block_id, resp_text = max(valid_ids), ""

            # 解析失败回退: 选最高层积木 (Jenga 顶层最安全)
            if block_id is None:
                block_id = max(valid_ids)
                print(f"  Step {step + 1}: 解析失败, 回退 #{block_id}")

            # ④ 执行动作
            obs, reward, done, _, info = env.step(block_id)
            total_attempts += 1
            collapsed = info.get("collapsed", False)
            lv, pi = divmod(block_id, 3)

            if collapsed:
                ep_collapsed = True
                print(f"  Step {step + 1}: #{block_id} L{lv}[{pi}] -> 坍塌  r={reward:.1f}")
                break

            ep_success += 1
            total_success += 1
            removed_ids.append(block_id)
            print(f"  Step {step + 1}: #{block_id} L{lv}[{pi}] -> 成功  r={reward:.1f}")

            if done:
                break

        all_meb.append(ep_success)
        print(f"  结果: MEB={ep_success}, {'坍塌' if ep_collapsed else '完整'}")

    elapsed = time.time() - t0
    sr = total_success / max(total_attempts, 1)

    print("\n" + "=" * 60)
    print("  VLM Baseline 汇总")
    print(f"  MEB: {np.mean(all_meb):.2f} +/- {np.std(all_meb):.2f}")
    print(f"  SR:  {sr:.1%}  ({total_success}/{total_attempts})")
    print(f"  各 EP: {all_meb}")
    print(f"  耗时: {elapsed:.1f}s  ({elapsed / max(total_attempts, 1):.1f}s/step)")
    print("=" * 60)

    base_env.close()


if __name__ == "__main__":
    main()
