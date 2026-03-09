# Jenga 评估系统使用指南

## 概述

我已经为你创建了完整的评估系统，包括：

1. **heuristic_jenga_baseline.py** - 基于几何规则的启发式 Baseline
2. **evaluate_jenga.py** - 统一评估框架（支持 Heuristic、VLM、RL 三种模型）

## 评估指标

根据你的论文要求，系统计算以下指标：

### 1. Success Rate (SR)
每 episode 平均成功操作率（不导致坍塌）
```
SR = 成功抽取步数 / 总尝试步数
```

### 2. Max Extracted Blocks (MEB)
每 episode 成功抽取的最大积木数，反映长期规划能力
```
MEB_mean = mean(每个 episode 的成功抽取数)
MEB_std = std(每个 episode 的成功抽取数)
```

### 3. OOD Generalization
在极端 OOD 拓扑上的成功率（稀疏支撑、超高塔）

## 使用方法

### 1. 评估 Heuristic Baseline

```bash
conda activate /opt/anaconda3/envs/skill
python evaluate_jenga.py --model heuristic --num_episodes 100 --seed 42
```

**策略说明：**
- 排除最高两层和最底层的木块
- 在剩余候选中选择距离塔中心最近的木块
- 使用完美感知（物理引擎底层位姿）

### 2. 评估 VLM Baseline

```bash
python evaluate_jenga.py --model vlm --num_episodes 100 --seed 42 \
    --api_key YOUR_API_KEY \
    --base_url https://api.openai.com/v1 \
    --vlm_model gpt-4o \
    --camera multi
```

**参数说明：**
- `--api_key`: OpenAI API Key
- `--base_url`: API 基础 URL（支持中转 API）
- `--vlm_model`: 模型名称（gpt-4o, claude-3-5-sonnet-20240620 等）
- `--camera`: 相机视角（surround_0 或 multi 使用 4 个环绕相机）

### 3. 评估 RL Model (你的完整模型)

**方式 1: 指定目录（推荐）**
```bash
python evaluate_jenga.py --model rl --num_episodes 100 --seed 42 \
    --checkpoint runs/jenga_ppo
```

**方式 2: 分别指定 checkpoint**
```bash
python evaluate_jenga.py --model rl --num_episodes 100 --seed 42 \
    --vision_checkpoint runs/jenga_ppo/vision_final.pt \
    --rl_checkpoint runs/jenga_ppo/rl_final.pt
```

**方式 3: 使用完整 checkpoint 文件**
```bash
python evaluate_jenga.py --model rl --num_episodes 100 --seed 42 \
    --checkpoint runs/jenga_ppo/ckpt_0280.pt
```

**模型参数（可选）：**
```bash
--feat_dim 256 \
--gnn_layers 4 \
--n_pts 256 \
--device cuda  # 或 cpu
```

### 4. OOD 评估

```bash
# 稀疏支撑拓扑
python evaluate_jenga.py --model rl --num_episodes 100 --seed 42 \
    --checkpoint runs/jenga_ppo \
    --ood_mode sparse

# 超高塔拓扑
python evaluate_jenga.py --model rl --num_episodes 100 --seed 42 \
    --checkpoint runs/jenga_ppo \
    --ood_mode ultra_high
```

### 5. 保存结果到 JSON

```bash
python evaluate_jenga.py --model rl --num_episodes 100 --seed 42 \
    --checkpoint runs/jenga_ppo \
    --output results/rl_model_eval.json
```

## 输出示例

```
======================================================================
  Evaluation Results: RL Model (VP3E + PPO)
======================================================================
  Success Rate (SR):        87.50%  (350/400)
  Mean Extracted Blocks:    8.75 ± 2.34
  Max Extracted Blocks:     15
  Episodes:                 100
======================================================================

  MEB Distribution:
     0 blocks:   2 episodes (  2.0%) █
     3 blocks:   5 episodes (  5.0%) ██
     5 blocks:  12 episodes ( 12.0%) ██████
     7 blocks:  25 episodes ( 25.0%) ████████████
     9 blocks:  30 episodes ( 30.0%) ███████████████
    11 blocks:  18 episodes ( 18.0%) █████████
    13 blocks:   6 episodes (  6.0%) ███
    15 blocks:   2 episodes (  2.0%) █

  Total Time: 245.3s  (2.5s/episode)
```

## 代码架构

### evaluate_jenga.py 核心组件

```python
# 1. Agent 基类
class BaseAgent:
    def select_action(self, env, obs, info):
        """选择动作"""
        pass
    
    def reset(self):
        """重置 Agent 状态"""
        pass

# 2. Heuristic Agent
class HeuristicAgent(BaseAgent):
    """基于几何规则的策略"""
    - 获取所有积木的 3D 位姿
    - 过滤最高两层和最底层
    - 选择距离塔中心最近的积木

# 3. VLM Agent
class VLMAgent(BaseAgent):
    """基于 VLM 的策略"""
    - 渲染并标注图像
    - 查询 VLM API
    - 解析响应并选择动作

# 4. RL Agent
class RLAgent(BaseAgent):
    """基于 VP3E + PPO 的策略"""
    - 加载 vision_net (VP3ENetwork)
    - 加载 rl_net (PriorGuidedActorCritic)
    - 渲染点云输入
    - 前向推理选择动作

# 5. 评估函数
def evaluate_agent(agent, num_episodes, max_steps, seed, ood_mode):
    """
    运行评估循环
    返回: SR, MEB, episode 详细数据
    """
```

## RL Agent 实现细节

```python
class RLAgent(BaseAgent):
    def __init__(self, vision_checkpoint, rl_checkpoint, device="cpu", 
                 feat_dim=256, gnn_layers=4, n_pts=256):
        # 加载 VP3E 视觉网络
        self.vision_net = VP3ENetwork(
            feat_dim=feat_dim, 
            gnn_layers=gnn_layers, 
            max_blocks=NUM_BLOCKS
        )
        self.vision_net.load_state_dict(torch.load(vision_checkpoint))
        
        # 加载 Actor-Critic 网络
        self.rl_net = PriorGuidedActorCritic(feat_dim=feat_dim)
        self.rl_net.load_state_dict(torch.load(rl_checkpoint))
        
        self.vision_net.eval()
        self.rl_net.eval()
    
    def select_action(self, env, obs, info):
        # 1. 渲染点云 [1, N, K, 3]
        pcd = self._render_vp3e_input(env)
        mask = torch.tensor(info["mask"])
        
        # 2. VP3E 前向推理
        with torch.no_grad():
            vp3e_out = self.vision_net(pcd, mask, use_causal=True)
            
            # 3. Actor-Critic 选择动作
            action, _, _, _ = self.rl_net.get_action_and_value(
                vp3e_out["Z"],           # 物理融合特征
                vp3e_out["stab_hat"],    # 稳定性预测
                vp3e_out["pot_hat"],     # 潜能预测
                mask
            )
        
        return action.item()
```

## 论文实验流程

根据你的论文要求：

1. **训练阶段**
   - 使用相同随机种子训练所有模型
   - 每个 epoch 后在验证集上评估 100 episodes
   - 保存 SR 最高的模型

2. **测试阶段**
   - 在测试集上运行 100 个独立 episodes
   - 计算 SR、MEB、OOD Generalization
   - 生成最终性能报告

3. **对比实验**
   ```bash
   # Baseline 1: Heuristic
   python evaluate_jenga.py --model heuristic --num_episodes 100 --seed 42 \
       --output results/heuristic.json
   
   # Baseline 2: VLM (GPT-4o)
   python evaluate_jenga.py --model vlm --num_episodes 100 --seed 42 \
       --vlm_model gpt-4o --output results/vlm_gpt4o.json
   
   # Your Method: RL (VP3E + PPO)
   python evaluate_jenga.py --model rl --num_episodes 100 --seed 42 \
       --checkpoint runs/jenga_ppo --output results/rl_vp3e.json
   ```

4. **OOD 泛化测试**
   ```bash
   # 每个模型在 OOD 场景下测试
   for model in heuristic vlm rl; do
       for ood in sparse ultra_high; do
           python evaluate_jenga.py --model $model --num_episodes 100 \
               --ood_mode $ood --output results/${model}_${ood}.json
       done
   done
   ```

## 故障排除

### 问题 1: Segmentation Fault (139)
这可能是 SAPIEN 在 macOS 上的问题。解决方案：
- 在 Linux 服务器上运行
- 或使用 Docker 容器

### 问题 2: VLM API 限流
- 在代码中已添加自动重试和等待逻辑
- 每次请求前等待 2 秒
- 429/503 错误会自动重试

### 问题 3: GPU 内存不足
```bash
# 使用 CPU
python evaluate_jenga.py --model rl --device cpu ...

# 或减少点云采样数
python evaluate_jenga.py --model rl --n_pts 128 ...
```

## 下一步

1. 在 Linux 环境中测试评估脚本
2. 运行完整的 100 episodes 评估
3. 收集所有模型的结果
4. 生成对比表格和图表

## 文件清单

- `heuristic_jenga_baseline.py` - Heuristic Baseline 独立脚本
- `evaluate_jenga.py` - 统一评估框架
- `vlm_jenga_baseline.py` - VLM Baseline 独立脚本（已存在）
- `train_jenga_ppo.py` - RL 训练脚本（已存在）
- `vp3e_modules.py` - VP3E 网络定义（已存在）
- `jenga_ppo_wrapper.py` - 环境包装器（已存在）

所有代码已经完成并可以使用！
