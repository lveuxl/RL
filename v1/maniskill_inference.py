import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from stack_picking_maniskill_env import StackPickingManiSkillEnv
from maniskill_config import ENV_CONFIG, PPO_CONFIG, LOGGING_CONFIG
from maniskill_utils import (
    setup_directories, 
    load_training_results, 
    calculate_training_statistics,
    print_training_summary,
    save_episode_data
)

class ManiSkillInference:
    """ManiSkill推理类 - 参考PyBullet项目的推理结构"""
    
    def __init__(self, model_path: str, num_episodes: int = 10, record_video: bool = True):
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.record_video = record_video
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置目录
        setup_directories()
        
        # 初始化环境
        self.env = self._create_environment()
        
        # 加载模型
        self.model = self._load_model()
        
        # 结果存储
        self.evaluation_results = {
            "episodes": [],
            "total_rewards": [],
            "success_counts": [],
            "success_rates": [],
            "episode_lengths": [],
            "remaining_objects": [],
            "grasp_attempts": [],
            "total_distances": []
        }
    
    def _create_environment(self):
        """创建推理环境"""
        def make_env():
            env = StackPickingManiSkillEnv(
                obj_num=ENV_CONFIG["obj_num"],
                max_objects=ENV_CONFIG["max_objects"],
                circle_center=ENV_CONFIG["circle_center"],
                circle_radius=ENV_CONFIG["circle_radius"],
                category_map=ENV_CONFIG["category_map"],
                object_properties=ENV_CONFIG["object_properties"],
                reward_config=ENV_CONFIG["reward_config"],
                obs_mode="rgbd",
                render_mode="rgb_array" if self.record_video else None,
                sim_freq=100,
                control_freq=20
            )
            return env
        
        # 创建向量化环境
        env = DummyVecEnv([make_env])
        
        # 如果需要录制视频
        if self.record_video:
            video_dir = "maniskill_outputs/videos"
            os.makedirs(video_dir, exist_ok=True)
            env = VecVideoRecorder(
                env,
                video_folder=video_dir,
                record_video_trigger=lambda x: True,  # 录制所有episode
                video_length=ENV_CONFIG["training_config"]["max_episode_steps"],
                name_prefix="maniskill_inference"
            )
        
        return env
    
    def _load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        try:
            print(f"正在加载模型: {self.model_path}")
            model = PPO.load(self.model_path, device=self.device)
            print(f"模型加载成功，设备: {self.device}")
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def run_inference(self):
        """运行推理评估"""
        print(f"开始推理评估，共 {self.num_episodes} episodes")
        print(f"设备: {self.device}")
        print(f"录制视频: {self.record_video}")
        print("-" * 60)
        
        for episode in range(self.num_episodes):
            episode_start_time = time.time()
            
            # 重置环境
            obs = self.env.reset()
            done = False
            step_count = 0
            total_reward = 0.0
            grasp_attempts = 0
            successful_grasps = 0
            
            episode_data = {
                "episode": episode + 1,
                "steps": [],
                "observations": [],
                "actions": [],
                "rewards": [],
                "infos": []
            }
            
            print(f"Episode {episode + 1}/{self.num_episodes} 开始...")
            
            while not done and step_count < ENV_CONFIG["training_config"]["max_episode_steps"]:
                # 模型预测动作
                action, _states = self.model.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, done, info = self.env.step(action)
                
                # 记录数据
                total_reward += reward[0]
                step_count += 1
                
                # 统计抓取尝试
                if info[0].get("grasp_attempted", False):
                    grasp_attempts += 1
                    if info[0].get("grasp_successful", False):
                        successful_grasps += 1
                
                # 保存step数据
                episode_data["steps"].append(step_count)
                episode_data["observations"].append(obs[0] if hasattr(obs[0], 'tolist') else obs[0])
                episode_data["actions"].append(action[0].tolist() if hasattr(action[0], 'tolist') else action[0])
                episode_data["rewards"].append(reward[0])
                episode_data["infos"].append(info[0])
                
                # 打印进度 (每50步)
                if step_count % 50 == 0:
                    print(f"  Step {step_count}, Reward: {reward[0]:.3f}, Total: {total_reward:.3f}")
            
            # Episode结束统计
            episode_time = time.time() - episode_start_time
            remaining_objects = info[0].get("remaining_objects", 0)
            success_rate = successful_grasps / max(grasp_attempts, 1)
            total_distance = info[0].get("total_distance", 0.0)
            
            # 保存episode结果
            self.evaluation_results["episodes"].append(episode + 1)
            self.evaluation_results["total_rewards"].append(total_reward)
            self.evaluation_results["success_counts"].append(successful_grasps)
            self.evaluation_results["success_rates"].append(success_rate)
            self.evaluation_results["episode_lengths"].append(step_count)
            self.evaluation_results["remaining_objects"].append(remaining_objects)
            self.evaluation_results["grasp_attempts"].append(grasp_attempts)
            self.evaluation_results["total_distances"].append(total_distance)
            
            # 保存详细episode数据
            episode_data.update({
                "total_reward": total_reward,
                "success_count": successful_grasps,
                "success_rate": success_rate,
                "episode_length": step_count,
                "remaining_objects": remaining_objects,
                "grasp_attempts": grasp_attempts,
                "total_distance": total_distance,
                "episode_time": episode_time
            })
            
            save_episode_data(episode_data, run_number=0, episode_number=episode + 1)
            
            # 打印episode结果
            print(f"Episode {episode + 1} 完成:")
            print(f"  总奖励: {total_reward:.3f}")
            print(f"  步数: {step_count}")
            print(f"  成功抓取: {successful_grasps}/{grasp_attempts}")
            print(f"  成功率: {success_rate:.2%}")
            print(f"  剩余物体: {remaining_objects}")
            print(f"  总距离: {total_distance:.3f}")
            print(f"  用时: {episode_time:.2f}秒")
            print("-" * 40)
        
        # 关闭环境
        self.env.close()
        
        # 计算和打印总体统计
        self._print_evaluation_summary()
        
        # 保存评估结果
        self._save_evaluation_results()
        
        return self.evaluation_results
    
    def _print_evaluation_summary(self):
        """打印评估总结"""
        results = self.evaluation_results
        
        print(f"\n{'='*60}")
        print(f"ManiSkill推理评估总结")
        print(f"{'='*60}")
        print(f"模型路径: {self.model_path}")
        print(f"总episodes: {len(results['episodes'])}")
        print(f"平均总奖励: {np.mean(results['total_rewards']):.3f} ± {np.std(results['total_rewards']):.3f}")
        print(f"平均成功抓取数: {np.mean(results['success_counts']):.2f} ± {np.std(results['success_counts']):.2f}")
        print(f"平均成功率: {np.mean(results['success_rates']):.2%} ± {np.std(results['success_rates']):.2%}")
        print(f"平均episode长度: {np.mean(results['episode_lengths']):.1f} ± {np.std(results['episode_lengths']):.1f}")
        print(f"平均剩余物体数: {np.mean(results['remaining_objects']):.2f} ± {np.std(results['remaining_objects']):.2f}")
        print(f"平均抓取尝试数: {np.mean(results['grasp_attempts']):.2f} ± {np.std(results['grasp_attempts']):.2f}")
        print(f"平均总距离: {np.mean(results['total_distances']):.3f} ± {np.std(results['total_distances']):.3f}")
        print(f"最高成功抓取数: {max(results['success_counts'])}")
        print(f"最高成功率: {max(results['success_rates']):.2%}")
        print(f"完美episodes (剩余物体=0): {sum(1 for x in results['remaining_objects'] if x == 0)}")
        print(f"高成功率episodes (>50%): {sum(1 for x in results['success_rates'] if x > 0.5)}")
        print(f"{'='*60}\n")
    
    def _save_evaluation_results(self):
        """保存评估结果到CSV文件"""
        import csv
        
        # 创建结果文件
        results_file = "maniskill_outputs/inference_results.csv"
        
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow([
                'Episode', '总奖励', '成功抓取数', '成功率', 'Episode长度',
                '剩余物体数', '抓取尝试数', '总距离'
            ])
            
            # 写入数据
            for i in range(len(self.evaluation_results['episodes'])):
                writer.writerow([
                    self.evaluation_results['episodes'][i],
                    f"{self.evaluation_results['total_rewards'][i]:.3f}",
                    self.evaluation_results['success_counts'][i],
                    f"{self.evaluation_results['success_rates'][i]:.2%}",
                    self.evaluation_results['episode_lengths'][i],
                    self.evaluation_results['remaining_objects'][i],
                    self.evaluation_results['grasp_attempts'][i],
                    f"{self.evaluation_results['total_distances'][i]:.3f}"
                ])
        
        print(f"评估结果已保存到: {results_file}")
        
        # 同时保存统计摘要
        summary_file = "maniskill_outputs/inference_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"ManiSkill推理评估统计摘要\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总episodes: {len(self.evaluation_results['episodes'])}\n")
            f.write(f"平均总奖励: {np.mean(self.evaluation_results['total_rewards']):.3f}\n")
            f.write(f"平均成功率: {np.mean(self.evaluation_results['success_rates']):.2%}\n")
            f.write(f"平均剩余物体数: {np.mean(self.evaluation_results['remaining_objects']):.2f}\n")
            f.write(f"完美episodes: {sum(1 for x in self.evaluation_results['remaining_objects'] if x == 0)}\n")
        
        print(f"统计摘要已保存到: {summary_file}")

def main():
    """主函数 - 参考PyBullet项目的命令行接口"""
    parser = argparse.ArgumentParser(description="ManiSkill模型推理评估")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的PPO模型路径 (.zip文件)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="评估的episode数量")
    parser.add_argument("--record_video", action="store_true",
                        help="是否录制评估视频")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 验证模型文件
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 创建推理器
    inference = ManiSkillInference(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        record_video=args.record_video
    )
    
    # 运行推理
    try:
        results = inference.run_inference()
        print("推理评估完成!")
        
        if args.record_video:
            print("视频文件保存在: maniskill_outputs/videos/")
        
    except KeyboardInterrupt:
        print("\n推理评估被用户中断")
    except Exception as e:
        print(f"推理评估出错: {e}")
        import traceback
        traceback.print_exc()

def evaluate_multiple_models(model_dir: str, num_episodes: int = 5):
    """评估目录中的多个模型 - 批量评估功能"""
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return
    
    # 查找所有模型文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    model_files.sort()
    
    if not model_files:
        print(f"在目录 {model_dir} 中未找到模型文件")
        return
    
    print(f"找到 {len(model_files)} 个模型文件，开始批量评估...")
    
    all_results = {}
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        print(f"\n正在评估模型: {model_file}")
        
        try:
            inference = ManiSkillInference(
                model_path=model_path,
                num_episodes=num_episodes,
                record_video=False  # 批量评估时不录制视频
            )
            
            results = inference.run_inference()
            all_results[model_file] = results
            
        except Exception as e:
            print(f"评估模型 {model_file} 时出错: {e}")
            continue
    
    # 保存批量评估结果对比
    if all_results:
        _save_batch_evaluation_results(all_results)

def _save_batch_evaluation_results(all_results: dict):
    """保存批量评估结果对比"""
    import csv
    
    comparison_file = "maniskill_outputs/batch_evaluation_comparison.csv"
    
    with open(comparison_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow([
            '模型文件', '平均总奖励', '平均成功率', '平均剩余物体数',
            '最高成功抓取数', '完美episodes数'
        ])
        
        # 写入每个模型的统计结果
        for model_file, results in all_results.items():
            writer.writerow([
                model_file,
                f"{np.mean(results['total_rewards']):.3f}",
                f"{np.mean(results['success_rates']):.2%}",
                f"{np.mean(results['remaining_objects']):.2f}",
                max(results['success_counts']),
                sum(1 for x in results['remaining_objects'] if x == 0)
            ])
    
    print(f"批量评估对比结果已保存到: {comparison_file}")

if __name__ == "__main__":
    main() 