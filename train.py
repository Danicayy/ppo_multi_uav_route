"""
训练脚本 - 多无人机PPO路径规划
"""

import os
import sys
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from config import Config
from env.multi_uav_env import MultiUAVEnv
from model.ppo_agent import PPOAgent


# 定义一个Logger类来同时输出到控制台和文件
class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8", buffering=1)  # 行缓冲

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        try:
            self.log.write(message)
            self.log.flush()
        except (OSError, IOError):
            pass  # 忽略写入错误

    def flush(self):
        self.terminal.flush()
        try:
            self.log.flush()
        except (OSError, IOError):
            pass  # 忽略刷新错误
    
    def close(self):
        try:
            self.log.close()
        except (OSError, IOError):
            pass

def find_latest_checkpoint(model_path, num_uavs):
    """查找最新的检查点"""
    if not os.path.exists(model_path):
        return None, 0
    
    # 查找所有检查点文件
    checkpoints = []
    for i in range(num_uavs):
        pattern = f'ppo_uav_{i}_episode_'
        files = [f for f in os.listdir(model_path) if f.startswith(pattern) and f.endswith('.pth')]
        for f in files:
            try:
                episode_num = int(f.replace(pattern, '').replace('.pth', ''))
                checkpoints.append(episode_num)
            except ValueError:
                continue
    
    if checkpoints:
        latest_episode = max(checkpoints)
        return latest_episode, latest_episode
    return None, 0


def train(resume_from=None):
    """
    训练函数
    Args:
        resume_from: 从哪个episode恢复训练。如果为'latest'，自动查找最新检查点。
                    如果为具体数字，从该episode恢复。如果为None，从头开始训练。
    """
    # 将所有输出重定向到Logger
    logger = Logger("log.txt")
    sys.stdout = logger
    sys.stderr = logger

    config = Config()
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    env = MultiUAVEnv(config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agents = [PPOAgent(obs_dim, action_dim, config) for _ in range(config.NUM_UAVS)]
    
    # 检查是否从检查点恢复
    start_episode = 0
    if resume_from == 'latest':
        checkpoint_episode, start_episode = find_latest_checkpoint(config.MODEL_SAVE_PATH, config.NUM_UAVS)
        if checkpoint_episode:
            print(f"找到检查点: Episode {checkpoint_episode}")
            for i, agent in enumerate(agents):
                checkpoint_path = os.path.join(
                    config.MODEL_SAVE_PATH, f'ppo_uav_{i}_episode_{checkpoint_episode}.pth'
                )
                agent.load(checkpoint_path)
                print(f"  已加载 Agent {i}: {checkpoint_path}")
            print(f"从 Episode {start_episode + 1} 继续训练")
        else:
            print("未找到检查点，从头开始训练")
    elif resume_from is not None and isinstance(resume_from, int):
        start_episode = resume_from
        print(f"从指定的 Episode {resume_from} 恢复训练")
        for i, agent in enumerate(agents):
            checkpoint_path = os.path.join(
                config.MODEL_SAVE_PATH, f'ppo_uav_{i}_episode_{resume_from}.pth'
            )
            if os.path.exists(checkpoint_path):
                agent.load(checkpoint_path)
                print(f"  已加载 Agent {i}: {checkpoint_path}")
            else:
                print(f"  警告: 未找到 {checkpoint_path}，Agent {i} 将使用随机初始化")
    
    # 创建唯一的日志目录避免冲突
    import time as time_module
    log_dir = os.path.join(config.LOG_DIR, f"run_{int(time_module.time())}")
    writer = SummaryWriter(log_dir)
    episode_rewards = []
    episode_success_rates = []
    
    print("开始训练...")
    print(f"环境大小: {config.ENV_SIZE}x{config.ENV_SIZE}")
    print(f"无人机数量: {config.NUM_UAVS}")
    print(f"障碍物数量: {config.NUM_OBSTACLES}")
    print(f"训练Episodes: {config.NUM_EPISODES}")
    if start_episode > 0:
        print(f"从 Episode {start_episode + 1} 恢复")
    print("-" * 50)
    
    start_time = time.time()
    global_step = start_episode * config.MAX_STEPS  # 恢复全局步数
    
    for episode in range(start_episode, config.NUM_EPISODES):
        obs, _ = env.reset()
        episode_reward = np.zeros(config.NUM_UAVS)
        done = False
        step = 0
        
        while not done:
            # 每个UAV选择动作
            actions = []
            log_probs = []
            values = []
            
            for i, agent in enumerate(agents):
                action, log_prob, value = agent.select_action(obs[i])
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
            
            actions = np.array(actions)

            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            for i, agent in enumerate(agents):
                agent.store_transition(
                    obs[i], actions[i], rewards[i], 
                    values[i], log_probs[i], done
                )
            
            episode_reward += rewards
            obs = next_obs
            step += 1
            global_step += 1

            if global_step % config.UPDATE_INTERVAL == 0:
                for i, agent in enumerate(agents):
                    loss = agent.update(next_obs[i])
                    writer.add_scalar(f'Loss/Agent_{i}', loss, global_step)

        avg_reward = np.mean(episode_reward)
        success_rate = info['num_reached'] / config.NUM_UAVS
        
        episode_rewards.append(avg_reward)
        episode_success_rates.append(success_rate)

        writer.add_scalar('Reward/Average', avg_reward, episode)
        writer.add_scalar('Reward/Total', np.sum(episode_reward), episode)
        writer.add_scalar('Success/Rate', success_rate, episode)
        writer.add_scalar('Success/Count', info['num_reached'], episode)
        writer.add_scalar('Episode/Steps', step, episode)

        # 每个episode都显示简短进度
        if True:  # 每轮都输出
            elapsed_time = time.time() - start_time
            episodes_done = episode - start_episode + 1
            avg_time_per_episode = elapsed_time / episodes_done if episodes_done > 0 else 0
            remaining_episodes = config.NUM_EPISODES - (episode + 1)
            eta_seconds = avg_time_per_episode * remaining_episodes
            
            # 格式化时间
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_mins = int((elapsed_time % 3600) // 60)
            elapsed_secs = int(elapsed_time % 60)
            
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)
            
            print(f"Episode {episode + 1}/{config.NUM_EPISODES} | "
                  f"奖励: {avg_reward:.2f} | "
                  f"成功: {info['num_reached']}/{config.NUM_UAVS} | "
                  f"步数: {step} | "
                  f"用时: {elapsed_hours:02d}:{elapsed_mins:02d}:{elapsed_secs:02d} | "
                  f"预计剩余: {eta_hours:02d}h{eta_mins:02d}m")

        if (episode + 1) % 100 == 0:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            avg_success_100 = np.mean(episode_success_rates[-100:])
            
            print(f"Episode {episode + 1}/{config.NUM_EPISODES}")
            print(f"  平均奖励 (最近100): {avg_reward_100:.2f}")
            print(f"  成功率 (最近100): {avg_success_100:.2%}")
            print(f"  当前Episode: 奖励={avg_reward:.2f}, 成功={info['num_reached']}/{config.NUM_UAVS}")
            print("-" * 50)

        if (episode + 1) % 500 == 0:
            for i, agent in enumerate(agents):
                model_path = os.path.join(
                    config.MODEL_SAVE_PATH, f'ppo_uav_{i}_episode_{episode + 1}.pth'
                )
                agent.save(model_path)
            print(f"模型已保存到 {config.MODEL_SAVE_PATH}")
    
    # 保存最终模型
    for i, agent in enumerate(agents):
        model_path = os.path.join(config.MODEL_SAVE_PATH, f'ppo_uav_{i}_final.pth')
        agent.save(model_path)
    
    print("训练完成!")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_success_rates)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOG_DIR, 'training_curves.png'))
    plt.show()
    
    writer.close()


if __name__ == '__main__':
    train()