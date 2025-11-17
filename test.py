"""
测试脚本 - 多无人机PPO路径规划
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import Config
from env.multi_uav_env import MultiUAVEnv
from model.ppo_agent import PPOAgent


def test(num_episodes=10, render=True, model_episode='final'):
    """测试训练好的模型"""
    # 配置
    config = Config()
    
    # 创建环境
    env = MultiUAVEnv(config)
    
    # 创建智能体并加载模型
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agents = [PPOAgent(obs_dim, action_dim, config) for _ in range(config.NUM_UAVS)]
    
    # 加载模型
    for i, agent in enumerate(agents):
        model_path = os.path.join(
            config.MODEL_SAVE_PATH, f'ppo_uav_{i}_{model_episode}.pth'
        )
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"加载模型: {model_path}")
        else:
            print(f"模型文件不存在 {model_path}")
            return
    
    print(f"\n开始测试 {num_episodes} 个episodes...")
    print("-" * 50)
    
    # 测试统计
    total_rewards = []
    success_counts = []
    steps_list = []
    
    if render:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = np.zeros(config.NUM_UAVS)
        done = False
        step = 0
        
        while not done:
            actions = []
            for i, agent in enumerate(agents):
                action, _, _ = agent.select_action(obs[i], deterministic=True)
                actions.append(action)
            
            actions = np.array(actions)
            
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            episode_reward += rewards
            obs = next_obs
            step += 1
            
            if render:
                env.render()

        total_rewards.append(np.sum(episode_reward))
        success_counts.append(info['num_reached'])
        steps_list.append(step)
        
        print(f"Episode {episode + 1}:")
        print(f"  总奖励: {np.sum(episode_reward):.2f}")
        print(f"  平均奖励: {np.mean(episode_reward):.2f}")
        print(f"  成功到达: {info['num_reached']}/{config.NUM_UAVS}")
        print(f"  步数: {step}")
        print("-" * 50)
    
    if render:
        plt.ioff()
        plt.show()

    print("\n测试总结:")
    print(f"  平均总奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  平均成功率: {np.mean(success_counts) / config.NUM_UAVS:.2%}")
    print(f"  平均步数: {np.mean(steps_list):.2f}")
    print(f"  最佳总奖励: {np.max(total_rewards):.2f}")
    print(f"  最差总奖励: {np.min(total_rewards):.2f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10, help='测试episodes数')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    parser.add_argument('--model', type=str, default='final', help='模型版本')
    
    args = parser.parse_args()
    
    test(num_episodes=args.episodes, render=args.render, model_episode=args.model)