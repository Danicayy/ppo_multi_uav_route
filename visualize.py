"""
可视化脚本 - 多无人机路径规划动画
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import torch

from config import Config
from env.multi_uav_env import MultiUAVEnv
from model.ppo_agent import PPOAgent


def visualize_episode(model_episode='final', save_video=False):
    """可视化一个完整的episode"""
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
        else:
            print(f"模型文件不存在 {model_path}")
            return
    
    # 运行一个episode并记录所有状态
    obs, _ = env.reset()
    done = False
    
    history = {
        'uav_positions': [],
        'obstacles': env.obstacles.copy(),
        'targets': env.targets.copy(),
        'reached': []
    }
    
    while not done:
        # 记录当前状态
        history['uav_positions'].append(env.uav_positions.copy())
        history['reached'].append(env.reached_target.copy())
        
        # 选择动作
        actions = []
        for i, agent in enumerate(agents):
            action, _, _ = agent.select_action(obs[i], deterministic=True)
            actions.append(action)
        
        actions = np.array(actions)
        
        # 环境步进
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
    
    # 最后一帧
    history['uav_positions'].append(env.uav_positions.copy())
    history['reached'].append(env.reached_target.copy())
    
    # 创建动画
    fig, ax = plt.subplots(figsize=(12, 12))
    
    colors = ['blue', 'orange', 'purple', 'brown', 'pink']
    
    def init():
        ax.clear()
        ax.set_xlim(0, config.ENV_SIZE)
        ax.set_ylim(0, config.ENV_SIZE)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        return []
    
    def update(frame):
        ax.clear()
        ax.set_xlim(0, config.ENV_SIZE)
        ax.set_ylim(0, config.ENV_SIZE)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制障碍物
        for obs_pos in history['obstacles']:
            circle = Circle(
                obs_pos, config.OBSTACLE_RADIUS, 
                color='red', alpha=0.5, label='Obstacle'
            )
            ax.add_patch(circle)
        
        # 绘制目标
        for i, target in enumerate(history['targets']):
            circle = Circle(
                target, config.TARGET_RADIUS, 
                color='green', alpha=0.3
            )
            ax.add_patch(circle)
            ax.plot(target[0], target[1], 'g*', markersize=20)
        
        # 绘制UAV轨迹
        for i in range(config.NUM_UAVS):
            # 轨迹
            if frame > 0:
                traj = np.array([history['uav_positions'][t][i] for t in range(frame + 1)])
                ax.plot(
                    traj[:, 0], traj[:, 1], 
                    color=colors[i % len(colors)], 
                    alpha=0.5, linewidth=2, label=f'UAV {i+1} trajectory'
                )
            

            pos = history['uav_positions'][frame][i]
            reached = history['reached'][frame][i]
            
            marker = 's' if reached else 'o'
            ax.plot(
                pos[0], pos[1], marker,
                color=colors[i % len(colors)], 
                markersize=15, label=f'UAV {i+1}'
            )
        
        ax.set_title(f'Multi-UAV Path Planning - Step {frame}/{len(history["uav_positions"])-1}', 
                    fontsize=16)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        return []
    
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(history['uav_positions']),
        interval=50, blit=True, repeat=True
    )

    if save_video:
        video_path = os.path.join(config.LOG_DIR, 'uav_path_planning.mp4')
        anim.save(video_path, writer='ffmpeg', fps=20)
        print(f"视频已保存到: {video_path}")
    
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='final', help='模型版本')
    parser.add_argument('--save', action='store_true', help='保存视频')
    
    args = parser.parse_args()
    
    visualize_episode(model_episode=args.model, save_video=args.save)