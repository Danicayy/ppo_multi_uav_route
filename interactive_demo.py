"""
交互式演示程序
允许用户手动标定起点、终点和障碍区，然后使用训练好的模型运行
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from config import Config
from model.ppo_agent import PPOAgent
from env.multi_uav_env import MultiUAVEnv

# 设置中文字体以避免警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class InteractiveDemo:
    def __init__(self, model_path):
        self.config = Config()
        self.env = MultiUAVEnv(self.config)
        self.model_path = model_path
        
        # 初始化存储用户输入
        self.start_positions = []
        self.target_positions = []
        self.obstacles = []
        
        # 交互状态
        self.current_mode = 'start'  # 'start', 'target', 'obstacle', 'done'
        self.current_uav_index = 0
        
    def run(self):
        """Run interactive demonstration"""
        print("=" * 60)
        print("Multi-UAV Path Planning Interactive Demo")
        print("=" * 60)
        print(f"Environment Size: {self.config.ENV_SIZE}x{self.config.ENV_SIZE}")
        print(f"Number of UAVs: {self.config.NUM_UAVS}")
        print(f"Number of Obstacles: {self.config.NUM_OBSTACLES}")
        print("=" * 60)
        
        # Step 1: Mark start and target positions
        self._interactive_setup()
        
        # Step 2: Load trained model
        print("\nLoading trained model...")
        agents = self._load_models()
        
        # Step 3: Run simulation
        print("\nStarting simulation...")
        self._run_simulation(agents)
        
    def _interactive_setup(self):
        """交互式标定起点、终点和障碍物"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.config.ENV_SIZE)
        ax.set_ylim(0, self.config.ENV_SIZE)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Click to Mark Positions - Mark Start Position for UAV 1', fontsize=14)
        
        # 绘制说明
        self._update_instructions(ax)
        
        def onclick(event):
            if event.inaxes != ax:
                return
            
            x, y = event.xdata, event.ydata
            
            if self.current_mode == 'start':
                # Mark start position
                self.start_positions.append([x, y])
                ax.plot(x, y, 'go', markersize=15, label=f'UAV{self.current_uav_index+1} Start')
                ax.text(x, y+2, f'S{self.current_uav_index+1}', ha='center', fontsize=10, color='green', weight='bold')
                
                self.current_uav_index += 1
                if self.current_uav_index >= self.config.NUM_UAVS:
                    self.current_mode = 'target'
                    self.current_uav_index = 0
                    ax.set_title('Mark Target Position for UAV 1', fontsize=14)
                else:
                    ax.set_title(f'Mark Start Position for UAV {self.current_uav_index+1}', fontsize=14)
                    
            elif self.current_mode == 'target':
                # Mark target position
                self.target_positions.append([x, y])
                ax.plot(x, y, 'r*', markersize=20, label=f'UAV{self.current_uav_index+1} Target')
                circle = Circle((x, y), self.config.TARGET_RADIUS, color='red', alpha=0.2)
                ax.add_patch(circle)
                ax.text(x, y+2, f'T{self.current_uav_index+1}', ha='center', fontsize=10, color='red', weight='bold')
                
                self.current_uav_index += 1
                if self.current_uav_index >= self.config.NUM_UAVS:
                    self.current_mode = 'obstacle'
                    self.current_uav_index = 0
                    ax.set_title(f'Mark Obstacle 1 Position (Remaining: {self.config.NUM_OBSTACLES})', fontsize=14)
                else:
                    ax.set_title(f'Mark Target Position for UAV {self.current_uav_index+1}', fontsize=14)
                    
            elif self.current_mode == 'obstacle':
                # Mark obstacle
                self.obstacles.append([x, y])
                ax.plot(x, y, 'kx', markersize=15)
                circle = Circle((x, y), self.config.OBSTACLE_RADIUS, color='black', alpha=0.3)
                ax.add_patch(circle)
                ax.text(x, y+2, f'O{len(self.obstacles)}', ha='center', fontsize=10, weight='bold')
                
                remaining = self.config.NUM_OBSTACLES - len(self.obstacles)
                if remaining > 0:
                    ax.set_title(f'Mark Obstacle {len(self.obstacles)+1} Position (Remaining: {remaining})', fontsize=14)
                else:
                    self.current_mode = 'done'
                    ax.set_title('Setup Complete! Close Window to Continue...', fontsize=14, color='green')
                    
            self._update_instructions(ax)
            plt.draw()
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        
        # 转换为numpy数组
        self.start_positions = np.array(self.start_positions)
        self.target_positions = np.array(self.target_positions)
        self.obstacles = np.array(self.obstacles)
        
        print("\nSetup Complete!")
        print(f"Start Positions: {self.start_positions}")
        print(f"Target Positions: {self.target_positions}")
        print(f"Obstacles: {self.obstacles}")
        
    def _update_instructions(self, ax):
        """Update instruction text"""
        instructions = []
        if self.current_mode == 'start':
            instructions.append(f"Current: Marking Start Positions ({len(self.start_positions)}/{self.config.NUM_UAVS})")
        elif self.current_mode == 'target':
            instructions.append(f"Current: Marking Target Positions ({len(self.target_positions)}/{self.config.NUM_UAVS})")
        elif self.current_mode == 'obstacle':
            instructions.append(f"Current: Marking Obstacles ({len(self.obstacles)}/{self.config.NUM_OBSTACLES})")
        else:
            instructions.append("Setup Complete!")
            
        # Display at bottom of chart
        ax.text(self.config.ENV_SIZE/2, -5, '\n'.join(instructions), 
                ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _load_models(self):
        """加载训练好的模型"""
        agents = []
        for i in range(self.config.NUM_UAVS):
            agent = PPOAgent(
                obs_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.shape[0],
                config=self.config
            )
            
            # 加载模型权重
            checkpoint = torch.load(self.model_path, map_location='cpu')
            agent.policy.load_state_dict(checkpoint['policy'])
            agent.policy.eval()
            
            agents.append(agent)
            print(f"  UAV {i} model loaded")
            
        return agents
    
    def _run_simulation(self, agents):
        """运行仿真"""
        # 重置环境并设置用户标定的位置
        self.env.reset()
        self.env.uav_positions = self.start_positions.copy()
        self.env.targets = self.target_positions.copy()
        self.env.obstacles = self.obstacles.copy()
        self.env.prev_distances = np.linalg.norm(self.env.uav_positions - self.env.targets, axis=1)
        
        # 创建可视化
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制环境
        ax.set_xlim(0, self.config.ENV_SIZE)
        ax.set_ylim(0, self.config.ENV_SIZE)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制障碍物
        for obs in self.obstacles:
            circle = Circle(obs, self.config.OBSTACLE_RADIUS, color='black', alpha=0.3)
            ax.add_patch(circle)
            ax.plot(obs[0], obs[1], 'kx', markersize=15)
        
        # 绘制目标
        for i, target in enumerate(self.target_positions):
            circle = Circle(target, self.config.TARGET_RADIUS, color='red', alpha=0.2)
            ax.add_patch(circle)
            ax.plot(target[0], target[1], 'r*', markersize=20)
            ax.text(target[0], target[1]+3, f'T{i+1}', ha='center', fontsize=10, color='red', weight='bold')
        
        # 绘制起点
        for i, start in enumerate(self.start_positions):
            ax.plot(start[0], start[1], 'go', markersize=15)
            ax.text(start[0], start[1]+3, f'S{i+1}', ha='center', fontsize=10, color='green', weight='bold')
        
        # 初始化UAV轨迹
        uav_colors = ['blue', 'cyan', 'magenta', 'yellow', 'orange']
        uav_plots = []
        trajectory_lines = []
        for i in range(self.config.NUM_UAVS):
            color = uav_colors[i % len(uav_colors)]
            plot, = ax.plot([], [], 'o', color=color, markersize=12, label=f'UAV{i+1}')
            uav_plots.append(plot)
            line, = ax.plot([], [], '-', color=color, alpha=0.5, linewidth=2)
            trajectory_lines.append(line)
        
        ax.legend(loc='upper right')
        ax.set_title('Path Planning Simulation - Running...', fontsize=14)
        
        # 存储轨迹
        trajectories = [[] for _ in range(self.config.NUM_UAVS)]
        
        # 运行仿真
        done = False
        step = 0
        all_reached = False
        
        while not done and step < self.config.MAX_STEPS:
            # 获取每个UAV的动作
            actions = []
            for i in range(self.config.NUM_UAVS):
                obs = self.env._get_observation(i)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                with torch.no_grad():
                    action, _ = agents[i].select_action(obs_tensor)
                actions.append(action)
            
            # 环境步进
            obs, rewards, done, truncated, info = self.env.step(actions)
            
            # 记录轨迹
            for i in range(self.config.NUM_UAVS):
                trajectories[i].append(self.env.uav_positions[i].copy())
            
            # 更新可视化 (每5步更新一次以提高性能)
            if step % 5 == 0:
                for i in range(self.config.NUM_UAVS):
                    uav_plots[i].set_data([self.env.uav_positions[i, 0]], [self.env.uav_positions[i, 1]])
                    
                    if len(trajectories[i]) > 0:
                        traj = np.array(trajectories[i])
                        trajectory_lines[i].set_data(traj[:, 0], traj[:, 1])
                
                # Check if all UAVs reached target
                all_reached = np.all(self.env.reached_target)
                if all_reached:
                    ax.set_title(f'Simulation Complete! All UAVs Reached Target (Steps: {step})', fontsize=14, color='green')
                else:
                    ax.set_title(f'Path Planning Simulation - Step: {step}', fontsize=14)
                
                plt.pause(0.01)
            
            step += 1
            
            if all_reached:
                break
        
        if all_reached:
            print(f"\n✓ Simulation completed successfully! All UAVs reached target in {step} steps")
        else:
            print(f"\n✗ Simulation ended. Some UAVs did not reach target (Max steps: {self.config.MAX_STEPS})")
        
        print("\nStatistics:")
        for i in range(self.config.NUM_UAVS):
            if self.env.reached_target[i]:
                print(f"  UAV{i+1}: Reached target ✓")
            else:
                dist = np.linalg.norm(self.env.uav_positions[i] - self.env.targets[i])
                print(f"  UAV{i+1}: Did not reach target (Distance: {dist:.2f})")
        
        plt.show()


def main():
    """主函数"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description='Multi-UAV Path Planning Interactive Demo - Manual Marking of Start/Target Positions and Obstacles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python interactive_demo.py                                      # Use latest model
  python interactive_demo.py models/ppo_uav_0_episode_124500.pth  # Specify model
  python interactive_demo.py --list                               # List all available models

Model file location:
  .pth files in the models/ directory
  Format: ppo_uav_0_episode_<episode_number>.pth
        '''
    )
    
    parser.add_argument(
        'model_path',
        nargs='?',
        default=None,
        help='Path to trained model file (optional, defaults to latest model)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available model files'
    )
    
    parser.add_argument(
        '--model-dir',
        default='models',
        help='Model file directory (default: models/)'
    )
    
    args = parser.parse_args()
    
    # Helper function: extract episode number from filename
    def extract_episode(filename):
        try:
            # Try to extract number from episode_XXXXX format
            parts = filename.replace('.pth', '').split('_')
            for i, part in enumerate(parts):
                if part == 'episode' and i + 1 < len(parts):
                    return int(parts[i + 1])
            # If episode not found, try to parse last part directly
            return int(parts[-1])
        except (ValueError, IndexError):
            # If unable to parse, return -1 for lowest priority
            return -1
    
    # List all available models
    if args.list:
        model_dir = args.model_dir
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if models:
                models.sort(key=extract_episode, reverse=True)
                print(f"\nFound {len(models)} models in {model_dir}/ directory:")
                print("-" * 60)
                for i, model in enumerate(models[:10], 1):  # Show first 10 only
                    episode = extract_episode(model)
                    size = os.path.getsize(os.path.join(model_dir, model)) / 1024
                    ep_str = f"Episode {episode}" if episode >= 0 else "Unknown"
                    print(f"{i:2d}. {model:45s} ({ep_str:>15s}, {size:6.1f} KB)")
                if len(models) > 10:
                    print(f"    ... and {len(models) - 10} more models")
                print("-" * 60)
                print(f"Latest model: {models[0]}")
            else:
                print(f"Error: No .pth model files found in {model_dir}/ directory")
        else:
            print(f"Error: Directory does not exist: {model_dir}/")
        return
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Use latest model
        model_dir = args.model_dir
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if models:
                # Sort by episode number, select latest
                models.sort(key=extract_episode, reverse=True)
                model_path = os.path.join(model_dir, models[0])
                print(f"Using latest model: {model_path}")
            else:
                print(f"Error: No trained models found in {model_dir}/ directory")
                print("\nHints:")
                print("  1. Run train.py to train a model first")
                print("  2. Or use --list to view available models")
                return
        else:
            print(f"Error: Model directory does not exist: {model_dir}/")
            return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file does not exist: {model_path}")
        print(f"\nUse --list to view all available models")
        return
    
    # Create and run demo
    demo = InteractiveDemo(model_path)
    demo.run()


if __name__ == "__main__":
    main()
