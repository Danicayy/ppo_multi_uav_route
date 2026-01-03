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
        """运行交互式演示"""
        print("=" * 60)
        print("多无人机路径规划交互式演示")
        print("=" * 60)
        print(f"环境大小: {self.config.ENV_SIZE}x{self.config.ENV_SIZE}")
        print(f"无人机数量: {self.config.NUM_UAVS}")
        print(f"障碍物数量: {self.config.NUM_OBSTACLES}")
        print("=" * 60)
        
        # 步骤1: 标定起点和终点
        self._interactive_setup()
        
        # 步骤2: 加载模型
        print("\n正在加载训练好的模型...")
        agents = self._load_models()
        
        # 步骤3: 运行仿真
        print("\n开始运行仿真...")
        self._run_simulation(agents)
        
    def _interactive_setup(self):
        """交互式标定起点、终点和障碍物"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.config.ENV_SIZE)
        ax.set_ylim(0, self.config.ENV_SIZE)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('点击标定位置 - 请标定无人机1的起点', fontsize=14)
        
        # 绘制说明
        self._update_instructions(ax)
        
        def onclick(event):
            if event.inaxes != ax:
                return
            
            x, y = event.xdata, event.ydata
            
            if self.current_mode == 'start':
                # 标定起点
                self.start_positions.append([x, y])
                ax.plot(x, y, 'go', markersize=15, label=f'UAV{self.current_uav_index+1} 起点')
                ax.text(x, y+2, f'S{self.current_uav_index+1}', ha='center', fontsize=10, color='green', weight='bold')
                
                self.current_uav_index += 1
                if self.current_uav_index >= self.config.NUM_UAVS:
                    self.current_mode = 'target'
                    self.current_uav_index = 0
                    ax.set_title('请标定无人机1的终点', fontsize=14)
                else:
                    ax.set_title(f'请标定无人机{self.current_uav_index+1}的起点', fontsize=14)
                    
            elif self.current_mode == 'target':
                # 标定终点
                self.target_positions.append([x, y])
                ax.plot(x, y, 'r*', markersize=20, label=f'UAV{self.current_uav_index+1} 终点')
                circle = Circle((x, y), self.config.TARGET_RADIUS, color='red', alpha=0.2)
                ax.add_patch(circle)
                ax.text(x, y+2, f'T{self.current_uav_index+1}', ha='center', fontsize=10, color='red', weight='bold')
                
                self.current_uav_index += 1
                if self.current_uav_index >= self.config.NUM_UAVS:
                    self.current_mode = 'obstacle'
                    self.current_uav_index = 0
                    ax.set_title('请标定障碍物1的位置 (剩余: 3)', fontsize=14)
                else:
                    ax.set_title(f'请标定无人机{self.current_uav_index+1}的终点', fontsize=14)
                    
            elif self.current_mode == 'obstacle':
                # 标定障碍物
                self.obstacles.append([x, y])
                ax.plot(x, y, 'kx', markersize=15)
                circle = Circle((x, y), self.config.OBSTACLE_RADIUS, color='black', alpha=0.3)
                ax.add_patch(circle)
                ax.text(x, y+2, f'O{len(self.obstacles)}', ha='center', fontsize=10, weight='bold')
                
                remaining = self.config.NUM_OBSTACLES - len(self.obstacles)
                if remaining > 0:
                    ax.set_title(f'请标定障碍物{len(self.obstacles)+1}的位置 (剩余: {remaining})', fontsize=14)
                else:
                    self.current_mode = 'done'
                    ax.set_title('标定完成！关闭窗口继续...', fontsize=14, color='green')
                    
            self._update_instructions(ax)
            plt.draw()
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        
        # 转换为numpy数组
        self.start_positions = np.array(self.start_positions)
        self.target_positions = np.array(self.target_positions)
        self.obstacles = np.array(self.obstacles)
        
        print("\n标定完成！")
        print(f"起点: {self.start_positions}")
        print(f"终点: {self.target_positions}")
        print(f"障碍物: {self.obstacles}")
        
    def _update_instructions(self, ax):
        """更新说明文字"""
        instructions = []
        if self.current_mode == 'start':
            instructions.append(f"当前: 标定起点 ({len(self.start_positions)}/{self.config.NUM_UAVS})")
        elif self.current_mode == 'target':
            instructions.append(f"当前: 标定终点 ({len(self.target_positions)}/{self.config.NUM_UAVS})")
        elif self.current_mode == 'obstacle':
            instructions.append(f"当前: 标定障碍物 ({len(self.obstacles)}/{self.config.NUM_OBSTACLES})")
        else:
            instructions.append("标定完成！")
            
        # 显示在图表底部
        ax.text(self.config.ENV_SIZE/2, -5, '\n'.join(instructions), 
                ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _load_models(self):
        """加载训练好的模型"""
        agents = []
        for i in range(self.config.NUM_UAVS):
            agent = PPOAgent(
                state_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.shape[0],
                config=self.config
            )
            
            # 加载模型权重
            checkpoint = torch.load(self.model_path, map_location='cpu')
            agent.actor.load_state_dict(checkpoint['actor'])
            agent.critic.load_state_dict(checkpoint['critic'])
            agent.actor.eval()
            agent.critic.eval()
            
            agents.append(agent)
            print(f"  UAV {i} 模型加载完成")
            
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
        ax.set_title('路径规划仿真 - 运行中...', fontsize=14)
        
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
                
                # 检查是否所有UAV都到达目标
                all_reached = np.all(self.env.reached_target)
                if all_reached:
                    ax.set_title(f'仿真完成！所有UAV已到达目标 (步数: {step})', fontsize=14, color='green')
                else:
                    ax.set_title(f'路径规划仿真 - 步数: {step}', fontsize=14)
                
                plt.pause(0.01)
            
            step += 1
            
            if all_reached:
                break
        
        if all_reached:
            print(f"\n✓ 仿真成功完成！所有UAV在 {step} 步内到达目标")
        else:
            print(f"\n✗ 仿真结束。部分UAV未到达目标 (最大步数: {self.config.MAX_STEPS})")
        
        print("\n统计信息:")
        for i in range(self.config.NUM_UAVS):
            if self.env.reached_target[i]:
                print(f"  UAV{i+1}: 已到达目标 ✓")
            else:
                dist = np.linalg.norm(self.env.uav_positions[i] - self.env.targets[i])
                print(f"  UAV{i+1}: 未到达目标 (距离: {dist:.2f})")
        
        plt.show()


def main():
    """主函数"""
    import sys
    import os
    
    # 检查是否提供了模型路径
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # 使用最新的模型
        model_dir = "models"
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if models:
                # 按episode数排序，选择最新的
                models.sort(key=lambda x: int(x.split('_')[-1].replace('.pth', '')), reverse=True)
                model_path = os.path.join(model_dir, models[0])
                print(f"使用模型: {model_path}")
            else:
                print("错误: 没有找到训练好的模型")
                print("用法: python interactive_demo.py [model_path]")
                return
        else:
            print("错误: models目录不存在")
            return
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    # 创建并运行演示
    demo = InteractiveDemo(model_path)
    demo.run()


if __name__ == "__main__":
    main()
