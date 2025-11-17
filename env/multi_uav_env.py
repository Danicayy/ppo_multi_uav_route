"""
多无人机协同路径规划环境
连续状态空间和动作空间
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt


class MultiUAVEnv(gym.Env):    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 环境参数
        self.env_size = config.ENV_SIZE
        self.num_uavs = config.NUM_UAVS
        self.num_obstacles = config.NUM_OBSTACLES
        self.max_steps = config.MAX_STEPS
        
        # 状态空间: 每个UAV观察 [自己位置(2), 自己速度(2), 目标位置(2), 
        #                      其他UAV相对位置和速度(2+2)*n, 障碍物相对位置(2)*m]
        obs_dim = 2 + 2 + 2 + (2 + 2) * (self.num_uavs - 1) + 2 * self.num_obstacles
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 动作空间: 连续动作 [ax, ay] - 加速度
        self.action_space = spaces.Box(
            low=-config.UAV_MAX_ACCELERATION,
            high=config.UAV_MAX_ACCELERATION,
            shape=(2,),
            dtype=np.float32
        )
        
        # 初始化环境状态
        self.uav_positions = np.zeros((self.num_uavs, 2))
        self.uav_velocities = np.zeros((self.num_uavs, 2))
        self.targets = np.zeros((self.num_uavs, 2))
        self.obstacles = np.zeros((self.num_obstacles, 2))
        self.reached_target = np.zeros(self.num_uavs, dtype=bool)
        
        # 用于计算距离变化的奖励
        self.prev_distances = np.zeros(self.num_uavs)
        
        self.step_count = 0
        self.trajectories = []
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机生成无人机起始位置
        self.uav_positions = np.random.uniform(
            10, self.env_size - 10, (self.num_uavs, 2)
        )
        self.uav_velocities = np.zeros((self.num_uavs, 2))
        
        # 随机生成目标位置（远离起始位置）
        self.targets = np.random.uniform(
            10, self.env_size - 10, (self.num_uavs, 2)
        )
        # 确保目标距离起始位置足够远；但在小环境中避免无限重试
        # 计算当前坐标范围内可能的最大距离，并根据其调整最小目标距离
        max_possible = np.sqrt(2.0) * (self.env_size - 20.0)
        min_target_distance = min(30.0, max_possible)
        max_attempts = 100
        for i in range(self.num_uavs):
            attempts = 0
            while np.linalg.norm(self.targets[i] - self.uav_positions[i]) < min_target_distance and attempts < max_attempts:
                self.targets[i] = np.random.uniform(10, self.env_size - 10, 2)
                attempts += 1
            # 如果经过若干次仍未找到满足条件的位置，使用一个确定性的回退：
            # 在随机方向上放置一个距离为 min_target_distance 的目标，并裁剪到合法范围
            if np.linalg.norm(self.targets[i] - self.uav_positions[i]) < min_target_distance:
                direction = np.random.randn(2)
                norm = np.linalg.norm(direction)
                if norm == 0:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = direction / norm
                fallback = self.uav_positions[i] + direction * min_target_distance
                self.targets[i] = np.clip(fallback, 10, self.env_size - 10)
        
        # 随机生成障碍物
        self.obstacles = np.random.uniform(
            15, self.env_size - 15, (self.num_obstacles, 2)
        )
        
        self.reached_target = np.zeros(self.num_uavs, dtype=bool)
        self.step_count = 0
        self.trajectories = [[] for _ in range(self.num_uavs)]
        
        # 初始化距离记录
        for i in range(self.num_uavs):
            self.prev_distances[i] = np.linalg.norm(
                self.uav_positions[i] - self.targets[i]
            )
        
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def _get_obs(self):
        """获取观察"""
        observations = []
        
        for i in range(self.num_uavs):
            obs = []
            
            # 自己的位置和速度（归一化）
            obs.extend(self.uav_positions[i] / self.env_size)
            obs.extend(self.uav_velocities[i] / self.config.UAV_MAX_SPEED)
            
            # 目标位置（归一化）
            obs.extend(self.targets[i] / self.env_size)
            
            # 其他UAV的相对位置和速度
            for j in range(self.num_uavs):
                if i != j:
                    rel_pos = (self.uav_positions[j] - self.uav_positions[i]) / self.env_size
                    rel_vel = (self.uav_velocities[j] - self.uav_velocities[i]) / self.config.UAV_MAX_SPEED
                    obs.extend(rel_pos)
                    obs.extend(rel_vel)
            
            # 障碍物相对位置
            for obs_pos in self.obstacles:
                rel_obs = (obs_pos - self.uav_positions[i]) / self.env_size
                obs.extend(rel_obs)
            
            observations.append(np.array(obs, dtype=np.float32))
        
        return np.array(observations)
    
    def step(self, actions):
        """执行动作"""
        self.step_count += 1
        rewards = np.zeros(self.num_uavs)
        
        # 更新每个UAV的状态
        for i in range(self.num_uavs):
            if not self.reached_target[i]:
                # 更新速度（应用加速度）
                self.uav_velocities[i] += actions[i]
                
                # 限制速度
                speed = np.linalg.norm(self.uav_velocities[i])
                if speed > self.config.UAV_MAX_SPEED:
                    self.uav_velocities[i] = (
                        self.uav_velocities[i] / speed * self.config.UAV_MAX_SPEED
                    )
                
                # 更新位置
                self.uav_positions[i] += self.uav_velocities[i]
                
                # 边界处理
                self.uav_positions[i] = np.clip(
                    self.uav_positions[i], 0, self.env_size
                )
                
                # 记录轨迹
                self.trajectories[i].append(self.uav_positions[i].copy())
        
        # 计算奖励
        rewards = self._calculate_rewards()
        
        # 检查终止条件
        terminated = all(self.reached_target) or self.step_count >= self.max_steps
        truncated = False
        
        obs = self._get_obs()
        info = {
            'reached_target': self.reached_target.copy(),
            'num_reached': np.sum(self.reached_target)
        }
        
        return obs, rewards, terminated, truncated, info
    
    def _calculate_rewards(self):
        """计算奖励 - 基于距离改进的奖励"""
        rewards = np.zeros(self.num_uavs)
        
        for i in range(self.num_uavs):
            if self.reached_target[i]:
                rewards[i] = 0  # 已到达目标，不再给奖励
                continue
            
            # 1. 计算当前距离目标的距离
            current_dist = np.linalg.norm(
                self.uav_positions[i] - self.targets[i]
            )
            
            # 2. 距离改进奖励（核心奖励）
            # 如果靠近目标则为正，远离目标则为负
            distance_improvement = self.prev_distances[i] - current_dist
            rewards[i] += distance_improvement * self.config.REWARD_DISTANCE
            
            # 更新上一步的距离
            self.prev_distances[i] = current_dist
            
            # 3. 到达目标的巨大奖励
            if current_dist < self.config.TARGET_RADIUS:
                rewards[i] += self.config.REWARD_TARGET
                self.reached_target[i] = True
                continue  # 到达目标后不再计算其他惩罚
            
            # 4. 障碍物碰撞 - 严厉惩罚
            collision_with_obstacle = False
            for obs_pos in self.obstacles:
                dist_to_obs = np.linalg.norm(self.uav_positions[i] - obs_pos)
                if dist_to_obs < self.config.OBSTACLE_RADIUS:
                    rewards[i] += self.config.REWARD_COLLISION_OBSTACLE
                    collision_with_obstacle = True
                    break  # 一次碰撞就够了
                elif dist_to_obs < self.config.OBSTACLE_RADIUS * 2.0:
                    # 接近障碍物的渐进惩罚（距离越近惩罚越大）
                    danger_zone = self.config.OBSTACLE_RADIUS * 2.0
                    proximity_penalty = 5.0 * (1.0 - (dist_to_obs / danger_zone))
                    rewards[i] -= proximity_penalty
            
            # 5. UAV间碰撞
            for j in range(self.num_uavs):
                if i != j:
                    dist_to_uav = np.linalg.norm(
                        self.uav_positions[i] - self.uav_positions[j]
                    )
                    if dist_to_uav < self.config.COLLISION_DISTANCE:
                        rewards[i] += self.config.REWARD_COLLISION_UAV
            
            # 6. 小的步数惩罚（鼓励快速到达）
            rewards[i] += self.config.REWARD_STEP
            
            # 7. 出界惩罚
            if (self.uav_positions[i, 0] <= 0 or self.uav_positions[i, 0] >= self.env_size or
                self.uav_positions[i, 1] <= 0 or self.uav_positions[i, 1] >= self.env_size):
                rewards[i] -= 5.0
        
        return rewards
    
    def render(self, mode='human'):
        """渲染环境"""
        plt.clf()
        plt.xlim(0, self.env_size)
        plt.ylim(0, self.env_size)
        plt.gca().set_aspect('equal')
        
        # 绘制障碍物
        for obs_pos in self.obstacles:
            circle = plt.Circle(
                obs_pos, self.config.OBSTACLE_RADIUS, 
                color='red', alpha=0.5, label='Obstacle'
            )
            plt.gca().add_patch(circle)
        
        # 绘制目标点
        for i, target in enumerate(self.targets):
            circle = plt.Circle(
                target, self.config.TARGET_RADIUS, 
                color='green', alpha=0.3
            )
            plt.gca().add_patch(circle)
            plt.plot(target[0], target[1], 'g*', markersize=15, label=f'Target {i+1}')
        
        # 绘制UAV轨迹
        colors = ['blue', 'orange', 'purple', 'brown', 'pink']
        for i in range(self.num_uavs):
            if len(self.trajectories[i]) > 1:
                traj = np.array(self.trajectories[i])
                plt.plot(
                    traj[:, 0], traj[:, 1], 
                    color=colors[i % len(colors)], 
                    alpha=0.5, linewidth=2
                )
            
            # 绘制当前UAV位置
            plt.plot(
                self.uav_positions[i, 0], self.uav_positions[i, 1],
                'o', color=colors[i % len(colors)], 
                markersize=10, label=f'UAV {i+1}'
            )
            
            # 绘制速度向量
            plt.arrow(
                self.uav_positions[i, 0], self.uav_positions[i, 1],
                self.uav_velocities[i, 0] * 2, self.uav_velocities[i, 1] * 2,
                head_width=2, head_length=1, 
                fc=colors[i % len(colors)], ec=colors[i % len(colors)]
            )
        
        plt.title(f'Multi-UAV Path Planning (Step: {self.step_count})')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # 去重legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.grid(True, alpha=0.3)
        plt.pause(0.01)