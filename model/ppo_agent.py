"""
PPO算法实现
包含Actor-Critic网络和PPO训练逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super(ActorCritic, self).__init__()
        
        # Actor网络（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
        # Actor标准差（可学习）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, obs):
        return self.actor(obs), self.critic(obs)
    
    def get_action(self, obs, deterministic=False):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        
        if deterministic:
            return mean, None, None
        
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        
        return action, log_prob, dist.entropy().sum(-1)
    
    def evaluate_actions(self, obs, actions):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        
        return log_prob, entropy, value


class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, obs_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.policy = ActorCritic(
            obs_dim, action_dim, config.HIDDEN_SIZE
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=config.LEARNING_RATE
        )
        
        # 存储经验
        self.buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _ = self.policy.get_action(obs_tensor, deterministic)
            value = self.policy.critic(obs_tensor).squeeze(-1)
        
        return action.cpu().numpy(), log_prob, value
    
    def store_transition(self, obs, action, reward, value, log_prob, done):
        self.buffer['obs'].append(obs)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def compute_gae(self, next_value):
        rewards = np.array(self.buffer['rewards'])
        values = torch.stack(self.buffer['values']).cpu().numpy()
        dones = np.array(self.buffer['dones'])
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # 逆序计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.GAMMA * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.config.GAMMA * self.config.GAE_LAMBDA * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, next_obs):
        # 计算下一个状态的价值
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
            next_value = self.policy.critic(next_obs_tensor).squeeze(-1).cpu().numpy()
        
        # 计算优势和回报
        advantages, returns = self.compute_gae(next_value)
        
        # 转换为tensor
        obs = torch.FloatTensor(np.array(self.buffer['obs'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.stack(self.buffer['log_probs']).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_loss = 0
        for _ in range(self.config.NUM_EPOCHS):
            # 随机打乱数据
            indices = np.random.permutation(len(obs))
            
            for start in range(0, len(obs), self.config.BATCH_SIZE):
                end = start + self.config.BATCH_SIZE
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估动作
                log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )
                
                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config.CLIP_EPSILON, 
                    1 + self.config.CLIP_EPSILON
                ) * batch_advantages
                
                # 损失函数
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, batch_returns)
                entropy_loss = -entropy.mean()
                
                loss = (
                    actor_loss + 
                    self.config.VALUE_LOSS_COEF * critic_loss + 
                    self.config.ENTROPY_COEF * entropy_loss
                )
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.MAX_GRAD_NORM
                )
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # 清空buffer
        self.clear_buffer()
        
        return total_loss / (self.config.NUM_EPOCHS * (len(obs) // self.config.BATCH_SIZE + 1))
    
    def clear_buffer(self):
        self.buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))