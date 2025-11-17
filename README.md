# 多无人机协同路径规划 - PPO算法实现

基于PPO（Proximal Policy Optimization）强化学习算法实现的多无人机协同路径规划系统。支持连续状态和动作空间，包含障碍物避障、目标到达和无人机间协作。

## 项目概述

### 主要特性

-  **PPO算法**: 使用先进的策略梯度算法
-  **多智能体协同**: 支持多架无人机协同规划
-  **连续空间**: 连续状态空间和动作空间
-  **障碍物避障**: 智能避开环境中的障碍物
-  **实时可视化**: 训练和测试过程可视化
-  **模块化设计**: 易于扩展和定制

### 算法原理

#### PPO (Proximal Policy Optimization)

PPO是一种策略优化算法，通过限制策略更新幅度来保证训练稳定性。

**核心思想**:
1. 收集轨迹数据
2. 计算优势函数（Advantage Function）
3. 使用裁剪目标函数更新策略
4. 重复以上步骤

**目标函数**:
```
L(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
```

其中:
- `r(θ) = π_θ(a|s) / π_θ_old(a|s)` 是概率比率
- `A` 是优势函数
- `ε` 是裁剪参数（通常为0.2）

#### Actor-Critic架构

- **Actor (策略网络)**: 输出动作的均值，学习最优策略
- **Critic (价值网络)**: 估计状态价值，用于计算优势函数

#### GAE (Generalized Advantage Estimation)

使用GAE计算优势函数，平衡偏差和方差:

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
```

其中 `δ_t = r_t + γV(s_{t+1}) - V(s_t)`

## 环境说明

### 状态空间

每个UAV的观察包含:
- 自身位置 (x, y)
- 自身速度 (vx, vy)
- 目标位置 (tx, ty)
- 其他UAV相对位置和速度
- 障碍物相对位置

**状态维度**: `2 + 2 + 2 + 4*(n-1) + 2*m`
- n: UAV数量
- m: 障碍物数量

### 动作空间

连续动作空间: `[ax, ay]`
- ax, ay: x和y方向的加速度
- 范围: `[-max_acceleration, max_acceleration]`

### 奖励函数

```python
总奖励 = 目标奖励 + 距离奖励 + 碰撞惩罚 + 协作奖励 + 步数惩罚
```

**具体奖励**:
- 到达目标: +100
- 距离奖励: +1.0 / (distance + 1)
- 碰撞障碍物: -50
- UAV碰撞: -30
- 协作奖励: +5 (当UAV在通信范围内协同工作)
- 每步惩罚: -0.1

## 项目结构

```
multi-uav-ppo/
├── README.md                 # 项目文档
├── requirements.txt          # 依赖包
├── config.py                 # 配置参数
├── env/
│   └── multi_uav_env.py     # 多无人机环境
├── model/
│   └── ppo_agent.py         # PPO智能体
├── train.py                  # 训练脚本
├── test.py                   # 测试脚本
├── visualize.py              # 可视化脚本
├── models/                   # 保存的模型
└── logs/                     # 训练日志
```

## 安装说明

### 1. 环境要求

- Python 3.8+
- CUDA (可选，用于GPU加速)

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 依赖包

- `torch>=2.0.0`: 深度学习框架
- `numpy>=1.24.0`: 数值计算
- `gymnasium>=0.29.0`: 强化学习环境
- `matplotlib>=3.7.0`: 可视化
- `pygame>=2.5.0`: 游戏引擎（可选）
- `tensorboard>=2.13.0`: 训练监控

## 使用指南

### 1. 训练模型

```bash
python train.py
```

**训练参数** (在 `config.py` 中修改):

- `NUM_EPISODES`: 训练轮数 
- `NUM_UAVS`: 无人机数量
- `NUM_OBSTACLES`: 障碍物数量 
- `LEARNING_RATE`: 学习率

**训练监控**:
```bash
tensorboard --logdir logs/
```

### 2. 测试模型

```bash
# 测试10个episodes并渲染
python test.py --episodes 10 --render

# 测试特定版本的模型
python test.py --episodes 5 --render --model episode_1000
```

**参数说明**:
- `--episodes`: 测试轮数
- `--render`: 是否显示可视化
- `--model`: 模型版本 (默认: final)

### 3. 可视化

```bash
# 显示动画
python visualize.py

# 保存为视频
python visualize.py --save

# 可视化特定模型
python visualize.py --model episode_1000
```

## 配置参数

### 环境参数

```python
ENV_SIZE = 100.0           # 环境大小
NUM_UAVS = 3              # 无人机数量
NUM_OBSTACLES = 8         # 障碍物数量
OBSTACLE_RADIUS = 5.0     # 障碍物半径
TARGET_RADIUS = 3.0       # 目标半径
MAX_STEPS = 500           # 最大步数
```

### UAV参数

```python
UAV_MAX_SPEED = 2.0           # 最大速度
UAV_MAX_ACCELERATION = 0.5    # 最大加速度
COLLISION_DISTANCE = 3.0      # 碰撞距离
COMMUNICATION_RANGE = 30.0    # 通信范围
```

### PPO参数

```python
LEARNING_RATE = 3e-4      # 学习率
GAMMA = 0.99              # 折扣因子
GAE_LAMBDA = 0.95         # GAE参数
CLIP_EPSILON = 0.2        # 裁剪参数
ENTROPY_COEF = 0.01       # 熵系数
VALUE_LOSS_COEF = 0.5     # 价值损失系数
```

## 算法流程

### 整体流程图

```
┌─────────────┐
│  初始化环境  │
│  和智能体   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  重置环境   │
│  获取初始   │
│   状态 s₀   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│    主循环 (每个Episode)  │
│  ┌─────────────────┐   │
│  │ 1. 选择动作 a   │   │
│  │ 2. 执行动作     │   │
│  │ 3. 获取奖励 r   │   │
│  │ 4. 存储经验     │   │
│  │ 5. 更新状态     │   │
│  └─────────────────┘   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────┐
│  经验积累   │
│  达到阈值?  │
└──────┬──────┘
       │ 是
       ▼
┌─────────────┐
│  计算GAE    │
│  和回报     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  PPO更新    │
│  (多个epoch)│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  评估和保存  │
└─────────────┘
```



