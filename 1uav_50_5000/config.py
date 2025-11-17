
class Config:
    # 环境参数（简化训练）
    ENV_SIZE = 50.0  # 环境大小缩小到50x50
    NUM_UAVS = 1  # 先训练单个UAV
    NUM_OBSTACLES = 3  # 减少障碍物
    OBSTACLE_RADIUS = 4.0  # 障碍物半径
    TARGET_RADIUS = 5.0  # 目标点半径
    MAX_STEPS = 200  # 减少最大步数
    
    # UAV参数
    UAV_MAX_SPEED = 2.0  # 最大速度
    UAV_MAX_ACCELERATION = 0.5  # 最大加速度
    COLLISION_DISTANCE = 3.0  # 碰撞距离
    COMMUNICATION_RANGE = 30.0  # 通信范围
    
    # PPO算法参数
    LEARNING_RATE = 5e-4  # 提高学习率
    GAMMA = 0.99  # 折扣因子
    GAE_LAMBDA = 0.95  # GAE参数
    CLIP_EPSILON = 0.2  # PPO裁剪参数
    ENTROPY_COEF = 0.01  # 熵系数（适度探索，避免过度随机）
    VALUE_LOSS_COEF = 0.5  # 价值损失系数
    MAX_GRAD_NORM = 0.5  # 梯度裁剪
    
    # 训练参数
    NUM_EPISODES = 5000  # 训练episode数
    UPDATE_INTERVAL = 512  # 更新间隔（约2-3个episode更新一次）
    BATCH_SIZE = 64
    NUM_EPOCHS = 5  # 每次更新的epoch数（降低防止过拟合）
    
    # 神经网络参数
    HIDDEN_SIZE = 256
    
    # 奖励权重（重新设计）
    REWARD_TARGET = 500.0  # 到达目标的巨大奖励
    REWARD_COLLISION_OBSTACLE = -100.0  # 碰撞障碍物严厉惩罚
    REWARD_COLLISION_UAV = -50.0  # UAV碰撞惩罚
    REWARD_STEP = -0.1  # 每步小惩罚（鼓励快速到达）
    REWARD_DISTANCE = 8.0  # 距离改进奖励系数（提升到8.0以增强信号）
    REWARD_COOPERATION = 0.0  # 暂时禁用协作奖励
    
    # 保存路径
    MODEL_SAVE_PATH = "models/"
    LOG_DIR = "logs/"