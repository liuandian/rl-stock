# 导入必要的库
import numpy as np                # 科学计算库，用于数组和矩阵运算
import torch                     # PyTorch深度学习库
import torch.nn as nn             # PyTorch神经网络模块
import torch.optim as optim       # PyTorch优化器模块
from torch.distributions import Normal  # PyTorch正态分布模块，用于随机策略
import gymnasium as gym          # 强化学习环境库
from gymnasium import spaces     # 环境空间定义
import time                      # 时间处理模块
from tqdm import tqdm            # 进度条模块，用于显示训练进度

class ActorCritic(nn.Module):
    """演员-评论家网络，包含策略网络（Actor）和价值网络（Critic）

    演员网络负责选择动作，评论家网络负责评估状态价值
    """
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        """初始化演员-评论家网络

        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            action_std_init: 动作标准差初始值，控制探索程度
        """
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim  # 动作空间维度
        # 初始化动作方差（用于探索）
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

        # 演员网络（Actor）- 输出动作
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),  # 输入层到隐藏层
            nn.Tanh(),                # 激活函数
            nn.Linear(64, 64),        # 第一个隐藏层到第二个隐藏层
            nn.Tanh(),                # 激活函数
            nn.Linear(64, action_dim), # 隐藏层到输出层
        )

        # 评论家网络（Critic）- 评估状态价值
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),  # 输入层到隐藏层
            nn.Tanh(),                # 激活函数
            nn.Linear(64, 64),        # 第一个隐藏层到第二个隐藏层
            nn.Tanh(),                # 激活函数
            nn.Linear(64, 1)          # 隐藏层到输出层（一个标量值）
        )

    def set_action_std(self, new_action_std):
        """设置动作的标准差（控制探索程度）

        Args:
            new_action_std: 新的动作标准差
        """
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)

    def forward(self):
        """前向传播函数（未实现）

        注意：这个函数没有实现，因为我们使用单独的act和evaluate函数
        """
        raise NotImplementedError

    def act(self, state):
        """根据状态选择动作

        Args:
            state: 当前状态

        Returns:
            action: 选择的动作
            action_logprob: 动作的对数概率
        """
        # 检查状态是否包含NaN值
        if torch.isnan(state).any():
            print("\n\n警告: 状态包含NaN值!")
            # 将NaN替换为0
            state = torch.nan_to_num(state, nan=0.0)

        action_mean = self.actor(state)  # 从演员网络获取动作均值

        # 检查action_mean是否包含NaN值
        if torch.isnan(action_mean).any():
            print("\n\n警告: 动作均值包含NaN值!")
            # 将NaN替换为0
            action_mean = torch.nan_to_num(action_mean, nan=0.0)

        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # 协方差矩阵
        dist = Normal(action_mean, torch.sqrt(self.action_var))  # 创建正态分布

        action = dist.sample()  # 从分布中采样动作
        action_logprob = dist.log_prob(action).sum(dim=-1)  # 计算动作的对数概率

        return action.detach(), action_logprob.detach()  # 分离计算图，返回值

    def evaluate(self, state, action):
        """评估状态和动作

        Args:
            state: 当前状态
            action: 执行的动作

        Returns:
            action_logprobs: 动作的对数概率
            state_values: 状态价值
            dist_entropy: 分布的熵（用于鼓励探索）
        """
        # 添加检查点，确保状态不包含NaN
        if torch.isnan(state).any():
            print("\n\n警告: 状态包含NaN值!")
            # 将NaN替换为0
            state = torch.nan_to_num(state, nan=0.0)

        action_mean = self.actor(state)  # 从演员网络获取动作均值

        # 检查action_mean是否包含NaN
        if torch.isnan(action_mean).any():
            print("\n\n警告: 动作均值包含NaN值!")
            # 将NaN替换为0
            action_mean = torch.nan_to_num(action_mean, nan=0.0)

        action_var = self.action_var.expand_as(action_mean)  # 扩展动作方差

        # 确保 action_var 中没有负值或者非常小的值
        action_var = torch.clamp(action_var, min=1e-6)

        dist = Normal(action_mean, torch.sqrt(action_var))  # 创建正态分布

        action_logprobs = dist.log_prob(action).sum(dim=-1)  # 计算动作的对数概率
        dist_entropy = dist.entropy().sum(dim=-1)  # 计算分布的熵
        state_values = self.critic(state)  # 从评论家网络获取状态价值

        # 检查返回值是否包含NaN
        if torch.isnan(action_logprobs).any() or torch.isnan(state_values).any() or torch.isnan(dist_entropy).any():
            print("\n\n警告: 评估结果包含NaN值!")
            # 将NaN替换为0
            action_logprobs = torch.nan_to_num(action_logprobs, nan=0.0)
            state_values = torch.nan_to_num(state_values, nan=0.0)
            dist_entropy = torch.nan_to_num(dist_entropy, nan=0.0)

        return action_logprobs, state_values, dist_entropy

class RolloutBuffer:
    """经验回放缓冲区，用于存储训练数据

    存储代理与环境交互的经验，包括状态、动作、奖励等
    """
    def __init__(self):
        """初始化经验回放缓冲区
        """
        self.actions = []        # 存储动作
        self.states = []         # 存储状态
        self.logprobs = []       # 存储动作的对数概率
        self.rewards = []        # 存储奖励
        self.is_terminals = []   # 存储是否为终止状态

    def clear(self):
        """清空缓冲区中的所有数据
        """
        del self.actions[:]      # 清空动作列表
        del self.states[:]       # 清空状态列表
        del self.logprobs[:]     # 清空对数概率列表
        del self.rewards[:]      # 清空奖励列表
        del self.is_terminals[:] # 清空终止状态列表

class PPO:
    """近端策略优化（Proximal Policy Optimization）算法实现

    PPO是一种流行的策略梯度算法，它通过限制策略更新的幅度来提高稳定性
    """
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=4, eps_clip=0.2, action_std_init=0.6):
        """初始化PPO算法

        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr_actor: 演员网络学习率
            lr_critic: 评论家网络学习率
            gamma: 折扣因子
            K_epochs: 每次更新的迭代次数
            eps_clip: 裁剪参数，限制策略更新的幅度
            action_std_init: 动作标准差初始值
        """
        self.gamma = gamma          # 折扣因子，用于计算未来奖励的现值
        self.eps_clip = eps_clip    # 裁剪参数，限制策略更新的幅度
        self.K_epochs = K_epochs    # 每次更新的迭代次数

        self.buffer = RolloutBuffer()  # 创建经验回放缓冲区

        # 创建当前策略网络
        self.policy = ActorCritic(state_dim, action_dim, action_std_init)
        # 创建优化器，分别为演员和评论家网络设置不同的学习率
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        # 创建旧策略网络（用于收集数据）
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init)
        # 将当前策略的参数复制到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 均方误差损失函数，用于评论家网络的训练
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        """设置动作的标准差

        Args:
            new_action_std: 新的动作标准差
        """
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def select_action(self, state):
        """根据状态选择动作

        Args:
            state: 当前状态

        Returns:
            选择的动作（NumPy数组）
        """
        with torch.no_grad():  # 不计算梯度
            state = torch.FloatTensor(state)  # 将状态转换为PyTorch张量
            action, action_logprob = self.policy_old.act(state)  # 使用旧策略选择动作

        # 将状态、动作和对数概率保存到缓冲区
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.numpy()  # 返回NumPy数组格式的动作

    def update(self):
        """更新策略网络

        使用收集的经验数据更新策略网络参数

        Returns:
            bool: 是否成功更新了策略
        """
        # 检查缓冲区中是否有足够的数据
        if len(self.buffer.rewards) < 5:
            print("\n缓冲区中的数据不足，跳过更新。当前数据量:", len(self.buffer.rewards))
            return False

        # 使用蒙特卡洛方法估计回报（未来折扣奖励的总和）
        rewards = []
        discounted_reward = 0
        # 从后向前遍历奖励和终止状态
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:  # 如果是终止状态，重置折扣奖励
                discounted_reward = 0
            # 计算折扣奖励：当前奖励 + 折扣因子 * 未来奖励
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)  # 将折扣奖励插入到列表开头

        # 对奖励进行标准化，使其均值为0，标准差为1
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # 安全地计算标准化
        if len(rewards) > 1 and rewards.std() > 1e-7:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # 添加小值避免除零

        # 将列表转换为张量
        old_states = torch.stack(self.buffer.states, dim=0).detach()  # 旧状态
        old_actions = torch.stack(self.buffer.actions, dim=0).detach()  # 旧动作
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()  # 旧对数概率

        # 对策略进行K轮优化
        for _ in range(self.K_epochs):
            # 评估旧动作和状态值
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 使状态值张量的维度与奖励张量匹配
            state_values = torch.squeeze(state_values)

            # 确保维度匹配
            if state_values.shape != rewards.shape:
                # 如果维度不匹配，尝试调整
                if len(state_values.shape) == 0 and len(rewards.shape) == 1 and rewards.shape[0] == 1:
                    # 将标量转换为大小为1的张量
                    state_values = state_values.unsqueeze(0)
                else:
                    print(f"\n警告: 状态值形状 {state_values.shape} 与奖励形状 {rewards.shape} 不匹配")

            # 计算比率（新策略概率 / 旧策略概率）
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算代理损失（Surrogate Loss）
            advantages = rewards - state_values.detach()  # 优势函数（奖励 - 状态值）
            surr1 = ratios * advantages  # 第一项代理目标
            # 第二项代理目标（裁剪比率）
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # 计算最终的PPO裁剪目标函数
            # -min(surr1, surr2): 策略目标（取负是因为我们要最大化这个值）
            # 0.5*MSE: 价值函数损失
            # -0.01*dist_entropy: 熵正则化项，鼓励探索
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # 执行梯度下降步骤
            self.optimizer.zero_grad()  # 清零梯度
            loss.mean().backward()       # 反向传播

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

            self.optimizer.step()        # 更新参数

        # 将新策略的权重复制到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空缓冲区
        self.buffer.clear()

        return True

    def save(self, checkpoint_path):
        """保存模型参数

        Args:
            checkpoint_path: 模型保存路径
        """
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        """加载模型参数

        Args:
            checkpoint_path: 模型加载路径
        """
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class DummyVecEnv:
    """向量化环境包装器

    一个简单的DummyVecEnv实现，模仿了stable-baselines3的接口
    将多个环境实例包装成一个向量化环境，以提高计算效率
    """
    def __init__(self, env_fns):
        """初始化向量化环境

        Args:
            env_fns: 环境创建函数列表
        """
        # 创建多个环境实例
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(env_fns)  # 环境数量

        # 从第一个环境获取观测空间和动作空间
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        """重置所有环境

        Returns:
            obs_list: 观测值列表
            info_list: 信息列表
        """
        obs_list = []  # 观测值列表
        info_list = []  # 信息列表
        # 遍历所有环境并重置
        for env in self.envs:
            obs, info = env.reset()  # 重置环境
            obs_list.append(obs)     # 添加观测值
            info_list.append(info)    # 添加信息
        return np.array(obs_list), info_list  # 返回观测值数组和信息列表

    def step(self, actions):
        """在所有环境中执行动作

        Args:
            actions: 动作列表，每个环境对应一个动作

        Returns:
            obs_list: 观测值数组
            reward_list: 奖励数组
            done_list: 完成标志数组
            truncated_list: 截断标志数组
            info_list: 信息列表
        """
        # 初始化结果列表
        obs_list, reward_list, done_list, truncated_list, info_list = [], [], [], [], []
        # 遍历所有环境并执行动作
        for i, env in enumerate(self.envs):
            # 在环境中执行动作
            obs, reward, done, truncated, info = env.step(actions[i])
            # 添加结果到列表
            obs_list.append(obs)           # 观测值
            reward_list.append(reward)     # 奖励
            done_list.append(done)         # 完成标志
            truncated_list.append(truncated)  # 截断标志
            info_list.append(info)         # 信息
        # 返回所有结果
        return np.array(obs_list), np.array(reward_list), np.array(done_list), np.array(truncated_list), info_list

    def render(self):
        """渲染环境

        只渲染第一个环境

        Returns:
            渲染结果
        """
        return self.envs[0].render()  # 返回第一个环境的渲染结果
