# -*- coding: utf-8 -*-

'''
强化学习交易模型

该模块包含了用于创建、训练和评估强化学习交易模型的类和函数。
主要功能包括：
1. 创建不同类型的强化学习模型（A2C, DDPG, PPO, TD3, SAC）
2. 训练模型
3. 使用训练好的模型进行回测
'''

# 导入必要的库
from typing import Any  # 类型提示
import pandas as pd     # 数据处理
import numpy as np      # 数值计算
import time             # 时间处理

# 导入强化学习算法
from stable_baselines3 import DDPG  # 深度确定性策略梯度算法
from stable_baselines3 import A2C   # Advantage Actor-Critic算法
from stable_baselines3 import PPO   # 近端策略优化算法
from stable_baselines3 import TD3   # 双延迟深度确定性策略梯度算法
from stable_baselines3 import SAC   # 软演员-评论家算法
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise  # 动作噪声

# 导入项目内部模块
from utils import config                # 配置参数
from utils.preprocessors import split_data  # 数据预处理
from utils.env import StockLearningEnv  # 强化学习交易环境

# 定义可用的模型字典
MODELS = {
    "a2c": A2C,    # Advantage Actor-Critic算法
    "ddpg": DDPG, # 深度确定性策略梯度算法
    "td3": TD3,   # 双延迟深度确定性策略梯度算法
    "sac": SAC,   # 软演员-评论家算法
    "ppo": PPO    # 近端策略优化算法
}

# 从配置文件中获取每个模型的默认超参数
MODEL_KWARGS = {x: config.__dict__["{}_PARAMS".format(x.upper())] for x in MODELS.keys()}

# 定义可用的动作噪声类型
NOISE = {
    "normal": NormalActionNoise,                # 正态分布噪声
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise  # Ornstein-Uhlenbeck过程噪声
}

class DRL_Agent():
    """强化学习交易智能体

    该类封装了强化学习交易智能体的创建、训练和评估功能。
    可以创建不同类型的强化学习模型，训练模型，并使用训练好的模型进行回测。

    Attributes:
        env: 强化学习环境，用于模型训练和评估
    """

    @staticmethod
    def DRL_prediction(
        model: Any, environment: Any
        ) -> pd.DataFrame:
        """使用训练好的模型进行回测

        该方法使用训练好的强化学习模型在给定环境中进行回测，
        记录每个时间步的账户价值和执行的动作。

        Args:
            model: 训练好的强化学习模型
            environment: 回测环境

        Returns:
            tuple: (账户价值记录, 动作记录)，两个都是pandas DataFrame
        """
        test_env, test_obs = environment.get_sb_env()

        account_memory = []
        actions_memory = []
        test_env.reset()

        len_environment = len(environment.df.index.unique())
        for i in range(len_environment):
            action, _states = model.predict(test_obs)
            test_obs, _, dones, _ = test_env.step(action)
            if i == (len_environment - 2):
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("回测完成!")
                break
        return account_memory[0], actions_memory[0]

    def __init__(self, env: Any) -> None:
        """初始化强化学习交易智能体

        Args:
            env: 强化学习环境，用于模型训练和评估
        """
        self.env = env

    def get_model(
        self,
        model_name: str,
        policy: str = "MlpPolicy",
        policy_kwargs: dict = None,
        model_kwargs: dict = None,
        verbose: int = 1
    ) -> Any:
        """根据超参数创建强化学习模型

        根据指定的模型类型和超参数创建一个强化学习模型。

        Args:
            model_name: 模型名称，必须是MODELS字典中的一个键
            policy: 策略网络类型，默认为"MlpPolicy"（多层感知机策略）
            policy_kwargs: 策略网络的额外参数
            model_kwargs: 模型的额外参数
            verbose: 详细程度，控制训练过程中的输出信息量

        Returns:
            创建好的强化学习模型

        Raises:
            NotImplementedError: 如果model_name不在MODELS字典中
        """
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)

        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log="{}/{}".format(config.TENSORBOARD_LOG_DIR, model_name),
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs
        )

        return model

    def train_model(
        self, model: Any, tb_log_name: str, total_timesteps: int = 5000
        ) -> Any:
        """训练强化学习模型

        使用指定的参数训练强化学习模型。

        Args:
            model: 要训练的强化学习模型
            tb_log_name: TensorBoard日志名称
            total_timesteps: 总训练步数

        Returns:
            训练好的强化学习模型
        """
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
        return model

# 当模块被直接运行时执行的代码
if __name__ == "__main__":
    from pull_data import Pull_data
    from preprocessors import FeatureEngineer, split_data
    from utils import config
    import time

    # 拉取数据
    print("正在拉取数据...")
    df = Pull_data(config.SSE_50[:2], save_data=False).pull_data()
    print("正在处理数据...")
    df = FeatureEngineer().preprocess_data(df)
    df = split_data(df, '2009-01-01','2019-01-01')
    print("数据处理完成，前5行数据:")
    print(df.head())

    # 处理环境参数
    print("\n设置环境参数...")
    stock_dimension = len(df.tic.unique()) # 股票数量
    state_space = 1 + 2*stock_dimension + \
        len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension # 状态空间维度
    print("股票数量: {}, 状态空间维度: {}".format(stock_dimension, state_space))
    env_kwargs = {
        "stock_dim": stock_dimension,       # 股票数量
        "hmax": 100,                       # 单次交易最大数量
        "initial_amount": 1e6,             # 初始资金
        "buy_cost_pct": 0.001,             # 买入手续费
        "sell_cost_pct": 0.001,            # 卖出手续费
        "reward_scaling": 1e-4,            # 奖励缩放因子
        "state_space": state_space,        # 状态空间维度
        "action_space": stock_dimension,   # 动作空间维度
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST  # 技术指标列表
    }

    # 创建并测试环境
    print("\n创建交易环境...")
    e_train_gym = StockLearningEnv(df = df, **env_kwargs)

    ### 单步测试（已注释）
    # observation = e_train_gym.reset()
    # print("重置后的观察值: ", observation)
    # action = e_train_gym.action_space.sample()
    # print("随机动作: ", action)
    # observation_later, reward, done, _ = e_train_gym.step(action)
    # print("执行动作后的观察值: ", observation_later)
    # print("奖励: {}, 结束标志: {}".format(reward, done))

    ### 多步测试
    print("\n进行多步测试...")
    observation, _ = e_train_gym.reset()       # 初始化环境，observation为环境状态
    count = 0
    for t in range(10):
        action = e_train_gym.action_space.sample()  # 随机采样动作
        observation, reward, terminated, truncated, info = e_train_gym.step(action)  # 与环境交互，获得下一个状态
        done = terminated or truncated  # 组合终止和截断标志
        if done:
            break
        count+=1
        time.sleep(0.2)      # 每次等待0.2秒
    print("最终观察值: ", observation)
    print("最终奖励: {}, 结束标志: {}".format(reward, done))

    # 测试模型创建和训练
    print("\n准备模型训练...")
    env_train, _ = e_train_gym.get_sb_env()
    print("环境类型:", type(env_train))

    # 创建智能体和模型
    print("\n创建SAC模型...")
    agent = DRL_Agent(env= env_train)
    SAC_PARAMS = {
        "batch_size": 128,         # 批量大小
        "buffer_size": 1000000,    # 经验回放缓冲区大小
        "learning_rate": 0.0001,   # 学习率
        "learning_starts": 100,    # 开始学习前的步数
        "ent_coef": "auto_0.1"     # 熵系数（自动调整）
    }
    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

    # 训练模型
    print("\n开始训练模型...")
    trained_sac = agent.train_model(
        model=model_sac,
        tb_log_name='sac',         # TensorBoard日志名称
        total_timesteps= 50000     # 总训练步数
    )
    print("模型训练完成!")