# 导入必要的库
import os                      # 操作系统相关功能，用于文件路径操作
import pickle                  # 用于序列化和反序列化Python对象
import pandas as pd            # 数据分析库，用于处理CSV文件和数据框
import numpy as np             # 科学计算库，用于数组和矩阵运算
import matplotlib.pyplot as plt # 绘图库，用于可视化结果
import matplotlib.font_manager as fm # 字体管理，用于支持中文显示

# 导入自定义模块
from SingleStockTradingEnv import StockTradingEnv, MAX_ACCOUNT_BALANCE, MAX_NUM_SHARES, MAX_SHARE_PRICE, TRANSACTION_FEE_PERCENT  # 股票交易环境及其常量

plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 导入强化学习算法
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

# 定义可用的模型字典
MODELS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
    "ppo": PPO
}
def stock_trade(stock_file, model_name="ppo", timesteps=int(1e5)):
    """使用指定的强化学习模型训练和测试股票交易策略

    Args:
        stock_file: 股票数据文件路径
        model_name: 使用的模型名称，可选值为 "a2c", "ddpg", "td3", "sac", "ppo"
        timesteps: 训练的时间步数

    Returns:
        day_profits: 每日收益率列表
        baseline_profits: 基准策略的收益率列表
    """
    day_profits = []
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')

    # 验证和预处理训练数据
    if df.isnull().any().any():
        print("警告：训练数据包含NaN值")
        print(df.isnull().sum())

        # 更智能地处理NaN值
        # 1. 对价格相关列使用前向填充，然后后向填充，避免使用零值
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                # 如果仍然有NaN，使用列的中位数
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())

        # 2. 对交易量和金额列，使用前向填充，然后使用当前市场平均值
        volume_cols = ['volume', 'amount', 'turn']
        for col in volume_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(method='ffill')
                # 如果仍然有NaN，使用该列的平均值（忽略异常值）
                if df[col].isnull().any():
                    # 计算非零非空值的平均值
                    mean_val = df[col][df[col] > 0].mean()
                    if pd.isna(mean_val):  # 如果平均值仍然为NaN，使用0
                        mean_val = 0
                    df[col] = df[col].fillna(mean_val)

        # 3. 对于估值比率列（PE、PB等），使用行业平均值或中位数
        ratio_cols = ['peTTM', 'pbMRQ', 'psTTM', 'pctChg']
        for col in ratio_cols:
            if col in df.columns and df[col].isnull().any():
                # 使用中位数而非平均值，因为这些比率通常有偏态分布
                median_val = df[col].median()
                if pd.isna(median_val):  # 如果中位数仍然为NaN
                    # 使用常见的行业平均值
                    default_values = {'peTTM': 15.0, 'pbMRQ': 1.5, 'psTTM': 1.0, 'pctChg': 0.0}
                    df[col] = df[col].fillna(default_values.get(col, 0))
                else:
                    df[col] = df[col].fillna(median_val)

    # 处理无穷大值
    if np.isinf(df.select_dtypes(include=np.number)).any().any():
        print("警告：训练数据包含无穷大值")
        # 对于无穷大值，替换为列的最大/最小有限值
        for col in df.select_dtypes(include=np.number).columns:
            if (df[col] == np.inf).any() or (df[col] == -np.inf).any():
                # 获取有限的最大和最小值
                finite_values = df[col][np.isfinite(df[col])]
                if len(finite_values) > 0:
                    max_val = finite_values.max()
                    min_val = finite_values.min()
                    # 替换无穷大和无穷小
                    df[col] = df[col].replace(np.inf, max_val * 1.5)
                    df[col] = df[col].replace(-np.inf, min_val * 1.5 if min_val < 0 else min_val * 0.5)
                else:
                    # 如果没有有限值，使用默认值
                    df[col] = df[col].replace([np.inf, -np.inf], 0)

    # 异常值检测和处理
    # 使用IQR方法检测异常值，并将其替换为上下限
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        # 将异常值限制在上下限范围内
        df[col] = df[col].clip(lower_bound, upper_bound)

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    # 选择指定的强化学习模型
    model_name = model_name.lower()
    if model_name not in MODELS:
        raise ValueError(f"不支持的模型: {model_name}。可用模型: {list(MODELS.keys())}")

    # 获取模型类
    ModelClass = MODELS[model_name]

    # 根据不同模型设置不同的参数
    if model_name == "ppo":
        model = ModelClass(
            "MlpPolicy",
            env,
            verbose=1,
            # tensorboard_log=f'./log/{model_name}',
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.01,
        )
    elif model_name == "a2c":
        model = ModelClass(
            "MlpPolicy",
            env,
            verbose=1,
            # tensorboard_log=f'./log/{model_name}',
            learning_rate=1e-4,
            n_steps=5,
            ent_coef=0.01,
        )
    elif model_name == "sac":
        model = ModelClass(
            "MlpPolicy",
            env,
            verbose=1,
            # tensorboard_log=f'./log/{model_name}',
            learning_rate=1e-4,
            buffer_size=10000,
            batch_size=64,
            ent_coef="auto",
        )
    elif model_name in ["ddpg", "td3"]:
        model = ModelClass(
            "MlpPolicy",
            env,
            verbose=1,
            # tensorboard_log=f'./log/{model_name}',
            learning_rate=1e-4,
            buffer_size=10000,
            batch_size=64,
        )

    # 训练模型
    model.learn(total_timesteps=timesteps,
                log_interval=1,
                progress_bar=True)

    df_test = pd.read_csv(stock_file.replace('train', 'test'))

    # 对测试数据使用与训练数据相同的预处理方法
    if df_test.isnull().any().any():
        print("警告：测试数据包含NaN值")
        print(df_test.isnull().sum())

        # 1. 对价格相关列使用前向填充，然后后向填充
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df_test.columns and df_test[col].isnull().any():
                df_test[col] = df_test[col].fillna(method='ffill').fillna(method='bfill')
                if df_test[col].isnull().any():
                    df_test[col] = df_test[col].fillna(df_test[col].median())

        # 2. 对交易量和金额列
        volume_cols = ['volume', 'amount', 'turn']
        for col in volume_cols:
            if col in df_test.columns and df_test[col].isnull().any():
                df_test[col] = df_test[col].fillna(method='ffill')
                if df_test[col].isnull().any():
                    mean_val = df_test[col][df_test[col] > 0].mean()
                    if pd.isna(mean_val):
                        mean_val = 0
                    df_test[col] = df_test[col].fillna(mean_val)

        # 3. 对于估值比率列
        ratio_cols = ['peTTM', 'pbMRQ', 'psTTM', 'pctChg']
        for col in ratio_cols:
            if col in df_test.columns and df_test[col].isnull().any():
                median_val = df_test[col].median()
                if pd.isna(median_val):
                    default_values = {'peTTM': 15.0, 'pbMRQ': 1.5, 'psTTM': 1.0, 'pctChg': 0.0}
                    df_test[col] = df_test[col].fillna(default_values.get(col, 0))
                else:
                    df_test[col] = df_test[col].fillna(median_val)

    # 处理无穷大值
    if np.isinf(df_test.select_dtypes(include=np.number)).any().any():
        print("警告：测试数据包含无穷大值")
        for col in df_test.select_dtypes(include=np.number).columns:
            if (df_test[col] == np.inf).any() or (df_test[col] == -np.inf).any():
                finite_values = df_test[col][np.isfinite(df_test[col])]
                if len(finite_values) > 0:
                    max_val = finite_values.max()
                    min_val = finite_values.min()
                    df_test[col] = df_test[col].replace(np.inf, max_val * 1.5)
                    df_test[col] = df_test[col].replace(-np.inf, min_val * 1.5 if min_val < 0 else min_val * 0.5)
                else:
                    df_test[col] = df_test[col].replace([np.inf, -np.inf], 0)

    # 异常值检测和处理
    for col in df_test.select_dtypes(include=np.number).columns:
        Q1 = df_test[col].quantile(0.25)
        Q3 = df_test[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df_test[col] = df_test[col].clip(lower_bound, upper_bound)

    # 计算基准策略（买入并持有）的收益
    # 假设在第一天以开盘价买入，最后一天以收盘价卖出
    initial_price = df_test.iloc[0]['open']
    baseline_profits = []
    initial_balance = 10000  # 与环境中的初始余额相同

    # 计算可以买入的股票数量（整数股）
    shares_bought = int(initial_balance / initial_price)
    remaining_balance = initial_balance - shares_bought * initial_price

    # 计算每一天的净资产和收益率
    for i in range(len(df_test)):
        current_price = df_test.iloc[i]['close']
        current_value = shares_bought * current_price + remaining_balance
        profit_rate = (current_value / initial_balance) - 1  # 计算收益率
        baseline_profits.append(profit_rate)

    # 强化学习策略的收益
    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()
    day_profits = []  # 重置收益列表


    for i in range(len(df_test) - 1):
        action, _ = model.predict(obs)
        next_obs, _, done, _ = env.step(action)

        # 从观测空间中提取账户状态数据
        # 观测空间索引参考 _next_observation 方法中的定义
        # 索引 13 是账户余额，14 是最大净资产，15 是持有股票数量
        balance = next_obs[0][13] * MAX_ACCOUNT_BALANCE  # 账户余额
        shares_held = next_obs[0][15] * MAX_NUM_SHARES  # 持有股票数量

        # 计算当前股票价格（使用收盘价）
        current_price = df_test.iloc[i+1]['close']

        # 计算当前净资产（账户余额 + 持有股票价值）
        current_net_worth = balance + shares_held * current_price

        # 收益率
        profit_rate = (current_net_worth / initial_balance) -1
        day_profits.append(profit_rate)

        # 更新观测值
        obs = next_obs

        if done:
            break
    print(day_profits)
    return day_profits, baseline_profits


def find_file(path, name):
    """在指定路径下查找包含特定名称的文件

    Args:
        path: 要搜索的路径
        name: 要查找的文件名称的一部分

    Returns:
        找到的文件的完整路径，如果没有找到则返回None
    """
    # print(path, name)  # 调试输出（已注释）
    for root, dirs, files in os.walk(path):  # 遍历指定路径下的所有文件和目录
        for fname in files:  # 遍历所有文件
            if name in fname:  # 如果文件名包含指定的名称
                return os.path.join(root, fname)  # 返回完整路径


def test_a_stock_trade(stock_code, model_name="ppo", timesteps=int(1e5)):
    """测试单只股票的交易策略并可视化结果

    Args:
        stock_code: 股票代码，如'sh.600036'
        model_name: 使用的模型名称，可选值为 "a2c", "ddpg", "td3", "sac", "ppo"
        timesteps: 训练的时间步数
    """
    # 查找股票训练数据文件
    stock_file = find_file('./stockdata/train', str(stock_code))

    # 训练模型并获取每日利润和基准利润
    daily_profits, baseline_profits = stock_trade(stock_file, model_name, timesteps)

    # 创建图表并绘制收益率曲线
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制强化学习策略的收益率曲线
    ax.plot(daily_profits, label=f'{model_name.upper()} Strategy ({stock_code})', marker='o', ms=8, alpha=0.7, mfc='orange', markevery=5)

    # 绘制基准策略（买入并持有）的收益率曲线，使用灰色
    ax.plot(baseline_profits, label='Baseline (Buy & Hold)', color='gray', linestyle='--', marker='s', ms=6, alpha=0.6, markevery=10)

    ax.grid()  # 添加网格
    plt.xlabel('Trading Steps')  # x轴标签（步数）
    plt.ylabel('Return Rate')  # y轴标签（收益率）
    plt.legend()  # 添加图例
    plt.title(f'{model_name.upper()} Trading Strategy vs Buy & Hold Strategy ({stock_code})')  # 添加标题
    plt.show()  # 显示图表


if __name__ == '__main__':
    # 程序入口点
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='股票交易强化学习模型')
    parser.add_argument('--model','-m', type=str, default='ppo', choices=list(MODELS.keys()),
                        help='强化学习模型 (默认: ppo)')
    parser.add_argument('--stock', type=str, default='sh.600036',
                        help='股票代码 (默认: sh.600073)')
    parser.add_argument('--timesteps','-tts', type=int, default=int(3e4),
                        help='训练时间步数 (默认: 30000)')

    # 解析命令行参数
    args = parser.parse_args()

    # 测试指定的股票和模型
    test_a_stock_trade(args.stock, args.model, args.timesteps)
