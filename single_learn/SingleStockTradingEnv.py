# 导入必要的库
import random                # 随机数生成库
import json                  # JSON处理库
import gymnasium as gym      # 强化学习环境库
from gymnasium import spaces # 环境空间定义
import pandas as pd          # 数据分析库
import numpy as np           # 科学计算库

# 定义各种限制参数
MAX_ACCOUNT_BALANCE = 2147483647  # 最大账户余额
MAX_NUM_SHARES = 2147483647       # 最大股票数量
MAX_SHARE_PRICE = 5000            # 最大股票价格
MAX_VOLUME = 1000e8                # 最大交易量
MAX_AMOUNT = 3e10                  # 最大交易金额
MAX_OPEN_POSITIONS = 5             # 最大持仓仓位数
MAX_STEPS = 20000                  # 最大步数
MAX_DAY_CHANGE = 1                 # 最大日变化率

INITIAL_ACCOUNT_BALANCE = 10000    # 初始账户余额
TRANSACTION_FEE_PERCENT = 0.001    # 交易手续费百分比（0.1%）


class StockTradingEnv(gym.Env):
    """股票交易环境，用于强化学习

    该环境模拟股票交易过程，允许代理进行买入、卖出和持有操作
    """
    metadata = {'render.modes': ['human']}  # 渲染模式元数据

    def __init__(self, df):
        """初始化股票交易环境

        Args:
            df: 包含股票数据的DataFrame，应包含开盘价、最高价、最低价、收盘价等列
        """
        super(StockTradingEnv, self).__init__()

        self.df = df  # 股票数据

        # # 验证数据完整性
        # required_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'adjustflag',
        #                 'tradestatus', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM']
        # missing_cols = [col for col in required_columns if col not in df.columns]
        # if missing_cols:
        #     raise ValueError(f"缺少必要列: {missing_cols}")
        # if df[required_columns].isnull().any().any():
        #     print("警告：数据包含NaN值")
        #     print(df[required_columns].isnull().sum())
        #     self.df = df.fillna(method='ffill').fillna(0)
        # if np.isinf(df[required_columns].select_dtypes(include=np.number)).any().any():
        #     print("警告：数据包含无穷大值")
        # self.df = df.replace([np.inf, -np.inf], 0)

        self.reward_range = (0, MAX_ACCOUNT_BALANCE)  # 奖励范围

        # 动作空间定义：一维连续空间，范围为[-1, 1]
        # 负值表示卖出，正值表示买入，值的绝对大小表示交易比例
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)

        # 观测空间定义：包含19个特征（价格、交易量、账户状态等）
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(19,), dtype=np.float32)
        
        self.returns_history = []
        self.window_size = 20  # 滑动窗口
        
    def _next_observation(self):
        """获取当前状态的观测值

        将股票数据和账户状态转换为观测值数组，并进行归一化处理

        Returns:
            归一化后的观测值数组
        """

    # 获取原始值并处理NaN
        peTTM = self.df.loc[self.current_step, 'peTTM']
        pbMRQ = self.df.loc[self.current_step, 'pbMRQ']
        psTTM = self.df.loc[self.current_step, 'psTTM']

        # 如果值是NaN，替换为0
        peTTM = 0 if pd.isna(peTTM) else peTTM
        pbMRQ = 0 if pd.isna(pbMRQ) else pbMRQ
        psTTM = 0 if pd.isna(psTTM) else psTTM


        obs = np.array([
            # 股票价格数据
            self.df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE,    # 开盘价
            self.df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE,    # 最高价
            self.df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE,     # 最低价
            self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,   # 收盘价
            self.df.loc[self.current_step, 'volume'] / MAX_VOLUME,       # 交易量
            self.df.loc[self.current_step, 'amount'] / MAX_AMOUNT,       # 交易金额
            self.df.loc[self.current_step, 'adjustflag'] / 10,           # 复权标志
            self.df.loc[self.current_step, 'tradestatus'] / 1,           # 交易状态
            self.df.loc[self.current_step, 'pctChg'] / 100,              # 涨跌幅
            peTTM / 1e4,
            pbMRQ / 100,
            psTTM / 100,
            self.df.loc[self.current_step, 'pctChg'] / 1e3,              # 涨跌幅（再次归一化）
            # 账户状态数据
            self.balance / MAX_ACCOUNT_BALANCE,                          # 账户余额
            self.max_net_worth / MAX_ACCOUNT_BALANCE,                    # 最大净资产
            self.shares_held / MAX_NUM_SHARES,                           # 持有股票数量
            self.cost_basis / MAX_SHARE_PRICE,                           # 持有股票的平均成本
            self.total_shares_sold / MAX_NUM_SHARES,                     # 已卖出的股票总数
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE), # 已卖出股票的总价值
        ])
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"第{self.current_step}步观测值无效: {obs}")
            raise ValueError("观测值包含NaN或inf")
        return obs

    def _take_action(self, action):
        """根据动作执行交易

        根据给定的动作，执行买入或卖出操作，并更新账户状态

        Args:
            action: 动作值，一维数组，范围为[-1, 1]，负值表示卖出，正值表示买入
        """
        # 在当前时间步内随机选择一个价格作为交易价格
        current_price = random.uniform(
            self.df.loc[self.current_step, "open"],
            self.df.loc[self.current_step, "close"])
        if not np.isfinite(current_price) or current_price <= 0:
            raise ValueError(f"无效价格: {current_price}")
        # 提取动作值（去除数组包装）
        action_value = action[0]  # 动作值，范围为[-1, 1]

        # 计算交易比例（动作的绝对值）
        amount = abs(action_value)

        if action_value > 0:  # 正值表示买入
            # 买入操作：使用账户余额的amount比例购买股票
            total_possible = int(self.balance / current_price)  # 最大可购买股票数量
            shares_bought = int(total_possible * amount)         # 实际购买股票数量
            prev_cost = self.cost_basis * self.shares_held       # 原有股票的总成本
            additional_cost = shares_bought * current_price       # 新购股票的总成本

            # 计算交易手续费
            transaction_fee = additional_cost * TRANSACTION_FEE_PERCENT
            self.transaction_fees += transaction_fee  # 累计交易手续费

            # 从账户余额中扣除购买费用和手续费
            self.balance -= (additional_cost + transaction_fee)

            # 计算新的平均成本
            total_shares = self.shares_held + shares_bought
            if total_shares > 0:
                self.cost_basis = (prev_cost + additional_cost) / total_shares
            else:
                self.cost_basis = 0
            self.shares_held += shares_bought  # 更新持有股票数量

        elif action_value < 0:  # 负值表示卖出
            # 卖出操作：卖出持有股票的amount比例
            shares_sold = int(self.shares_held * amount)  # 要卖出的股票数量
            sale_amount = shares_sold * current_price      # 卖出总金额

            # 计算交易手续费
            transaction_fee = sale_amount * TRANSACTION_FEE_PERCENT
            self.transaction_fees += transaction_fee  # 累计交易手续费

            # 将卖出收入（减去手续费）添加到账户余额
            self.balance += (sale_amount - transaction_fee)
            self.shares_held -= shares_sold                # 减少持有股票数量
            self.total_shares_sold += shares_sold          # 增加已卖出股票总数
            self.total_sales_value += sale_amount          # 增加已卖出股票总价值
        # 如果动作值接近于0，则不执行交易（持有）

        # 计算新的净资产（账户余额 + 持有股票价值）
        self.net_worth = self.balance + self.shares_held * current_price
        if not np.isfinite(self.net_worth):
            raise ValueError(f"无效净值: {self.net_worth}")
        # 更新最大净资产
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # 如果没有持有股票，将平均成本重置为0
        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        """执行一个环境时间步

        根据给定的动作执行交易，并返回新的状态、奖励和完成标志

        Args:
            action: 动作值，一维数组，范围为[-1, 1]，负值表示卖出，正值表示买入

        Returns:
            obs: 新的观测值
            reward: 奖励值
            done: 是否结束
            truncated: 是否被截断
            info: 额外信息字典
        """
        # 执行动作
        self._take_action(action)
        done = False      # 完成标志
        truncated = False # 截断标志

        # 移动到下一个时间步
        self.current_step += 1

        # 如果到达数据的结尾，循环回到开始
        if self.current_step > len(self.df) - 1:
            self.current_step = 0  # 循环训练
            # done = True  # 可选：在数据结束时结束环境（已注释）

        # 计算奖励
        # 1. 计算对数收益率: log(当前资产/上一时刻资产)
        log_return = 0
        if self.prev_net_worth > 0 and self.net_worth > 0:
            log_return = np.log(self.net_worth / self.prev_net_worth+10e-6)

        # 2. 计算与最高资产的差距: (当前资产/最高资产-1)
        max_net_worth_penalty = 0
        if self.max_net_worth > 0:
            max_net_worth_penalty = (self.net_worth / self.max_net_worth) - 1

        # 3. 计算交易手续费影响: 交易手续费/当前资产
        # 注意：手续费是负面影响，所以这里是减去手续费的影响
        transaction_fee_penalty = 0
        if self.net_worth > 0:
            transaction_fee_penalty = -self.transaction_fees / self.net_worth

        # 组合奖励
        reward = log_return*10 + max_net_worth_penalty*0.1 + transaction_fee_penalty*1
        
        # 计算收益率
        # returns = self.net_worth / self.prev_net_worth - 1
        # self.returns_history.append(returns)
        # if len(self.returns_history) > self.window_size:
        #     self.returns_history.pop(0)
        
        # # 计算夏普比率
        # mean_returns = np.mean(self.returns_history)
        # std_returns = np.std(self.returns_history) + 1e-6
        # sharpe_ratio = mean_returns / std_returns
        
        # # 综合奖励
        # reward = sharpe_ratio * 100 - self.transaction_fees
        # reward = np.clip(reward, -1000, 1000)  # 防止极端值        

        # 验证奖励值
        if not np.isfinite(reward):
            raise ValueError(f"无效奖励: {reward}")

        # 重置交易手续费（只计算当前步骤的手续费影响）
        self.transaction_fees = 0

        # 保存当前的净资产作为下一步的参考
        self.prev_net_worth = self.net_worth

        # 如果净资产小于等于0，结束环境
        if self.net_worth <= 0:
            done = True

        # 获取新的观测值
        obs = self._next_observation()

        # 返回状态、奖励、完成标志、截断标志和额外信息
        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        """重置环境到初始状态

        参数:
            seed: 随机种子，用于复现结果
            options: 重置的额外选项（可以包含新的数据集）

        返回:
            初始观测值和信息字典
        """
        # 如果提供了随机种子，进行设置
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 重置账户状态
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.prev_net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.transaction_fees = 0  # 交易手续费累计

        # 如果在选项中提供了新的数据集，则使用新数据集
        if options and "new_df" in options:
            self.df = options["new_df"]

        # 重置当前步数
        self.current_step = 0

        return self._next_observation(), {}

    def render(self):
        """渲染环境状态

        将当前环境状态打印到屏幕上，包括账户余额、持有股票、利润等信息

        Returns:
            profit: 当前利润（净资产 - 初始资产）
        """
        # 计算利润
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        # 打印分隔线
        print('-'*30)
        # 打印当前步数
        print(f'Step: {self.current_step}')
        # 打印账户余额
        print(f'Balance: {self.balance}')
        # 打印持有股票数量和已卖出的股票总数
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        # 打印持有股票的平均成本和已卖出股票的总价值
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        # 打印净资产和最大净资产
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        # 打印利润和交易手续费
        print(f'Profit: {profit}')
        print(f'Transaction fees: {self.transaction_fees}')
        # 返回利润
        return profit
