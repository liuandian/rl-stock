# -*- coding: utf-8 -*-

'''
强化学习交易环境

该模块实现了一个基于gymnasium的强化学习交易环境，用于股票交易的模拟和训练。
主要功能包括：
1. 模拟股票交易环境，包括买入、卖出操作和手续费计算
2. 提供状态表示、动作执行和奖励计算
3. 支持向量化环境，用于并行训练
4. 提供交易记录和性能评估
'''

# 导入必要的库
from typing import Any, List, Tuple  # 类型提示
import numpy as np                   # 数值计算
import pandas as pd                  # 数据处理
import random                        # 随机数生成
from copy import deepcopy            # 深拷贝
import gymnasium as gym              # 强化学习环境库
from gymnasium import spaces         # 环境空间定义
import time                          # 时间处理

# 导入向量化环境工具
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

class StockLearningEnv(gym.Env):
    """构建强化学习交易环境

    该类实现了一个基于gymnasium的强化学习交易环境，用于股票交易的模拟和训练。
    环境支持多只股票的交易，考虑了交易手续费，并提供了灵活的状态表示和奖励计算。

    Attributes:
        df (pd.DataFrame): 构建环境时所需要用到的行情数据，包含日期、股票代码、价格等信息
        buy_cost_pct (float): 买入股票时的手续费比例
        sell_cost_pct (float): 卖出股票时的手续费比例
        date_col_name (str): 日期列的名称
        hmax (int): 单次交易最大可交易的数量
        print_verbosity (int): 打印信息的频率，每隔多少步打印一次
        initial_amount (float): 初始资金量
        daily_information_cols (List[str]): 构建状态时所考虑的列，如开盘价、收盘价等
        cache_indicator_data (bool): 是否把数据预先加载到内存中以提高速度
        random_start (bool): 是否随机位置开始交易（训练环境通常为True，回测环境为False）
        patient (bool): 是否在资金不够时不执行交易操作，等到有足够资金时再执行
        currency (str): 货币单位，用于显示
    """

    metadata = {"render_modes": ["human"]}  # gymnasium中使用render_modes替代render.modes
    def __init__(
        self,
        df: pd.DataFrame,
        buy_cost_pct: float = 3e-3,
        sell_cost_pct: float = 3e-3,
        date_col_name: str = "date",
        hmax: int = 10,
        print_verbosity: int = 10,
        initial_amount: int = 1e6,
        daily_information_cols: List = ["open", "close", "high", "low", "volume"],
        cache_indicator_data: bool = True,
        random_start: bool = True,
        patient: bool = False,
        currency: str = "￥"
    ) -> None:
        """
        初始化交易环境

        Args:
            df: 包含股票数据的DataFrame，必须包含date_col_name和stock_col列
            buy_cost_pct: 买入股票时的手续费比例，默认为0.3%
            sell_cost_pct: 卖出股票时的手续费比例，默认为0.3%
            date_col_name: 日期列的名称，默认为"date"
            hmax: 单次交易最大可交易的数量，默认为10
            print_verbosity: 打印信息的频率，默认为每10步打印一次
            initial_amount: 初始资金量，默认为1,000,000
            daily_information_cols: 构建状态时所考虑的列，默认为["open", "close", "high", "low", "volume"]
            cache_indicator_data: 是否预先加载数据到内存，默认为True
            random_start: 是否随机位置开始交易，默认为True
            patient: 是否在资金不够时不执行交易操作，默认为False
            currency: 货币单位，默认为"￥"
        """
        self.df = df
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start
        self.patient = patient
        self.currency = currency
        self.df = self.df.set_index(date_col_name)
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.daily_information_cols = daily_information_cols
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.turbulence = 0
        self.episode = -1
        self.episode_history = []
        self.printed_header = False
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        self.max_total_assets = 0
        if self.cache_indicator_data:
            """cashing data 的结构:
               [[date1], [date2], [date3], ...]
               date1 : [stock1 * cols, stock2 * cols, ...]
            """
            print("加载数据缓存")
            self.cached_data = [
                self.get_date_vector(i) for i, _ in enumerate(self.dates)
            ]
            print("数据缓存成功!")

    def seed(self, seed: Any = None) -> None:
        """
        设置随机种子，确保实验可重复性

        Args:
            seed: 随机种子，如果为None则使用当前时间戳作为种子
        """
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)

    @property
    def current_step(self) -> int:
        """
        当前回合的运行步数

        Returns:
            int: 当前回合已经执行的步数
        """
        return self.date_index - self.starting_point

    @property
    def cash_on_hand(self) -> float:
        """
        当前拥有的现金

        Returns:
            float: 当前账户中的现金数量
        """
        return self.state_memory[-1][0]

    @property
    def holdings(self) -> List:
        """
        当前的持仓数据

        Returns:
            List: 当前持有的每只股票的数量列表
        """
        return self.state_memory[-1][1: len(self.assets) + 1]

    @property
    def closings(self) -> List:
        """
        每支股票当前的收盘价

        Returns:
            List: 当前日期每只股票的收盘价列表
        """
        return np.array(self.get_date_vector(self.date_index, cols=["close"]))

    def get_date_vector(self, date: int, cols: List = None) -> List:
        """
        获取指定日期的行情数据

        Args:
            date: 日期索引，对应self.dates中的索引
            cols: 需要获取的列名列表，如果为None则使用self.daily_information_cols

        Returns:
            List: 指定日期所有股票的行情数据，按股票和列名展平
        """
        if(cols is None) and (self.cached_data is not None):
            return self.cached_data[date]
        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[[date]]
            res = []
            for asset in self.assets:
                tmp_res = trunc_df[trunc_df[self.stock_col] == asset]
                res += tmp_res.loc[date, cols].tolist()
            assert len(res) == len(self.assets) * len(cols)
            return res

    def reset(self, *, seed=None, options=None) -> tuple:
        """
        重置环境到初始状态

        该方法在每个回合开始时调用，重置环境状态，包括现金、持仓、日期等。
        如果random_start为True，则会随机选择一个开始日期。

        Args:
            seed: 随机种子，用于确保实验可重复性
            options: 其他重置选项，目前未使用

        Returns:
            tuple: (初始状态, 信息字典)
        """
        # 调用父类的reset方法，设置随机种子
        super().reset(seed=seed, options=options)
        self.seed(seed)

        # 重置交易相关变量
        self.sum_trades = 0  # 总交易次数
        self.max_total_assets = self.initial_amount  # 最大资产价值，用于计算回撤

        # 设置起始日期
        if self.random_start:
            # 随机选择前半段数据作为起始点（用于训练）
            self.starting_point = random.choice(range(int(len(self.dates) * 0.5)))
        else:
            # 从第一天开始（用于回测）
            self.starting_point = 0
        self.date_index = self.starting_point

        # 重置其他状态变量
        self.turbulence = 0  # 市场波动指标
        self.episode += 1    # 回合计数器

        # 重置记忆数组
        self.actions_memory = []      # 动作历史
        self.transaction_memory = []  # 交易历史
        self.state_memory = []        # 状态历史

        # 重置账户信息
        self.account_information = {
            "cash": [],          # 现金历史
            "asset_value": [],   # 资产价值历史
            "total_assets": [],  # 总资产历史
            "reward": []         # 奖励历史
        }

        # 创建初始状态：[现金, 持仓1, 持仓2, ..., 技术指标1, 技术指标2, ...]
        init_state = np.array(
            [self.initial_amount]  # 初始现金
            + [0] * len(self.assets)  # 初始持仓全为0
            + self.get_date_vector(self.date_index)  # 初始日期的技术指标
        )
        self.state_memory.append(init_state)

        # 返回初始状态和空的信息字典
        return init_state, {}

    def log_step(
        self, reason: str, terminal_reward: float=None
        ) -> None:
        """
        记录并打印当前步骤的信息

        Args:
            reason: 记录原因，如"update"或"CASH SHORTAGE"等
            terminal_reward: 终止时的奖励，如果不是终止状态则为None
        """
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]

        assets = self.account_information["total_assets"][-1]
        tmp_retreat_ptc = assets / self.max_total_assets - 1
        retreat_pct = tmp_retreat_ptc if assets < self.max_total_assets else 0
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount

        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(assets))}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{retreat_pct*100:0.2f}%"
        ]
        self.episode_history.append(rec)
        print(self.template.format(*rec))

    def return_terminal(
        self, reason: str = "Last Date", reward: int = 0
    ) -> Tuple[List, float, bool, bool, dict]:
        """
        处理环境终止时的操作

        当环境达到终止条件时调用此方法，记录最终状态，计算最终奖励，并返回结果。

        Args:
            reason: 终止原因，默认为"Last Date"（达到最后一天）
            reward: 终止时的额外奖励，默认为0

        Returns:
            Tuple: (最终状态, 最终奖励, 终止标志, 截断标志, 信息字典)
        """
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount

        # 创建信息字典来存储各种指标，替代使用logger.record
        info = {
            "environment/GainLoss_pct": (gl_pct - 1) * 100,
            "environment/total_assets": int(self.account_information["total_assets"][-1]),
            "environment/total_reward_pct": (gl_pct - 1) * 100,
            "environment/total_trades": self.sum_trades,
            "environment/avg_daily_trades": self.sum_trades / (self.current_step)
        }

        return state, reward, True, False, info

    def log_header(self) -> None:
        """
        打印日志的表头

        在第一次调用log_step之前打印表头，包括回合、步数、终止原因等列名。
        """
        if not self.printed_header:
            self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"
            # 0, 1, 2, ... 是序号
            # 4, 4, 15, ... 是占位格的大小
            print(
                self.template.format(
                    "EPISODE",
                    "STEPS",
                    "TERMINAL_REASON",
                    "CASH",
                    "TOT_ASSETS",
                    "TERMINAL_REWARD",
                    "GAINLOSS_PCT",
                    "RETREAT_PROPORTION"
                )
            )
            self.printed_header = True

    def get_reward(self) -> float:
        """
        计算当前步骤的奖励值

        奖励由两部分组成：
        1. 总资产相对于初始资金的收益率
        2. 当前资产相对于历史最高资产的回撤惩罚

        Returns:
            float: 计算得到的奖励值
        """
        if self.current_step == 0:
            return 0
        else:
            assets = self.account_information["total_assets"][-1]
            retreat = 0
            if assets >= self.max_total_assets:
                self.max_total_assets = assets
            else:
                retreat = assets / self.max_total_assets - 1
            reward = assets / self.initial_amount - 1
            reward += retreat
            return reward

    def get_transactions(self, actions: np.ndarray) -> np.ndarray:
        """
        根据动作计算实际交易的股数

        将模型输出的动作转换为实际交易的股票数量，考虑了最大交易限制和当前持仓限制。

        Args:
            actions: 模型输出的动作，范围为[-1, 1]，表示买卖的比例

        Returns:
            np.ndarray: 实际交易的股票数量，正数表示买入，负数表示卖出
        """
        self.actions_memory.append(actions)
        actions = actions * self.hmax

        # 收盘价为 0 的不进行交易
        actions = np.where(self.closings > 0, actions, 0)

        # 去除被除数为 0 的警告
        out = np.zeros_like(actions)
        zero_or_not = self.closings != 0
        actions = np.divide(actions, self.closings, out=out, where = zero_or_not)

        # 不能卖的比持仓的多
        actions = np.maximum(actions, -np.array(self.holdings))

        # 将 -0 的值全部置为 0
        actions[actions == -0] = 0

        return actions

    def step(self, actions: np.ndarray) -> Tuple[List, float, bool, bool, dict]:
        """
        执行一步交易

        根据给定的动作执行一步交易，更新环境状态，并返回新状态、奖励和是否终止的信息。

        Args:
            actions: 模型输出的动作，范围为[-1, 1]，表示买卖的比例

        Returns:
            Tuple: (新状态, 奖励, 终止标志, 截断标志, 信息字典)
        """
        self.sum_trades += np.sum(np.abs(actions))
        self.log_header()
        if(self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        if self.date_index == len(self.dates) - 1:
            state, reward, terminated, info = self.return_terminal(reward=self.get_reward())
            return state, reward, terminated, False, info
        else:
            begin_cash = self.cash_on_hand
            assert min(self.holdings) >= 0
            assert_value = np.dot(self.holdings, self.closings)
            self.account_information["cash"].append(begin_cash)
            self.account_information["asset_value"].append(assert_value)
            self.account_information["total_assets"].append(begin_cash + assert_value)
            reward = self.get_reward()
            self.account_information["reward"].append(reward)

            transactions = self.get_transactions(actions)
            sells = -np.clip(transactions, -np.inf, 0)
            proceeds = np.dot(sells, self.closings)
            costs = proceeds * self.sell_cost_pct
            coh = begin_cash + proceeds # 计算现金的数量

            buys = np.clip(transactions, 0, np.inf)
            spend = np.dot(buys, self.closings)
            costs += spend * self.buy_cost_pct

            if (spend + costs) > coh: # 如果买不起
                if self.patient:
#                     self.log_step(reason="CASH SHORTAGE")
                    transactions = np.where(transactions > 0, 0, transactions)
                    spend = 0
                    costs = 0
                else:
                    return self.return_terminal(
                        reason="CASH SHORTAGE", reward=self.get_reward()
                    )
            self.transaction_memory.append(transactions)
            assert (spend + costs) <= coh
            coh = coh - spend - costs
            holdings_updated = self.holdings + transactions
            self.date_index += 1
            state = (
                [coh] + list(holdings_updated) + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(state)
            return state, reward, False, False, {}

    def get_sb_env(self) -> Tuple[Any, Any]:
        """
        获取stable-baselines3兼容的向量化环境

        创建一个DummyVecEnv包装的环境，用于stable-baselines3库的训练。

        Returns:
            Tuple: (向量化环境, 初始观察)
        """
        def get_self():
            return deepcopy(self)

        e = DummyVecEnv([get_self])
        reset_result = e.reset()
        # 处理不同版本的reset返回格式
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        return e, obs

    def get_multiproc_env(
        self, n: int = 10
    ) -> Tuple[Any, Any]:
        """
        获取多进程向量化环境

        创建一个SubprocVecEnv包装的多进程环境，用于加速训练。

        Args:
            n: 并行环境的数量，默认为10

        Returns:
            Tuple: (向量化环境, 初始观察)
        """
        def get_self():
            return deepcopy(self)

        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        reset_result = e.reset()
        # 处理不同版本的reset返回格式
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        return e, obs

    def save_asset_memory(self) -> pd.DataFrame:
        """
        保存账户资产历史记录

        将账户资产历史记录转换为DataFrame格式，包括日期、现金、资产价值等信息。

        Returns:
            pd.DataFrame: 账户资产历史记录，如果当前步数为0则返回None
        """
        if self.current_step == 0:
            return None
        else:
            # 添加日期信息
            self.account_information["date"] = self.dates[
                -len(self.account_information["cash"]):
            ]
            return pd.DataFrame(self.account_information)

    def save_action_memory(self) -> pd.DataFrame:
        """
        保存动作和交易历史记录

        将动作和交易历史记录转换为DataFrame格式，包括日期、动作和实际交易数量。

        Returns:
            pd.DataFrame: 动作和交易历史记录，如果当前步数为0则返回None
        """
        if self.current_step == 0:
            return None
        else:
            return pd.DataFrame(
                {
                    "date": self.dates[-len(self.account_information["cash"]):],
                    "actions": self.actions_memory,        # 模型输出的动作
                    "transactions": self.transaction_memory # 实际交易的股票数量
                }
            )
