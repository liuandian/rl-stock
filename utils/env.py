from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import gymnasium as gym      # 强化学习环境库
from gymnasium import spaces # 环境空间定义
import time

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

class StockLearningEnv(gym.Env):
    """构建强化学习交易环境 (Environment)

    该类实现了一个基于gymnasium的强化学习交易环境，用于股票交易的模拟和训练。
    环境支持多只股票的交易，考虑了交易手续费，并提供了灵活的状态表示和奖励计算。

    在强化学习中，环境(Environment)是智能体(Agent)与之交互的外部系统。
    在这个股票交易环境中，智能体通过观察市场状态，执行买卖操作，并获得相应的奖励。
    环境负责根据智能体的动作更新状态，并提供反馈（奖励）。

    奖励函数 (Reward Function):
    奖励函数由三部分组成：
    1. 对数收益率 (Log Return)：使用对数收益率代替简单收益率，更符合金融理论
       - 计算公式：log(assets / initial_amount)
       - 对数收益率在连续复利情况下更准确，且可以更好地处理不同时间尺度的收益

    2. 回撤惩罚 (Drawdown Penalty)：惩罚资产价值从历史最高点的下跌
       - 计算公式：log(assets / max_total_assets)（这是一个负值或0）
       - 这鼓励智能体避免大幅度的资产回撤，追求稳定增长

    3. 交易成本惩罚 (Transaction Cost Penalty)：惩罚过度交易
       - 计算公式：-交易量 * 交易成本系数
       - 这鼓励智能体减少不必要的交易，避免过度交易导致的成本损失

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

    metadata = {"render.modes": ["human"]}
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
        """设置随机种子"""
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)

    @property
    def current_step(self) -> int:
        """当前回合的运行步数"""
        return self.date_index - self.starting_point

    @property
    def cash_on_hand(self) -> float:
        """当前拥有的现金"""
        return self.state_memory[-1][0]

    @property
    def holdings(self) -> List:
        """当前的持仓数据"""
        return self.state_memory[-1][1: len(self.assets) + 1]

    @property
    def closings(self) -> List:
        """每支股票当前的收盘价"""
        return np.array(self.get_date_vector(self.date_index, cols=["close"]))

    def get_date_vector(self, date: int, cols: List = None) -> List:
        """获取 date 那天的行情数据"""
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
        super().reset(seed=seed, options=options)
        self.seed(seed)
        self.sum_trades = 0
        self.max_total_assets = self.initial_amount
        if self.random_start:
            self.starting_point = random.choice(range(int(len(self.dates) * 0.5)))
        else:
            self.starting_point = 0
        self.date_index = self.starting_point
        self.turbulence = 0
        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": []
        }
        init_state = np.array(
            [self.initial_amount]
            + [0] * len(self.assets)
            + self.get_date_vector(self.date_index)
        )
        self.state_memory.append(init_state)
        return init_state, {}  # 返回初始状态和空的info字典

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

        # 获取当前资产信息
        assets = self.account_information["total_assets"][-1]
        cash = self.account_information["cash"][-1]

        # 计算回撤百分比
        tmp_retreat_ptc = assets / self.max_total_assets - 1
        retreat_pct = tmp_retreat_ptc if assets < self.max_total_assets else 0

        # 计算总盈亏百分比
        gl_pct = assets / self.initial_amount - 1

        prev_assets = self.account_information["total_assets"][-2]
        
        # 计算交易成本
        # 如果有交易记录，计算最近一次交易的成本
        if len(self.transaction_memory) > 0:
            recent_transactions = np.abs(self.transaction_memory[-1])
            transaction_volume = np.sum(recent_transactions * self.closings)
            avg_cost_pct = (self.buy_cost_pct + self.sell_cost_pct) / 2
            transaction_cost = transaction_volume * avg_cost_pct 
        else:
            transaction_cost = 0

        # 计算变化率（当前资产相对于上一步的变化）
        if len(self.account_information["total_assets"]) > 1:
            change_pct = np.log(assets / prev_assets)
        else:
            change_pct = 0

        # 创建记录数组
        rec = [
            self.episode,                                                      # 回合数
            self.date_index - self.starting_point,                             # 步数
            reason,                                                            # 原因
            f"{self.currency}{'{:0,.0f}'.format(float(cash))}",                 # 现金
            f"{self.currency}{'{:0,.0f}'.format(float(assets))}",               # 总资产
            f"{terminal_reward*100:0.2f}%",                                    # 奖励
            f"{change_pct*100:0.2f}%",                                         # 变化率
            f"{retreat_pct*100:0.2f}%",                                        # 回撤率
            f"{self.currency}{'{:0,.0f}'.format(float(transaction_cost))}",    # 交易成本
            f"{gl_pct*100:0.2f}%"                                              # 总盈亏百分比
        ]

        # 保存记录并打印
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
            # 修改模板以包含10个元素
            self.template = "{0:4}|{1:4}|{2:12}|{3:12}|{4:12}|{5:8}|{6:8}|{7:8}|{8:8}|{9:8}"
            # 0, 1, 2, ... 是序号
            # 4, 4, 12, ... 是占位格的大小
            print(
                self.template.format(
                    "Ep",       # 回合数
                    "Step",     # 步数
                    "Reason",    # 终止原因
                    "Cash",      # 现金
                    "Total",     # 总资产
                    "Reward",    # 奖励
                    "Change",    # 变化率
                    "Retreat",   # 回撤率
                    "Cost",      # 交易成本
                    "GainLoss"   # 盈亏百分比
                )
            )
            self.printed_header = True

    def get_reward(self) -> float:
        """
        计算当前步骤的奖励值 (Reward Function)

        奖励函数由三部分组成：
        1. 对数收益率 (Log Return)：使用对数收益率代替简单收益率，更符合金融理论
           - 计算公式：log(当前资产 / 上一时刻资产)
           - 对数收益率在连续复利情况下更准确，且可以更好地处理不同时间尺度的收益

        2. 回撤惩罚 (Drawdown Penalty)：惩罚资产价值从历史最高点的下跌
           - 计算公式：(assets / self.max_total_assets) - 1（这是一个负值或0）
           - 这鼓励智能体避免大幅度的资产回撤，追求稳定增长

        3. 交易成本惩罚 (Transaction Cost Penalty)：惩罚过度交易
           - 计算公式：-交易量 * 交易成本系数
           - 这鼓励智能体减少不必要的交易，避免过度交易导致的成本损失

        总奖励 = 对数收益率 + 回撤惩罚权重 * 回撤惩罚 + 交易成本权重 * 交易成本惩罚

        Returns:
            float: 计算得到的奖励值
        """
        if self.current_step == 0:
            return 0
        else:
            # 获取当前总资产价值
            assets = self.account_information["total_assets"][-1]

            # 1. 计算对数收益率 (Log Return)
            # 使用对数收益率代替简单收益率，更符合金融理论
            # 获取上一时刻的资产价值，如果是第一步，则使用初始资金
            if len(self.account_information["total_assets"]) > 1:
                prev_assets = self.account_information["total_assets"][-2]
            else:
                prev_assets = self.initial_amount

            # 计算对数收益率：log(当前资产 / 上一时刻资产)
            log_return = np.log(assets / prev_assets)

            # 2. 计算回撤惩罚 (Drawdown Penalty)
            # 更新历史最大资产价值
            if assets >= self.max_total_assets:
                self.max_total_assets = assets
                drawdown_penalty = 0  # 如果创新高，没有回撤
            else:
                # 使用对数比例计算回撤
                drawdown_penalty = (assets / self.max_total_assets) - 1  # 这是一个负值

            # 3. 计算交易成本惩罚 (Transaction Cost Penalty)
            # 如果是第一步，没有交易成本
            if len(self.transaction_memory) == 0:
                transaction_cost_penalty = 0
            else:
                # 获取最近一次交易的绝对值总和（交易量）
                recent_transactions = np.abs(self.transaction_memory[-1])
                transaction_volume = np.sum(recent_transactions * self.closings)

                # 交易成本惩罚与交易量和成本比例成正比
                avg_cost_pct = (self.buy_cost_pct + self.sell_cost_pct) / 2
                transaction_cost_penalty = -transaction_volume * avg_cost_pct / prev_assets

            # 设置权重系数
            drawdown_weight = 0.1  # 回撤惩罚权重
            transaction_cost_weight = 1.0  # 交易成本惩罚权重

            # 计算总奖励
            reward = log_return + drawdown_weight * drawdown_penalty + transaction_cost_weight * transaction_cost_penalty

            return reward

    def get_transactions(self, actions: np.ndarray) -> np.ndarray:
        """
        根据动作计算实际交易的股数 (Action到Transaction的转换)

        这个函数实现了从智能体的抽象动作到具体交易指令的转换。
        在强化学习中，动作空间通常是标准化的（如[-1,1]范围），
        需要将其转换为实际环境中的操作（如具体买卖多少股）。

        这个转换过程也与奖励函数中的交易成本惩罚相关。在奖励函数中，
        我们使用交易量（股数 * 价格）和交易成本比例来计算交易成本惩罚，
        从而鼓励智能体减少不必要的交易。

        Args:
            actions: 模型输出的动作，范围为[-1, 1]，表示买卖的比例
                    -1表示卖出最大允许数量
                    1表示买入最大允许数量
                    0表示不交易

        Returns:
            np.ndarray: 实际交易的股票数量，正数表示买入，负数表示卖出
        """
        # 记录原始动作
        self.actions_memory.append(actions)

        # 将动作比例转换为交易金额
        # 例如：如果action=0.5且hmax=10，则交易金额为5
        actions = actions * self.hmax

        # 收盘价为0的股票不进行交易（避免除以0的错误）
        actions = np.where(self.closings > 0, actions, 0)

        # 将交易金额转换为股票数量（金额/价格=数量）
        # 同时处理除以0的情况
        out = np.zeros_like(actions)
        zero_or_not = self.closings != 0
        actions = np.divide(actions, self.closings, out=out, where=zero_or_not)

        # 卖出限制：不能卖出超过当前持有的股票数量
        actions = np.maximum(actions, -np.array(self.holdings))

        # 将-0的值全部置为0（避免符号问题）
        actions[actions == -0] = 0

        return actions

    def step(self, actions: np.ndarray) -> Tuple[List, float, bool, bool, dict]:
        """
        执行一步交易 (Environment Step Function)

        这是强化学习环境中的核心函数，实现了环境的状态转移。
        每次调用step函数，环境会根据智能体提供的动作执行一步交易，
        然后返回新的状态、奖励和是否终止的信息。

        在强化学习的MDP（马尔可夫决策过程）框架中，step函数实现了从
        当前状态s和动作a到下一状态s'的转移，并计算即时奖励r。

        这个函数与奖励函数密切相关，它计算交易成本并更新账户状态，
        这些信息随后被奖励函数用于计算对数收益率、回撤惩罚和交易成本惩罚。

        Args:
            actions: 模型输出的动作，范围为[-1, 1]，表示买卖的比例

        Returns:
            Tuple: (新状态, 奖励, 终止标志, 截断标志, 信息字典)
        """
        # 累计交易量统计，用于计算交易成本惩罚
        self.sum_trades += np.sum(np.abs(actions))

        # 打印日志头（如果需要）
        self.log_header()
        if(self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        if self.date_index == len(self.dates) - 1:
            # 直接返回终止状态的结果，包含5个值（状态、奖励、终止标志、截断标志、信息字典）
            return self.return_terminal(reward=self.get_reward())
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
                    # 资金不足时终止环境
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
        def get_self():
            return deepcopy(self)

        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs

    def get_multiproc_env(
        self, n: int = 10
    ) -> Tuple[Any, Any]:
        def get_self():
            return deepcopy(self)

        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        obs = e.reset()
        return e, obs

    def save_asset_memory(self) -> pd.DataFrame:
        if self.current_step == 0:
            return None
        else:
            self.account_information["date"] = self.dates[
                -len(self.account_information["cash"]):
            ]
            return pd.DataFrame(self.account_information)

    def save_action_memory(self) -> pd.DataFrame:
        if self.current_step == 0:
            return None
        else:
            return pd.DataFrame(
                {
                    "date": self.dates[-len(self.account_information["cash"]):],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory
                }
            )
