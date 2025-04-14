import pandas as pd
import os
import codecs
from stable_baselines3.common.vec_env import DummyVecEnv
from data import Data
import sys
from argparse import ArgumentParser
from typing import Optional 
sys.path.append("..")
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from utils import config
from utils.env import StockLearningEnv
from utils.models import DRL_Agent


class Trader(object):
    """用来测试训练结果的类

    该类负责加载训练好的强化学习模型，使用交易数据集进行回测，执行交易策略，
    并保存和展示交易结果。

    Attributes:
        model_name (str): 强化学习的算法名称（如 'a2c', 'ppo' 等），用于调用指定算法
        train_dir (str): 存储训练模型的目录，默认为 "train_file"
        data_dir (str): 存储数据的目录，默认为 "data_file"
        trade_dir (str): 存储交易结果的目录，默认为 "trade_file"
    """

    def __init__(self, model_name: str = 'a2c') -> None:
        """初始化 Trader 类

        Args:
            model_name (str, optional): 强化学习算法名称，默认为 'a2c'
        """
        self.model_name = model_name  # 设置强化学习算法名称
        self.train_dir = "train_file"  # 设置训练模型存储目录
        self.data_dir = "data_file"  # 设置数据存储目录
        self.trade_dir = "trade_file"  # 设置交易结果存储目录
        self.create_trade_dir()  # 调用方法创建交易结果存储目录

    def create_trade_dir(self) -> None:
        """创建存储交易结果的文件夹

        检查是否存在交易结果存储目录（self.trade_dir），如果不存在则创建，
        并打印相应的提示信息。

        Returns:
            None
        """
        if not os.path.exists(self.trade_dir):
            os.makedirs(self.trade_dir)
            print(f"{self.trade_dir} 文件夹创建成功!")
        else:
            print(f"{self.trade_dir} 文件夹已存在!")

    def trade(self) -> None:
        """使用训练好的模型进行交易

        加载交易数据集，创建交易环境，加载指定的强化学习模型，
        执行交易预测，保存交易结果，并打印回测信息。如果模型不存在，
        则提示用户先运行训练脚本。

        Returns:
            None
        """
        # 获取交易数据集
        trade_data = self.get_trade_data()
        # 创建交易环境
        e_trade_gym = self.get_env(trade_data)
        # 初始化深度强化学习代理
        agent = DRL_Agent(env=e_trade_gym)

        # 获取训练好的模型
        model = self.get_model(agent)

        if model is not None:
            # 使用模型进行交易预测，获取账户净值和交易行为
            df_account_value, df_actions = DRL_Agent.DRL_prediction(model=model, 
                                                                   environment=e_trade_gym)
            # 保存交易结果
            self.save_trade_result(df_account_value, df_actions)
            # 打印交易结果
            self.print_trade_result(df_account_value, df_actions)
        else:
            # 如果模型不存在，提示用户
            print(f"{self.train_dir} 文件夹中未找到 {self.model_name} model，"
                  f"请先运行 trainer.py 或者将训练好的 {self.model_name} model 放入 {self.train_dir} 中")

    def get_trade_data(self) -> pd.DataFrame:
        """获取交易数据集

        从数据目录中读取交易数据集（trade.csv），如果数据集不存在，
        则调用 Data 类的 pull_data 方法下载数据。

        Returns:
            pd.DataFrame: 交易数据集
        """
        trade_data_path = os.path.join(self.data_dir, "trade.csv")
        if not os.path.exists(trade_data_path):
            print("数据不存在，开始下载")
            Data().pull_data()  # 调用 Data 类下载和处理数据

        # 读取交易数据集
        trade_data = pd.read_csv(trade_data_path)
        print("数据读取成功!")
        
        return trade_data

    def get_env(self, trade_data: pd.DataFrame) -> DummyVecEnv:
        """获取交易环境

        根据交易数据集创建强化学习交易环境，使用 StockLearningEnv 类，
        并应用配置文件中的环境参数。

        Args:
            trade_data (pd.DataFrame): 交易数据集

        Returns:
            DummyVecEnv: 强化学习交易环境
        """
        e_trade_gym = StockLearningEnv(df=trade_data,
                                      random_start=False,  # 固定起点，不随机
                                      **config.ENV_PARAMS)  # 传入环境参数
        return e_trade_gym

    def get_model(self, agent) -> Optional[object]:
        """获取训练好的模型

        根据指定的模型名称和参数，从 DRL_Agent 获取模型，并尝试加载保存的模型文件。
        如果模型文件不存在，则返回 None。

        Args:
            agent: 深度强化学习代理对象，包含获取模型的方法

        Returns:
            Optional[object]: 加载的模型对象，如果模型不存在则返回 None
        """
        # 获取模型，传入模型名称和参数
        model = agent.get_model(self.model_name,  
                                model_kwargs=config.__dict__[f"{self.model_name.upper()}_PARAMS"], 
                                verbose=0)  # 设置 verbose=0 减少日志输出
        # 构造模型文件路径
        model_dir = os.path.join(self.train_dir, f"{self.model_name}.model")
        
        if os.path.exists(model_dir):
            # 加载保存的模型
            model.load(model_dir)
            return model
        else:
            return None

    def save_trade_result(self, 
                          df_account_value: pd.DataFrame, 
                          df_actions: pd.DataFrame) -> None:
        """保存交易后的数据

        将交易回测的账户净值和交易行为分别保存到 CSV 文件中，
        文件名包含模型名称以区分不同模型的结果。

        Args:
            df_account_value (pd.DataFrame): 账户净值数据
            df_actions (pd.DataFrame): 交易行为数据

        Returns:
            None
        """
        # 保存账户净值数据
        account_value_path = os.path.join(self.trade_dir, f"account_value_{self.model_name}.csv")
        df_account_value.to_csv(account_value_path, index=False)

        # 保存交易行为数据
        actions_path = os.path.join(self.trade_dir, f"actions_{self.model_name}.csv")
        df_actions.to_csv(actions_path, index=False)

    def print_trade_result(self, 
                           df_account_value: pd.DataFrame, 
                           df_actions: pd.DataFrame) -> None:
        """打印交易结果

        输出回测的时间窗口、账户净值的前后几行数据，以及最近的交易行为，
        以便用户检查交易结果。

        Args:
            df_account_value (pd.DataFrame): 账户净值数据
            df_actions (pd.DataFrame): 交易行为数据

        Returns:
            None
        """
        # 打印回测时间窗口
        print(f"回测的时间窗口：{config.End_Trade_Date} 至 {config.End_Test_Date}")

        # 打印账户净值数据
        print("查看日账户净值")
        print("开始: ")
        print(df_account_value.head())
        print("")
        print("结束: ")
        print(df_account_value.tail())

        # 打印交易行为数据
        print("查看每日所作的交易")
        print(df_actions.tail())


def start_trade():
    """启动交易程序的主函数

    通过命令行参数解析模型名称，创建 Trader 实例并调用 trade 方法开始交易。
    """
    # 创建命令行参数解析器
    parser = ArgumentParser(description="set parameters for train mode")
    parser.add_argument(
        '--model', '-m',
        dest='model',
        default='a2c',  # 默认模型为 a2c
        help='choose the model you want to train',
        metavar="MODEL",
        type=str
    )

    # 解析命令行参数
    options = parser.parse_args()
    # 创建 Trader 实例并启动交易
    Trader(model_name=options.model).trade()


if __name__ == "__main__":
    """主程序入口

    调用 start_trade 函数，解析命令行参数并执行交易流程。
    """
    start_trade()