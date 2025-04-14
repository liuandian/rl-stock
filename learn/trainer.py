import codecs
import os
import sys
import pandas as pd
from argparse import ArgumentParser
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append("..")
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from utils import config
from utils.env import StockLearningEnv
from utils.models import DRL_Agent
from data import Data


class Trainer(object):
    """用来训练强化学习模型的类

    该类负责加载训练和交易数据集，创建强化学习环境，训练指定的强化学习模型，
    并保存训练好的模型。

    Attributes:
        model_name (str): 强化学习的算法名称（如 'a2c', 'ppo' 等），用于调用指定算法
        total_timesteps (int): 总的训练步数，决定模型训练的迭代次数
        train_dir (str): 存储训练模型的目录，默认为 "train_file"
        data_dir (str): 存储数据的目录，默认为 "data_file"
    """

    def __init__(self, model_name: str = 'a2c', total_timesteps: int = 200000) -> None:
        """初始化 Trainer 类

        Args:
            model_name (str, optional): 强化学习算法名称，默认为 'a2c'
            total_timesteps (int, optional): 总训练步数，默认为 200000
        """
        self.model_name = model_name  # 设置强化学习算法名称
        self.total_timesteps = total_timesteps  # 设置总训练步数
        self.train_dir = "train_file"  # 设置训练模型存储目录
        self.data_dir = "data_file"  # 设置数据存储目录
        self.create_train_dir()  # 调用方法创建训练模型存储目录

    def create_train_dir(self) -> None:
        """创建存储训练结果的文件夹

        检查是否存在训练模型存储目录（self.train_dir），如果不存在则创建，
        并打印相应的提示信息。

        Returns:
            None
        """
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
            print(f"{self.train_dir} 文件夹创建成功!")
        else:
            print(f"{self.train_dir} 文件夹已存在!")

    def train(self) -> None:
        """开始训练强化学习模型

        加载训练和交易数据集，创建训练和交易环境，初始化强化学习代理，
        使用指定的算法和参数训练模型，并保存训练好的模型。

        Returns:
            None
        """
        # 获取训练和交易数据集
        train_data, trade_data = self.get_data()

        # 获取训练和交易环境
        # 注意：新版本的 stable_baselines3 中，learn() 方法不再接受 eval_env 参数，
        # 但保留交易环境以保持代码结构一致性
        env_train, env_trade = self.get_env(train_data, trade_data)

        # 避免 IDE 警告：env_trade 未使用，但保留以保持代码结构
        _ = env_trade  # 这行仅用于消除未使用变量警告

        # 初始化深度强化学习代理
        agent = DRL_Agent(env=env_train)

        # 获取指定算法的模型，传入模型参数
        model = agent.get_model(self.model_name,
                                model_kwargs=config.__dict__[f"{self.model_name.upper()}_PARAMS"],
                                verbose=0)  # 设置 verbose=0 减少日志输出

        # 开始训练模型
        model.learn(total_timesteps=self.total_timesteps,
                    log_interval=1,  # 每 1 次迭代记录日志
                    tb_log_name='env_cashpenalty_highlr',  # TensorBoard 日志名称
                    progress_bar=True)  # 显示训练进度条

        # 保存训练好的模型
        self.save_model(model)

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """获取训练数据集和交易数据集

        从数据目录中读取训练数据集（train.csv）和交易数据集（trade.csv），
        如果数据集不存在，则调用 Data 类的 pull_data 方法下载数据。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 训练数据集和交易数据集
        """
        train_data_path = os.path.join(self.data_dir, "train.csv")
        trade_data_path = os.path.join(self.data_dir, "trade.csv")
        # 检查训练和交易数据集是否存在
        if not (os.path.exists(train_data_path) and os.path.exists(trade_data_path)):
            print("数据不存在，开始下载")
            Data().pull_data()  # 调用 Data 类下载和处理数据

        # 读取训练和交易数据集
        train_data = pd.read_csv(train_data_path)
        trade_data = pd.read_csv(trade_data_path)
        print("数据读取成功!")

        return train_data, trade_data

    def get_env(self,
                train_data: pd.DataFrame,
                trade_data: pd.DataFrame) -> tuple[DummyVecEnv, DummyVecEnv]:
        """创建训练环境和交易环境

        根据训练和交易数据集，分别创建强化学习训练环境和交易环境，
        使用 StockLearningEnv 类，并应用配置文件中的环境参数。

        Args:
            train_data (pd.DataFrame): 训练数据集
            trade_data (pd.DataFrame): 交易数据集

        Returns:
            tuple[DummyVecEnv, DummyVecEnv]: 训练环境和交易环境
        """
        # 创建训练环境，启用随机起点
        e_train_gym = StockLearningEnv(df=train_data,
                                      random_start=True,  # 训练时随机选择起点
                                      **config.ENV_PARAMS)  # 传入环境参数
        env_train, _ = e_train_gym.get_sb_env()  # 获取 stable_baselines3 兼容的环境

        # 创建交易环境，固定起点
        e_trade_gym = StockLearningEnv(df=trade_data,
                                      random_start=False,  # 交易时不随机选择起点
                                      **config.ENV_PARAMS)  # 传入环境参数
        env_trade, _ = e_trade_gym.get_sb_env()  # 获取 stable_baselines3 兼容的环境

        return env_train, env_trade

    def save_model(self, model) -> None:
        """保存训练好的模型

        将训练好的模型保存到训练目录中，文件名包含模型名称。

        Args:
            model: 训练好的强化学习模型对象

        Returns:
            None
        """
        model_path = os.path.join(self.train_dir, f"{self.model_name}.model")
        model.save(model_path)  # 保存模型到指定路径


def start_train():
    """启动训练程序的主函数

    通过命令行参数解析模型名称和总训练步数，创建 Trainer 实例并调用 train 方法开始训练。
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

    parser.add_argument(
        '--total_timesteps', '-tts',
        dest='total_timesteps',
        default=200000,  # 默认总训练步数为 200000
        help='set the total_timesteps when you train the model',
        metavar="TOTAL_TIMESTEPS",
        type=int
    )

    # 解析命令行参数
    options = parser.parse_args()
    # 创建 Trainer 实例并启动训练
    Trainer(model_name=options.model,
            total_timesteps=options.total_timesteps).train()


if __name__ == "__main__":
    """主程序入口

    调用 start_train 函数，解析命令行参数并执行训练流程。
    """
    start_train()