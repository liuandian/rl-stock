import sys
import codecs
import os
from typing import List
import pandas as pd

sys.path.append("..")
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from utils.pull_data import Pull_data
from utils.preprocessors import FeatureEngineer, split_data
from utils import config

class Data(object):
    """用来获取和处理股票数据的类

    该类负责初始化股票代码列表、创建数据存储目录、下载股票数据、进行数据预处理、
    分割训练和交易数据集，并保存处理后的数据。

    Attributes:
        stock_list (List): 股票代码列表，默认为 config.SSE_50 中的股票代码
        data_dir (str): 数据存储目录的路径，默认为 "data_file"
    """

    def __init__(self, 
                 stock_list: List = config.SSE_50) -> None:
        """初始化 Data 类

        Args:
            stock_list (List, optional): 股票代码列表，默认为 config.SSE_50 中定义的股票代码
        """
        self.stock_list = stock_list  # 设置股票代码列表
        self.data_dir = "data_file"  # 设置数据存储目录
        self.create_data_dir()  # 调用方法创建数据存储目录

    def create_data_dir(self) -> None:
        """创建存储数据的文件夹

        检查是否存在指定的数据存储目录（self.data_dir），如果不存在则创建，
        并打印相应的提示信息。

        Returns:
            None
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"{self.data_dir} 文件夹创建成功!")
        else:
            print(f"{self.data_dir} 文件夹已存在!")

    def pull_data(self) -> pd.DataFrame:
        """使用 Tushare API 下载股票数据并进行预处理

        该方法通过 Tushare API 下载指定股票列表的数据，调用特征工程类进行预处理，
        计算额外的金融指标（如交易额、价格变动百分比、每日价格波动），
        并将处理后的数据保存到 CSV 文件中。

        Returns:
            None (数据直接保存到文件，方法本身不返回 DataFrame)
        """
        # 使用 Pull_data 类拉取股票数据
        data = Pull_data(self.stock_list).pull_data()

        # 对数据按日期和股票代码排序，并显示前几行（仅用于调试）
        data.sort_values(['date', 'tic'], ignore_index=True).head()
        print(f"数据下载的时间区间为：{config.Start_Date} 至 {config.End_Date}")
        print("下载的股票列表为: ")
        print(self.stock_list)

        # 使用 FeatureEngineer 类进行数据预处理，添加技术指标
        processed_df = FeatureEngineer(use_technical_indicator=True).preprocess_data(data)

        # 计算额外的金融指标
        processed_df['amount'] = processed_df.volume * processed_df.close  # 计算交易额（成交量 * 收盘价）
        processed_df['change'] = (processed_df.close - processed_df.open) / processed_df.close  # 计算价格变动百分比
        processed_df['daily_variance'] = (processed_df.high - processed_df.low) / processed_df.close  # 计算每日价格波动
        processed_df = processed_df.fillna(0)  # 填充缺失值为 0

        # 打印技术指标相关信息
        print("技术指标列表: ")
        print(config.TECHNICAL_INDICATORS_LIST)
        print(f"技术指标数: {len(config.TECHNICAL_INDICATORS_LIST)}个")
        print("预处理后的数据（前几行）:")
        print(processed_df.head())

        # 将预处理后的数据保存到 CSV 文件
        processed_df.to_csv(os.path.join(self.data_dir, "data.csv"), index=False)

        # 调用数据分割方法，将数据分为训练和交易数据集
        self.data_split(processed_df)

    def data_split(self, data: pd.DataFrame) -> pd.DataFrame:
        """将数据分为训练数据集和交易数据集

        根据配置的时间范围，将输入的数据分割为训练数据集和交易数据集，
        并分别传递给后续方法进行打印和保存。

        Args:
            data (pd.DataFrame): 预处理后的完整数据集，包含股票数据和技术指标

        Returns:
            None (数据通过 save_data 方法保存，方法本身不返回 DataFrame)
        """
        # 分割训练数据集（从 Start_Trade_Date 到 End_Trade_Date）
        train_data = split_data(data, config.Start_Trade_Date, config.End_Trade_Date)
        # 分割交易数据集（从 End_Trade_Date 到 End_Test_Date）
        trade_data = split_data(data, config.End_Trade_Date, config.End_Test_Date)

        # 打印训练和交易数据集的信息
        self.print_data_information(train_data, trade_data)
        # 保存训练和交易数据集
        self.save_data(train_data, trade_data)

    def print_data_information(self,
                               train_data: pd.DataFrame,
                               trade_data: pd.DataFrame) -> None:
        """打印训练和交易数据集的信息

        输出训练和交易数据集的时间范围、数据长度、比例，以及数据集的前几行数据，
        以便用户检查数据分割是否正确。

        Args:
            train_data (pd.DataFrame): 训练数据集
            trade_data (pd.DataFrame): 交易数据集

        Returns:
            None
        """
        print(f"训练数据的范围：{config.Start_Trade_Date} 至 {config.End_Trade_Date}")
        print(f"测试数据的范围：{config.End_Trade_Date} 至 {config.End_Test_Date}")
        print(f"训练数据的长度: {len(train_data)}, 测试数据的长度: {len(trade_data)}")
        print(f"训练集数据 : 测试集数据: {round(len(train_data)/len(trade_data), 1)} : 1")
        print("train_data.head():")
        print(train_data.head())
        print("trade_data.head():")
        print(trade_data.head())

    def save_data(self, 
                  train_data: pd.DataFrame,
                  trade_data: pd.DataFrame) -> None:
        """保存训练和交易数据集到 CSV 文件

        将训练数据集和交易数据集分别保存到 data_dir 目录下的 train.csv 和 trade.csv 文件中。

        Args:
            train_data (pd.DataFrame): 训练数据集
            trade_data (pd.DataFrame): 交易数据集

        Returns:
            None
        """
        train_data.to_csv(os.path.join(self.data_dir, "train.csv"), index=False)
        trade_data.to_csv(os.path.join(self.data_dir, "trade.csv"), index=False)

if __name__ == "__main__":
    """主程序入口

    创建 Data 类的实例，并调用 pull_data 方法开始数据下载、处理和保存流程。
    """
    Data().pull_data()