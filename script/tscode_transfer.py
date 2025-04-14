import re
import os

import os
import re
from typing import List

def create_dir(txt_transfer_dir: str) -> None:
    """创建用于存储转换后文件的目录

    检查指定的目录是否存在，如果不存在则创建，并打印相应的提示信息。

    Args:
        txt_transfer_dir (str): 转换后文件的目标存储目录路径

    Returns:
        None
    """
    if not os.path.exists(txt_transfer_dir):
        os.makedirs(txt_transfer_dir)  # 创建目录
        print(f"创建 {txt_transfer_dir} 目录成功!")
    else:
        print(f"{txt_transfer_dir} 目录已存在!")

def transfer(txt_dir: str, transfer_dir: str, file_name: str) -> None:
    """将原始股票代码文件批量转换为 Tushare API 可识别的股票代码格式

    该函数读取原始股票代码文件（.txt），根据股票代码的开头数字（6 表示上海证券交易所，其他表示深圳证券交易所），
    添加相应的后缀（.SH 或 .SZ），并将转换后的结果保存为新的文件，格式为 Python 列表形式。

    Args:
        txt_dir (str): 原始股票代码文件存放的目录
        transfer_dir (str): 转换后文件存放的目录
        file_name (str): 原始文件的文件名（包含 .txt 扩展名）

    Returns:
        None
    """
    # 初始化存储转换后内容的列表
    transferd: List[str] = []
    
    # 使用正则表达式提取文件名（去掉 .txt 扩展名），作为指数标识
    file_name_sub = re.findall(r"(.+?).txt", file_name)[0]
    # 在转换后的文件首行添加指数标识，例如 "SSE_50 = ["
    transferd.append(f"{file_name_sub} = [")

    # 构造输入和输出文件的完整路径
    read_dir = os.path.join(txt_dir, file_name)
    write_dir = os.path.join(transfer_dir, f"{file_name_sub}_transferred.txt")
    
    # 读取原始文件内容
    with open(read_dir, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去除每行的换行符
            # 根据股票代码首字符判断交易所并添加后缀
            if line[0] == '6':
                # 上海证券交易所，添加 .SH 后缀
                transferd.append(f"    \"{line}.SH\"")
            else:
                # 深圳证券交易所，添加 .SZ 后缀
                transferd.append(f"    \"{line}.SZ\"")

    # 将转换后的内容写入新文件
    with open(write_dir, "w") as f:
        # 写入首行（指数标识）
        f.write(transferd[0])
        f.write('\n')

        # 写入中间行，每行末尾添加逗号
        for i in range(1, len(transferd) - 1):
            f.write(transferd[i] + ',')
            f.write('\n')
        
        # 写入最后一行，末尾添加闭合括号 ']'
        f.write(transferd[-1] + "]")

if __name__ == "__main__":
    """主程序入口

    自动处理指定目录中的所有股票代码文件，将其转换为 Tushare API 可识别的格式，
    并将结果保存到新的目录中。
    """
    # 定义原始文件目录和转换后文件目录
    txt_dir = "index_txt"
    txt_transfer_dir = txt_dir + "_transfer"

    # 创建转换后文件的存储目录
    create_dir(txt_transfer_dir)

    # 获取原始文件目录中的所有文件名
    txt_list = os.listdir(txt_dir)
    
    # 遍历每个文件并进行转换
    for txt in txt_list:
        transfer(txt_dir=txt_dir, transfer_dir=txt_transfer_dir, file_name=txt)