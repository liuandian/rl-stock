"""
简化版回测分析函数，不依赖pyfolio库
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.pull_data import Pull_data
from utils import config

def get_daily_return(df, value_col_name="account_value"):
    """获取每日收益率"""
    df = df.copy()
    df["daily_return"] = df[value_col_name].pct_change(1)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True, drop=True)
        # 尝试添加UTC时区信息，如果已经有时区信息则忽略
        try:
            df.index = df.index.tz_localize("UTC")
        except TypeError:
            # 如果已经有时区信息，则会抛出异常，我们可以忽略
            pass
    return pd.Series(df["daily_return"], index=df.index)

def get_baseline(ticker, start, end):
    """获取指数的行情数据"""
    baselines = Pull_data(
        ticker_list=ticker,
        start_date=start,
        end_date=end,
        pull_index=True
    ).pull_data()
    return baselines

def simple_backtest_plot(account_value, baseline_start, baseline_end, baseline_ticker, value_col_name="account_value"):
    """简化版回测分析函数，不依赖pyfolio库"""
    # 准备账户价值数据
    df = account_value.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 获取基准指数数据
    baseline_df = get_baseline(
        ticker=baseline_ticker,
        start=baseline_start,
        end=baseline_end
    )
    baseline_df['date'] = pd.to_datetime(baseline_df['date'])
    baseline_df.set_index('date', inplace=True)
    
    # 确保日期范围匹配
    print("账户价值数据日期范围:", df.index.min(), "to", df.index.max())
    print("基准指数数据日期范围:", baseline_df.index.min(), "to", baseline_df.index.max())
    
    # 计算累计收益率
    start_value = df[value_col_name].iloc[0]
    df['cum_return'] = df[value_col_name] / start_value
    
    start_close = baseline_df['close'].iloc[0]
    baseline_df['cum_return'] = baseline_df['close'] / start_close
    
    # 找出共同的日期
    common_dates = df.index.intersection(baseline_df.index)
    if len(common_dates) == 0:
        print("警告: 测试数据和基准指数数据没有共同的日期")
        # 尝试将时区信息移除
        df.index = df.index.tz_localize(None)
        baseline_df.index = baseline_df.index.tz_localize(None)
        common_dates = df.index.intersection(baseline_df.index)
        if len(common_dates) == 0:
            print("错误: 即使移除时区信息后仍然没有共同日期")
            return
    
    print("共同日期数量:", len(common_dates))
    
    # 仅使用共同的日期
    df = df.loc[common_dates]
    baseline_df = baseline_df.loc[common_dates]
    
    # 绘制图表
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cum_return'], label='strategy',color='red')
    plt.plot(baseline_df.index, baseline_df['cum_return'], label='benchmark',color='gray')
    plt.title('comapre')
    plt.xlabel('date')
    plt.ylabel('cumulative return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 计算一些基本指标
    strategy_return = df[value_col_name].iloc[-1] / df[value_col_name].iloc[0] - 1
    benchmark_return = baseline_df['close'].iloc[-1] / baseline_df['close'].iloc[0] - 1
    
    print(f"策略总收益率: {strategy_return:.2%}")
    print(f"基准总收益率: {benchmark_return:.2%}")
    print(f"超额收益率: {strategy_return - benchmark_return:.2%}")
    
    # 计算年化收益率
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.0
    strategy_annual_return = (1 + strategy_return) ** (1 / years) - 1
    benchmark_annual_return = (1 + benchmark_return) ** (1 / years) - 1
    
    print(f"策略年化收益率: {strategy_annual_return:.2%}")
    print(f"基准年化收益率: {benchmark_annual_return:.2%}")
    print(f"年化超额收益率: {strategy_annual_return - benchmark_annual_return:.2%}")
    
    # 计算最大回撤
    df['cum_max'] = df['cum_return'].cummax()
    df['drawdown'] = (df['cum_max'] - df['cum_return']) / df['cum_max']
    max_drawdown = df['drawdown'].max()
    
    baseline_df['cum_max'] = baseline_df['cum_return'].cummax()
    baseline_df['drawdown'] = (baseline_df['cum_max'] - baseline_df['cum_return']) / baseline_df['cum_max']
    benchmark_max_drawdown = baseline_df['drawdown'].max()
    
    print(f"策略最大回撤: {max_drawdown:.2%}")
    print(f"基准最大回撤: {benchmark_max_drawdown:.2%}")
    
    # 绘制回撤图
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['drawdown'], label='strategy drawdown',color='red')
    plt.plot(baseline_df.index, baseline_df['drawdown'], label='benchmark drawdown',color='gray')
    plt.title('drawdown compare')
    plt.xlabel('date')
    plt.ylabel('drawdown')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return df, baseline_df