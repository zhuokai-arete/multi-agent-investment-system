import os
import pandas as pd
import numpy as np

def load_market_data(filename, benchmarkname):
    data_dir = os.path.join(os.getcwd(), "data")
    data_path = os.path.join(data_dir, filename)
    benchmark_path = os.path.join(data_dir, benchmarkname)
    # 读取资产文件
    data = pd.read_excel(data_path, header=1, index_col=0, parse_dates=True)
    asset_list = data.columns.tolist()
    data, asset_types = preprocess_market_data(data)
    data = data.replace(0, np.nan)
    data = data.iloc[:-56,:]
    pd.DataFrame(data).to_csv('new_data.csv', index=False)
    returns = data.pct_change()
    rolling_cov_matrices, available_assets = compute_rolling_cov_matrices(returns, window_size = 90)

    # 计算技术指标
    feature_data = compute_features(data)
    
    # 读取基准文件
    benchmark = pd.read_excel(benchmark_path, header=1, index_col=0, parse_dates=True)
    benchmark = benchmark.iloc[:-1,-1]
    return returns, available_assets, rolling_cov_matrices, benchmark, feature_data, asset_list

def preprocess_market_data(data):
    """
    对市场数据进行预处理：
    - 将美元资产按USDCNY.EX汇率换算为人民币
    - 对美股资产平移日期（shift），对齐中美时差
    - 其他资产根据需要可扩展港币、日元换算
    - 返回处理后的DataFrame
    """
    # 定义美股资产（shift+美元汇率换算）
    usd_assets = ['SPY.P', 'QQQ.O', 'SPX.GI', 'NDX.GI']
    usdcny_col = 'USDCNY.EX'
    
    # 判断数据中是否包含这些列
    existing_usd_assets = [col for col in usd_assets if col in data.columns]
    
    # 先对美股资产平移1天
    for asset in existing_usd_assets:
        data[asset] = data[asset].shift(1)
    
    # 再按USDCNY.EX进行汇率换算
    for asset in existing_usd_assets:
        data[asset] = data[asset] * data[usdcny_col]
    
    # 标记资产类型（人民币、美元、日元、港币）
    asset_types = {}
    for col in data.columns:
        if col in usd_assets:
            asset_types[col] = 'USD'
        elif col in ['N225.GI', 'TOPX.T']:
            asset_types[col] = 'JPY'
        elif col in ['HSI.HI', 'HSTECH.HI']:
            asset_types[col] = 'HKD'
        elif col == usdcny_col:
            asset_types[col] = 'ExchangeRate'
        else:
            asset_types[col] = 'CNY'
    
    # 如果后续需要处理港币、日元汇率，可在这里加处理逻辑（比如按USDJPY、USDHKD换算）
    
    return data, asset_types


def compute_rolling_cov_matrices(returns, window_size):
    rolling_cov_matrices = {}
    available_assets = {}
    dates = returns.index[window_size:]
    for current_date in dates:
        # .iloc[] 左闭右开
        # 假设所选日期索引为7，window_size=3，则要取5,6,7
        # .iloc[]就需要是.iloc[5:8]
        start_idx = returns.index.get_loc(current_date) - window_size + 1
        end_idx = returns.index.get_loc(current_date) + 1
        past_data = returns.iloc[start_idx:end_idx]
        # ~、.any()、.isna()在 pandas 中是如何组合、先后判断的
        # 先后顺序核心原则：1. 操作（点运算）优先级最高，会优先调用对象的方法；2~ 是按位取反操作符，作用在布尔值或布尔数组上；
        # 组合逻辑：先进行方法调用（.isna()、.any()），再进行取反（~）。
        # Step1：past_data.isna()对 DataFrame past_data 每个元素，检查是否为 NaN，返回布尔 DataFrame。
        # Step2：past_data.isna().any() 对布尔 DataFrame 按列（默认 axis=0），判断每列是否有任意 True（即是否有 NaN）。
        # 返回 Series，索引是列名，值是 True/False（True=该列含NaN，False=无NaN）。
        # Step3：~对 Series 进行按位取反（True→False，False→True），即：
        # 含NaN → 不可用 → False
        # 无NaN → 可用 → True
        # Step4：past_data.columns[...] 根据布尔 Series 筛选 past_data.columns，只保留可用资产。
        valid_assets = past_data.columns[~past_data.isna().any()]
        if len(valid_assets) == 0:
            continue
        # rolling_cov_matrices：
        # 字典，key = 日期（current_date），value = 该日期窗口内计算的协方差矩阵（DataFrame，行列=资产，值=协方差）。
        # available_assets：
        # 字典，key = 日期（current_date），value = valid_assets（Index，包含可用资产名称）。
        available_assets[current_date] = valid_assets
        rolling_cov_matrices[current_date] = past_data[valid_assets].cov()
    return rolling_cov_matrices, available_assets

def compute_features(data):
    feature_list = []
    for col in data.columns:
        col_data = data[col]
        tmp = pd.DataFrame(index=data.index)
        # 价格
        tmp[f"{col}_price"] = col_data
        # 收益率
        tmp[f"{col}_return"] = col_data.pct_change()
        # 波动率（过去20日）
        tmp[f"{col}_volatility"] = col_data.pct_change().rolling(20).std()
        # 均线（过去10日）
        tmp[f"{col}_ma10"] = col_data.rolling(10).mean()
        # 动量（过去5日）
        tmp[f"{col}_momentum"] = col_data - col_data.shift(5)
        # RSI（过去14日）
        delta = col_data.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        tmp[f"{col}_rsi"] = 100 - 100 / (1 + rs)
        feature_list.append(tmp)
    features = pd.concat(feature_list, axis=1)
    features = features.fillna(0)
    return features