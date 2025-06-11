import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
from agents.calculate_portfolio_index_with_units_adjusted import calculate_portfolio_index_with_units
from agents.base_agent import BaseAgent



class RiskParityAgent(BaseAgent):
    def __init__(self, returns, available_assets, rolling_cov_matrices):
        self.returns = returns
        self.available_assets = available_assets
        self.rolling_cov_matrices = rolling_cov_matrices
        self.weights = self.optimize()

    def risk_parity_objective(self, weights, cov_matrix):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        risk_contributions = weights * np.dot(cov_matrix, weights) / portfolio_vol
        avg_risk = np.mean(risk_contributions)
        return np.sum((risk_contributions - avg_risk) ** 2)

    def optimize(self):
        optimized_weights = {}
        # optimize 方法只遍历 rolling_cov_matrices.keys()（每个日期）。
        # 每个日期只优化 rolling_cov_matrices[date]，并用 available_assets[date] 作为权重索引。
        for date in self.rolling_cov_matrices:
            cov = self.rolling_cov_matrices[date]
            n = len(cov)
            # 得到初始权重
            w0 = np.ones(n) / n
            bounds = [(1e-10, 1 - 1e-10)] * n
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            '''
            minimize(func, x0, args=(), ...)
            func 是要最小化的目标函数。
            x0 是优化的变量初始值（比如权重 w0）。
            args 是一个元组（tuple），表示传给 func 除 x0 外的额外参数。
            '''
            res = minimize(self.risk_parity_objective, w0, args=(cov,), method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 5000, 'disp': True, 'tol': 1e-10,
                                                             'ftol': 1e-10, 'gtol': 1e-10})
            if res.success:
                weights = pd.Series(res.x, index=self.available_assets[date])
                # 字典 optimized_weights，其中 key 为日期，value 为 Series（索引为资产，值为权重）。
                optimized_weights[date] = weights
        return optimized_weights

    # 对接CVaRDownsideVolAgent.py部分
    def get_full_weights(self):
        """
        生成完整权重 DataFrame，包含所有日期和资产
        """
        # .from_dict 用于从字典创造DataFrame
        # self.weights 是字典：{日期: pd.Series（资产权重）}。
        # orient='index' 表示字典的 key（日期）作为 DataFrame 的索引，value（Series）作为 DataFrame 的行。
        full_weights = pd.DataFrame.from_dict(self.weights, orient='index')
        # self.returns.columns：包含所有资产名（原始数据的列名）。
        # reindex(columns=...)：补全缺失资产列（某些日期某些资产可能无权重），补齐后为全资产列表。
        # .fillna(0)：对缺失资产权重填充0，表示“该资产该日未持仓”。
        full_weights = full_weights.reindex(columns=self.returns.columns).fillna(0)
        # self.returns.index：包含所有日期（原始数据的日期索引）。
        # reindex(self.data.index)：补全缺失日期（某些日期可能未生成权重），补齐后为全日期列表。
        # .fillna(method='ffill')：对空值向前填充（使用前一日权重），保证连续性。
        # .fillna(0)：对最开始没有数据的日期填充0（初始无持仓）。
        full_weights = full_weights.reindex(self.returns.index).fillna(method='ffill').fillna(0)
        return full_weights


    def get_decision(self, market_state):
        date = market_state['date']
        return self.weights.get(date, pd.Series(0, index=self.returns.columns))

    # 计算协方差矩阵可靠度
    def get_confidence(self, market_state):
        date = market_state['date']
        cov = self.rolling_cov_matrices.get(date)
        if cov is not None:
            # 计算条件数：最大特征值 / 最小特征值
            eigvals = np.linalg.eigvalsh(cov)
            max_eig = np.max(eigvals)
            min_eig = np.min(eigvals)
            if min_eig <= 1e-10:  # 防止除0或奇异矩阵
                condition_number = 1e10
            else:
                condition_number = max_eig / min_eig
            # 置信度计算：条件数越大置信度越低，条件数越小置信度越高
            confidence = 1.0 / (1.0 + np.log1p(condition_number))  # log1p(x) = log(1+x)，防止过大溢出
            return confidence
        return 0.0  # 如果当日没有协方差矩阵，置信度为0


    def get_embedding(self, market_state):
        date = market_state['date']
        cov = self.rolling_cov_matrices.get(date)
        if cov is not None:
            return cov.values.flatten()
        else:
            return np.zeros(len(self.data.columns)**2)

