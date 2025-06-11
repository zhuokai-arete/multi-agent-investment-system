import numpy as np
import pandas as pd
from agents.calculate_portfolio_index_with_units_adjusted import calculate_portfolio_index_with_units
from agents.base_agent import BaseAgent

class CVaRDownsideVolAgent(BaseAgent):
    '''
    每日调仓，但是一旦达到任意指标阈值只清仓归到现金但是不调仓
    full_weights：原始风险平价策略下的结果
    asset_prices：资产价格
    available_assets：可用资产
    cvar_alpha ：
    cvar_window：
    cvar_thresh_pct：
    down_window：
    down_thresh_pct：
    cash_holding_days：持有现金天数
    '''
    def __init__(self, returns, available_assets, full_weights, rolling_cov_matrices,
                 cvar_alpha=0.05, cvar_window=180, cvar_window_judge=90, cvar_thresh_pct=95, 
                 down_window=90, down_window_judge=90, down_thresh_pct=90, cash_holding_days=10):
        print("✅ CVaRDownsideVolAgent: 初始化开始")
        self.returns = returns
        self.available_assets = available_assets
        self.full_weights = full_weights
        self.rolling_cov_matrices = rolling_cov_matrices
        self.cvar_alpha = cvar_alpha
        self.cvar_window = cvar_window
        self.cvar_window_judge = cvar_window_judge
        self.cvar_thresh_pct = cvar_thresh_pct
        self.down_window = down_window
        self.down_window_judge = down_window_judge
        self.down_thresh_pct = down_thresh_pct
        self.cash_holding_days = cash_holding_days
        self.adjusted_weights, self.rolling_cvar, self.rolling_downvol = self.calculate_adjusted_weights()
        print("✅ CVaRDownsideVolAgent: 初始化结束")


    def calculate_adjusted_weights(self):
            print("✅ CVaRDownsideVolAgent: 准备开始 adjusted weights")
            adjusted_weights = self.full_weights.copy()

            # CVaR计算优化：提前计算rolling quantile + mask筛选
            print("🔸 计算 rolling CVaR 和阈值（批量）")
            q_cvar = self.returns.rolling(self.cvar_window).quantile(self.cvar_alpha)
            mask_cvar = self.returns <= q_cvar
            rolling_cvar = (self.returns.where(mask_cvar)).rolling(self.cvar_window).mean() * -1

            cvar_thresh = rolling_cvar.rolling(self.cvar_window_judge).quantile(self.cvar_thresh_pct / 100)

            # DownVol计算优化：提前标记负值
            print("🔸 计算 rolling DownVol 和阈值（批量）")
            neg_returns = self.returns.mask(self.returns >= 0)
            rolling_downvol = neg_returns.rolling(self.down_window).std() * np.sqrt(252)
            downvol_thresh = rolling_downvol.rolling(self.down_window_judge).quantile(self.down_thresh_pct / 100)


            # 提前stack成Series，避免重复查索引
            print("✅ 数据扁平化")
            rolling_cvar_s = rolling_cvar.stack()
            cvar_thresh_s = cvar_thresh.stack()
            rolling_downvol_s = rolling_downvol.stack()
            downvol_thresh_s = downvol_thresh.stack()

            # 初始化标志矩阵
            in_cash_flags = pd.DataFrame(False, index=adjusted_weights.index, columns=adjusted_weights.columns)
            cash_days_counters = pd.DataFrame(0, index=adjusted_weights.index, columns=adjusted_weights.columns)

            print("✅ 开始循环调整权重")
            total_dates = len(adjusted_weights.index)
            for date_idx, date in enumerate(adjusted_weights.index):
                if date_idx % 100 == 0:
                    print(f"🔸进度: {date_idx}/{total_dates} 日期: {date}")

                for asset in adjusted_weights.columns:
                    key = (date, asset)
                    cvar = rolling_cvar_s.get(key, np.nan)
                    cvar_th = cvar_thresh_s.get(key, np.inf)
                    downvol = rolling_downvol_s.get(key, np.nan)
                    downvol_th = downvol_thresh_s.get(key, np.inf)

                    if in_cash_flags.loc[date, asset]:
                        adjusted_weights.loc[date, asset] = 0
                        cash_days_counters.loc[date, asset] -= 1
                        if cash_days_counters.loc[date, asset] <= 0 and not pd.isna(downvol) and downvol <= downvol_th:
                            in_cash_flags.loc[date, asset] = False
                    else:
                        if not pd.isna(cvar) and cvar > cvar_th:
                            adjusted_weights.loc[date, asset] = 0
                            in_cash_flags.loc[date, asset] = True
                            cash_days_counters.loc[date, asset] = self.cash_holding_days

            print("✅ CVaRDownsideVolAgent: adjusted weights 计算结束")
            return adjusted_weights, rolling_cvar, rolling_downvol

    def get_decision(self, market_state):
        date = market_state['date']
        return self.adjusted_weights.loc[date] if date in self.adjusted_weights.index else pd.Series(0, index=self.full_weights.columns)

    # 计算协方差矩阵可靠度
    def get_confidence(self, market_state):
        date = market_state['date']
    
        # 基础：协方差矩阵条件数（复用RiskParity）
        cov = self.rolling_cov_matrices.get(date)
        if cov is not None:
            eigvals = np.linalg.eigvalsh(cov)
            max_eig = np.max(eigvals)
            min_eig = np.min(eigvals)
            condition_number = max_eig / min_eig if min_eig > 1e-10 else 1e10
            conf_cov = 1.0 / (1.0 + np.log1p(condition_number))
        else:
            conf_cov = 0.0

        confidences = {}
        if date not in self.adjusted_weights.index:
            # 没有当日数据，返回每个资产置信度为0
            for asset in self.full_weights.columns:
                confidences[asset] = 0.0
            return confidences

        weights_today = self.adjusted_weights.loc[date]

        for asset in self.full_weights.columns:
            cvar_val = self.rolling_cvar.get((date, asset), np.nan)
            cvar_th = self.cvar_thresh_pct
            downvol_val = self.rolling_downvol.get((date, asset), np.nan)
            downvol_th = self.down_thresh_pct

            # 计算比值
            cvar_ratio = max(0, (cvar_val - cvar_th) / cvar_th) if not np.isnan(cvar_val) and cvar_th != 0 else 0
            downvol_ratio = max(0, (downvol_val - downvol_th) / downvol_th) if not np.isnan(downvol_val) and downvol_th != 0 else 0

            beta, gamma = 0.5, 0.5  # 可调参数

            # 计算置信度（含clip至1）
            confidence = conf_cov * (1 + beta * cvar_ratio) * (1 + gamma * downvol_ratio)
            confidence = min(confidence, 1.0)  # 限制最大值为1

            confidences[asset] = confidence
        
        return confidences




    def get_embedding(self, market_state):
        date = market_state['date']
        cvar = self.rolling_cvar.get(date, np.nan)
        downvol = self.rolling_downvol.get(date, np.nan)
        cvar_thresh = self.cvar_thresh_pct
        downvol_thresh = self.down_thresh_pct
        # 动态特征：当日指标及其比值
        dynamic = [
            cvar, cvar/cvar_thresh if cvar_thresh!=0 else np.nan,
            downvol, downvol/downvol_thresh if downvol_thresh!=0 else np.nan
        ]
        # 拼接静态和动态特征
        static = [self.cvar_alpha, self.cvar_window, self.cvar_thresh_pct, self.down_window, self.down_thresh_pct]
        return np.array(dynamic + static)