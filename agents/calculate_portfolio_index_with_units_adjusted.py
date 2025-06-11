import pandas as pd

def calculate_portfolio_index_with_units(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    available_assets: dict[str, list[str]] = None,
    cvar_flags: pd.Series = None,
    reentry_flags: pd.Series = None,
    initial_value: float = 100
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    计算真实净值（资产单位×价格 + 现金），支持 available_assets。
    返回净值、现金序列、单位数。
    """
    portfolio_index = pd.Series(index=prices.index, dtype=float)
    cash_series = pd.Series(index=prices.index, dtype=float)
    asset_units = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    cash = 0
    portfolio_index.iloc[0] = initial_value
    cash_series.iloc[0] = 0

    # 初始建仓日可用资产
    if available_assets is not None:
        valid_assets = available_assets.get(prices.index[0], [])
        weights_today = weights.loc[prices.index[0], valid_assets].copy()
        weights_today /= weights_today.sum()
        prices_today = prices.loc[prices.index[0], valid_assets]
        asset_units.loc[prices.index[0], valid_assets] = (initial_value * weights_today) / prices_today
        asset_units = asset_units.fillna(0)
    else:
        weights_today = weights.iloc[0]
        prices_today = prices.iloc[0]
        asset_units.iloc[0] = (initial_value * weights_today) / prices_today

    for i in range(1, len(prices)):
        date = prices.index[i]
        prev_date = prices.index[i - 1]
        price_today = prices.iloc[i]
        weight_today = weights.loc[date]

        asset_units.iloc[i] = asset_units.iloc[i - 1]
        cash_series.iloc[i] = cash_series.iloc[i - 1]

        if available_assets is not None:
            valid_assets = available_assets.get(date, [])
        else:
            valid_assets = prices.columns.tolist()

        # 清仓
        if cvar_flags is not None and cvar_flags.loc[date]:
            cash = (asset_units.iloc[i][valid_assets] * price_today[valid_assets]).sum() + cash_series.iloc[i]
            asset_units.iloc[i] = 0
            cash_series.iloc[i] = cash
            portfolio_index.iloc[i] = cash
            continue

        # 重建
        if reentry_flags is not None and reentry_flags.loc[date]:
            weight_today = weight_today[valid_assets].copy()
            weight_today /= weight_today.sum()
            cash = cash_series.iloc[i]
            asset_units.iloc[i] = 0
            asset_units.iloc[i][valid_assets] = (cash * weight_today) / price_today[valid_assets]
            cash_series.iloc[i] = 0

        # 正常净值
        portfolio_value = (asset_units.iloc[i][valid_assets] * price_today[valid_assets]).sum() + cash_series.iloc[i]
        portfolio_index.iloc[i] = portfolio_value

    return portfolio_index, cash_series, asset_units
