from data_utils.data_loader import load_market_data
from agents.RiskParityAgent import RiskParityAgent
from agents.CVaRDownsideVolAgent import CVaRDownsideVolAgent
from agents.DRLAgent import PPOAgentContinuous
from controllers.debate_room import DebateRoomAgent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
print("✅ Main: 开始加载数据")
returns, available_assets, rolling_cov_matrices, benchmark, feature_data, asset_list = load_market_data("资产池.xlsx", "中证多资产风险平价指数.xlsx")

print("✅ Main: 初始化 RiskParityAgent")
risk_agent = RiskParityAgent(returns, available_assets, rolling_cov_matrices)
print("✅ Main: 初始化 CVaRDownsideVolAgent")
cvar_agent = CVaRDownsideVolAgent(returns, available_assets, risk_agent.get_full_weights(), rolling_cov_matrices)
state_dim = feature_data.shape[1] + 2 * len(asset_list)
action_dim = len(asset_list)
print("✅ Main: 初始化 PPOAgentContinuous")
drl_agent = PPOAgentContinuous(state_dim=state_dim, action_dim=action_dim, action_asset_list=asset_list)

train_end_date, test_start_date = '2020-12-31', '2021-01-04'
train_dates = returns.index[returns.index <= train_end_date]
test_dates = returns.index[returns.index >= test_start_date]
train_features, test_features = feature_data.loc[train_dates], feature_data.loc[test_dates]

# 构造训练数据
states, actions, rewards, masks = [], [], [], []
portfolio_value_prev = 1.0
for date in train_dates:
    market_state = {'date': date}
    feature_vector = feature_data.loc[date].values.tolist()
    rp_decision = risk_agent.get_decision(market_state).reindex(returns.columns, fill_value=0)
    cvar_decision = cvar_agent.get_decision(market_state).reindex(returns.columns, fill_value=0)

    ppo_input = np.concatenate([feature_vector, rp_decision.values, cvar_decision.values])
    ppo_return = np.sum(returns.loc[date].fillna(0) * rp_decision)  # 这里仍使用 rp_return 模拟
    volatility = np.std(returns.loc[date].fillna(0))
    cost = 0.001
    rule_return = np.sum(returns.loc[date].fillna(0) * ((rp_decision + cvar_decision) / 2))
    alpha, lambd = 0.1, 0.05
    portfolio_value_now = portfolio_value_prev * (1 + ppo_return)
    reward = np.log(portfolio_value_now / portfolio_value_prev) - lambd * volatility - cost + alpha * (ppo_return - rule_return)
    portfolio_value_prev = portfolio_value_now

    # ✅ 使用平均策略作为训练动作（连续向量）
    combined_decision = ((rp_decision + cvar_decision) / 2).values
    states.append(ppo_input)
    actions.append(combined_decision)
    rewards.append(reward)
    masks.append(1)

# ✅ 构建连续策略下的训练数据（四元组）
trajectories = list(zip(states, actions, rewards, masks))
print("🚀 开始训练 PPOAgentContinuous")
drl_agent.train(trajectories)


def run_evaluation(dates, features):
    daily_returns = []
    for date in dates:
        feature_vector = features.loc[date].values
        rp_decision = risk_agent.get_decision({'date': date}).reindex(returns.columns, fill_value=0).values
        cvar_decision = cvar_agent.get_decision({'date': date}).reindex(returns.columns, fill_value=0).values
        market_state_full = {
            'feature_vector': feature_vector,
            'rp_decision': rp_decision,
            'cvar_decision': cvar_decision,
            'date': date
        }
        decision = drl_agent.get_decision(market_state_full)  # 采用方法1
        decision = decision.reindex(returns.columns, fill_value=0)
        daily_return = np.sum(returns.loc[date].fillna(0) * decision)
        daily_returns.append(daily_return)
    return daily_returns


def calculate_metrics(daily_returns):
    daily_returns = np.array(daily_returns)
    annual_return = np.mean(daily_returns) * 252
    annual_volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
    cumulative = np.cumprod(1 + daily_returns)
    max_drawdown = np.max(np.maximum.accumulate(cumulative) - cumulative) / np.max(np.maximum.accumulate(cumulative))
    return annual_return, annual_volatility, sharpe_ratio, max_drawdown

train_returns = run_evaluation(train_dates, train_features)
test_returns = run_evaluation(test_dates, test_features)

train_metrics = calculate_metrics(train_returns)
test_metrics = calculate_metrics(test_returns)
print("📈 训练集：年化收益 {:.2%}, 年化波动 {:.2%}, 夏普比率 {:.2f}, 最大回撤 {:.2%}".format(*train_metrics))
print("📈 测试集：年化收益 {:.2%}, 年化波动 {:.2%}, 夏普比率 {:.2f}, 最大回撤 {:.2%}".format(*test_metrics))

print("✅ Main: 初始化 DebateRoomAgent")
controller = DebateRoomAgent([risk_agent, cvar_agent, drl_agent])
rp_returns, cvar_returns, drl_returns, debate_returns, dates_list = [], [], [], [], []
for date in returns.index:
    if date not in available_assets: continue
    market_state = {'date': date}
    daily_ret = returns.loc[date].fillna(0)
    rp_decision = risk_agent.get_decision(market_state).reindex(returns.columns, fill_value=0)
    cvar_decision = cvar_agent.get_decision(market_state).reindex(returns.columns, fill_value=0)
    ppo_input = np.concatenate([feature_data.loc[date].values, rp_decision.values, cvar_decision.values])
    drl_decision = pd.Series(drl_agent.get_action(ppo_input),index=drl_agent.action_asset_list).reindex(returns.columns, fill_value=0)

    feature_vector = feature_data.loc[date].values
    rp_decision = risk_agent.get_decision(market_state).reindex(returns.columns, fill_value=0).values
    cvar_decision = cvar_agent.get_decision(market_state).reindex(returns.columns, fill_value=0).values
    market_state_full = {
        'feature_vector': feature_vector,
        'rp_decision': rp_decision,
        'cvar_decision': cvar_decision,
        'date': date
    }
    debate_decision = controller.get_decision(market_state_full).reindex(returns.columns, fill_value=0)

    
    rp_returns.append(np.sum(daily_ret * rp_decision))
    cvar_returns.append(np.sum(daily_ret * cvar_decision))
    drl_returns.append(np.sum(daily_ret * drl_decision))
    debate_returns.append(np.sum(daily_ret * debate_decision))
    dates_list.append(date)

metrics_list = [
    ('RiskParity', calculate_metrics(rp_returns)),
    ('CVaR', calculate_metrics(cvar_returns)),
    ('PPO', calculate_metrics(drl_returns)),
    ('DebateRoom', calculate_metrics(debate_returns))
]
for name, metrics in metrics_list:
    print(f"📈 {name} : 年化收益 {metrics[0]:.2%}, 年化波动 {metrics[1]:.2%}, 夏普比率 {metrics[2]:.2f}, 最大回撤 {metrics[3]:.2%}")

plt.figure(figsize=(12, 6))
cumulative_curves = {
    'RiskParity': np.cumprod(1 + np.array(rp_returns)),
    'CVaR': np.cumprod(1 + np.array(cvar_returns)),
    'PPO': np.cumprod(1 + np.array(drl_returns)),
    'DebateRoom': np.cumprod(1 + np.array(debate_returns))
}
for label, curve in cumulative_curves.items():
    plt.plot(dates_list, curve, label=label)
plt.xlabel('Date')
plt.ylabel('Cumulative Net Value')
plt.title('Cumulative Performance Comparison')
plt.legend()
plt.grid(True)
plt.savefig('agent_performance_comparison.png')
plt.show()
print("✅ 所有结果输出与图形绘制完毕")



