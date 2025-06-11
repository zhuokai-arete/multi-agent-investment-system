# Multi-Agent Investment System

A modular, extensible portfolio management system integrating multiple investment strategies, including Risk Parity, CVaR-based risk control, and reinforcement learning (PPO). The system is designed to support strategy diversity, dynamic decision fusion, and future expansion into cooperative/competitive multi-agent architectures.

## 📌 Project Overview

This project explores how rule-based strategies and reinforcement learning agents can be integrated into a multi-agent investment decision-making framework. The system is structured to allow comparative evaluation, strategy fusion, and scalable extension toward more complex agent interactions.

## ⚙️ Core Components

- `RiskParityAgent`: Allocates asset weights by minimizing portfolio variance under a risk parity constraint.
- `CVaRDownsideVolAgent`: Implements tail-risk-aware decisions using Conditional Value at Risk and downside volatility, with automatic cash-holding triggers.
- `PPOAgent`: A continuous-action actor-critic agent trained with portfolio-level reward signals, including return, risk penalty, and transaction costs.

## 🧠 Current Capabilities

- Modular agent interface and unified control loop via `DebateRoom` controller.
- Full training + evaluation loop, supporting strategy performance comparison across train/test sets.
- Multi-metric visualization: cumulative return, rolling Sharpe, drawdown, volatility, MCRT chart, and weight dynamics.
- Market data preprocessing and reward function customization.

## 🚧 In Progress

- Multi-agent coordination: planned implementation of attention-based cooperation and adversarial signals.
- Strategy diversity boosting via agent-type heterogeneity and explicit diversity regularization.
- Long-term goal: simulate human-like investing logic under market regime shifts via interacting agents.

## 📊 Sample Output (Coming Soon)

## 📁 Folder Structure (Planned)

📦 multi-agent-investment-system  
│  
├── agents/                             # 🧠 多种投资策略智能体模块  
│   ├── base_agent.py                       # 智能体通用接口抽象类  
│   ├── RiskParityAgent.py                  # 基于风险平价策略的权重分配Agent  
│   ├── CVaRDownsideVolAgent.py             # 使用CVaR和下行波动率控制尾部风险的Agent  
│   ├── DRLAgent.py                         # PPO算法实现的Actor-Critic智能体  
│   └── calculate_portfolio_index_with_units_adjusted.py  # 组合净值计算工具  
│  
├── controllers/                       # 🧩 智能体策略融合控制器  
│   ├── debate_room.py                     # 多Agent策略集成与统一调度主控模块  
│   └── attention_fusion.py                # Attention机制驱动的Agent协同模块（开发中）  
│  
├── configs/                          # ⚙️ 配置文件目录（训练参数等）  
│  
├── data/                             # 📊 数据源与实验输出目录  
│   ├── 资产池.xlsx                         # 多资产历史价格数据  
│   ├── 中证多资产风险平价指数.xlsx           # 用于对比分析的实际指数  
│   └── agent_performance_comparison.png   # 多策略净值表现对比图  
│  
├── utils/                            # 🔧 辅助模块  
│   └── llm_tools.py                      # 预留用于未来 LLM Agent 的接口模块  
│  
├── main.py                           # 🚀 系统主运行入口（训练+评估+可视化）  
│  
├── README.md                         # 📘 项目说明文档  
└── requirements.txt                  # 📦 Python运行环境依赖清单  

