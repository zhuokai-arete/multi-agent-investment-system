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

