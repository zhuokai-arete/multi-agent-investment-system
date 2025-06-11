from agents.base_agent import BaseAgent
import numpy as np
import pandas as pd

class DebateRoomAgent(BaseAgent):
    def __init__(self, agents):
        """
        :param agents: List of BaseAgent
        """
        self.agents = agents

    def get_decision(self, market_state):
        assets = self.agents[0].get_full_weights().columns
        combined_decision = pd.Series(0.0, index=self.agents[0].get_full_weights().columns)  # 用完整资产列表初始化
        for agent in self.agents:
            decision = agent.get_decision(market_state)
            confidence = agent.get_confidence(market_state)
            # 对决策和置信度对齐
            decision_series = decision.reindex(assets).fillna(0) if isinstance(decision, pd.Series) else pd.Series(decision, index=assets)
            conf_vector = np.array([confidence.get(asset, 0.0) for asset in assets]) if isinstance(confidence, dict) else np.ones(len(assets)) * confidence
            combined_decision += conf_vector * decision_series.values

        return combined_decision


    def get_confidence(self, market_state):
        assets = self.agents[0].full_weights.columns
        total_confidence = {asset: 0.0 for asset in assets}
        count = len(self.agents)

        for agent in self.agents:
            conf = agent.get_confidence(market_state)
            if isinstance(conf, dict):
                for asset in assets:
                    total_confidence[asset] += conf.get(asset, 0.0)
            else:
                for asset in assets:
                    total_confidence[asset] += conf

        for asset in assets:
            total_confidence[asset] /= count
        return total_confidence

    def get_embedding(self, market_state):
        embeddings = np.array([agent.get_embedding(market_state) for agent in self.agents])
        return np.mean(embeddings, axis=0).tolist()
