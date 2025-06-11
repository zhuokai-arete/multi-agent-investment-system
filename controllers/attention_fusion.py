import numpy as np
from agents.base_agent import BaseAgent

class AttentionFusionAgent(BaseAgent):
    def __init__(self, agents):
        """
        初始化 Attention 融合器
        :param agents: 需要融合的 agent 列表
        """
        self.agents = agents

    def get_decision(self, market_state):
        '''
        步骤：
        1️⃣ 获取每个子代理的 embedding（暂时未使用但可能有拓展用途）。
        2️⃣ 收集每个代理的 confidence（信心分数），用于后续加权。
        3️⃣ 调用 softmax 对 confidence 序列进行归一化，生成权重（weights）。
        4️⃣ 获取所有代理的 decision（决策向量），按 weights 加权平均。
        5️⃣ 返回融合后的决策向量（list）。
        逻辑：通过 softmax 加权，强调置信度高的代理，形成动态加权的“注意力融合”。
        '''
        embeddings = np.array([agent.get_embedding(market_state) for agent in self.agents])
        # 这里的 get_confidence 调用的不是 AttentionFusionAgent 自己的 get_confidence 方法，
        # 而是 agents 列表里每个具体子代理的 get_confidence 方法。
        confidences = np.array([agent.get_confidence(market_state) for agent in self.agents])
        weights = self.softmax(confidences)
        decisions = np.array([agent.get_decision(market_state) for agent in self.agents])
        fused_decision = np.average(decisions, axis=0, weights=weights)
        return fused_decision.tolist()

    def get_confidence(self, market_state):
        '''
        对所有代理的 confidence 按 softmax 权重加权平均，得到融合信心。
        输出为一个浮点值（整体置信度）。
        '''
        confidences = np.array([agent.get_confidence(market_state) for agent in self.agents])
        weights = self.softmax(confidences)
        return float(np.average(confidences, weights=weights))

    '''
    为什么get_decision和get_confidence中都出现了confidences = np.array([agent.get_confidence(market_state) for agent in self.agents])
    这是否是一种重复？
    在**get_decision**中：
        confidences 是用来计算 softmax 权重（weights），用于加权各个代理的 决策向量（decision）。
        不考虑单独的置信度，而是把 confidence 当作权重生成器。
    在**get_confidence**中：
        confidences 不只是生成权重，而是融合所有代理的 信心值本身，形成一个单一的总体信心分数（浮点数）。
        也就是这里不仅仅是“权重生成器”，还直接作为融合对象。
    核心区别：
        get_decision：最终输出是决策向量，加权用的是置信度。
        get_confidence：最终输出就是融合后的置信度（浮点），加权用的是置信度。
    '''

    def get_embedding(self, market_state):
        '''
        类似 get_decision，但融合的对象是 embedding（向量特征表示）。
        可以将多个代理的特征编码融合为一个整体特征。
        '''
        embeddings = np.array([agent.get_embedding(market_state) for agent in self.agents])
        confidences = np.array([agent.get_confidence(market_state) for agent in self.agents])
        weights = self.softmax(confidences)
        fused_embedding = np.average(embeddings, axis=0, weights=weights)
        return fused_embedding.tolist()

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
