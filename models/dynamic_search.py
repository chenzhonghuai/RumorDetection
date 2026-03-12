"""
动态搜索增强模块
实现智能搜索触发、多策略查询生成、来源可信度评估
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass


@dataclass
class SearchResult:
    """搜索结果数据类"""
    title: str
    snippet: str
    url: str
    domain: str
    publish_date: Optional[str] = None
    
    
class SearchTriggerDecision(nn.Module):
    """
    智能搜索触发决策模块
    综合评估知识图谱覆盖率、事件时效性和热点匹配度
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        threshold: float = 0.4,
        time_weight: float = 0.3,
        event_weight: float = 0.2
    ):
        super().__init__()
        self.threshold = threshold
        self.time_weight = time_weight
        self.event_weight = event_weight
        
        # 搜索必要性评估网络
        self.necessity_scorer = nn.Sequential(
            nn.Linear(embedding_dim + 3, hidden_dim),  # +3 for coverage, freshness, hot_event
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 时效性词汇嵌入
        self.time_keywords = [
            "今天", "昨天", "明天", "刚刚", "最新", "突发", "紧急",
            "今日", "昨日", "近日", "日前", "最近", "即将", "马上"
        ]
    
    def compute_freshness(self, text: str) -> float:
        """
        计算文本时效性得分
        
        Args:
            text: 输入文本
            
        Returns:
            时效性得分 [0, 1]
        """
        score = 0.0
        for keyword in self.time_keywords:
            if keyword in text:
                score += 0.15
        return min(score, 1.0)
    
    def check_hot_event(self, text: str, hot_topics: List[str]) -> float:
        """
        检查是否匹配热点事件
        
        Args:
            text: 输入文本
            hot_topics: 热点话题列表
            
        Returns:
            热点匹配得分
        """
        for topic in hot_topics:
            if topic in text:
                return 1.0
        return 0.0
    
    def forward(
        self,
        text_embedding: torch.Tensor,
        kg_coverage: torch.Tensor,
        text: str,
        hot_topics: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        决策是否触发搜索
        
        Args:
            text_embedding: 文本嵌入 [batch, dim]
            kg_coverage: 知识图谱覆盖率 [batch]
            text: 原始文本
            hot_topics: 热点话题列表
            
        Returns:
            决策结果字典
        """
        if hot_topics is None:
            hot_topics = []
        
        # 计算各项指标
        coverage_need = 1 - kg_coverage  # 覆盖不足程度
        freshness = torch.tensor([self.compute_freshness(text)], device=text_embedding.device)
        hot_event = torch.tensor([self.check_hot_event(text, hot_topics)], device=text_embedding.device)
        
        # 拼接特征
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
        
        features = torch.cat([
            text_embedding,
            coverage_need.unsqueeze(-1) if coverage_need.dim() == 0 else coverage_need.unsqueeze(-1),
            freshness.unsqueeze(-1) if freshness.dim() == 0 else freshness.unsqueeze(-1),
            hot_event.unsqueeze(-1) if hot_event.dim() == 0 else hot_event.unsqueeze(-1)
        ], dim=-1)
        
        # 计算搜索必要性得分
        search_need = self.necessity_scorer(features).squeeze(-1)
        
        # 决策
        trigger = search_need > self.threshold
        
        return {
            "search_need": search_need,
            "trigger": trigger,
            "coverage_need": coverage_need,
            "freshness": freshness,
            "hot_event": hot_event
        }


class QueryGenerator(nn.Module):
    """
    多策略查询生成模块
    生成声明提取、实体扩展、上下文和反向验证四类查询
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # 查询类型嵌入
        self.query_type_embedding = nn.Embedding(4, embedding_dim)
        
        # 查询生成网络
        self.query_generator = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def generate_claim_query(self, text: str) -> str:
        """
        生成声明提取查询
        提取文本中的核心声明作为查询
        """
        # 简化处理：去除无关词汇，保留核心内容
        # 实际应用中可使用更复杂的NLP处理
        stopwords = ["的", "了", "是", "在", "有", "和", "与", "或", "但", "而"]
        query = text
        for word in stopwords:
            query = query.replace(word, " ")
        query = " ".join(query.split()[:10])  # 限制长度
        return query.strip()
    
    def generate_entity_query(self, text: str, entities: List[str]) -> str:
        """
        生成实体扩展查询
        基于识别的实体生成查询
        """
        if entities:
            return " ".join(entities[:3])  # 使用前3个实体
        return self.generate_claim_query(text)
    
    def generate_context_query(self, text: str) -> str:
        """
        生成上下文查询
        添加验证相关的上下文词
        """
        context_words = ["真假", "辟谣", "核实", "证实"]
        base_query = self.generate_claim_query(text)
        return f"{base_query} {context_words[0]}"
    
    def generate_reverse_query(self, text: str) -> str:
        """
        生成反向验证查询
        寻找可能的反驳证据
        """
        reverse_words = ["假的", "谣言", "不实", "辟谣"]
        base_query = self.generate_claim_query(text)
        return f"{base_query} {reverse_words[0]}"
    
    def forward(
        self,
        text: str,
        text_embedding: torch.Tensor,
        entities: List[str] = None
    ) -> Dict[str, any]:
        """
        生成多策略查询
        
        Args:
            text: 原始文本
            text_embedding: 文本嵌入
            entities: 识别的实体列表
            
        Returns:
            查询字典
        """
        if entities is None:
            entities = []
        
        queries = {
            "claim": self.generate_claim_query(text),
            "entity": self.generate_entity_query(text, entities),
            "context": self.generate_context_query(text),
            "reverse": self.generate_reverse_query(text)
        }
        
        # 生成查询嵌入
        query_embeddings = {}
        for i, (query_type, query_text) in enumerate(queries.items()):
            type_emb = self.query_type_embedding(
                torch.tensor([i], device=text_embedding.device)
            )
            if text_embedding.dim() == 1:
                text_embedding = text_embedding.unsqueeze(0)
            combined = torch.cat([text_embedding, type_emb], dim=-1)
            query_emb = self.query_generator(combined)
            query_embeddings[query_type] = query_emb
        
        return {
            "queries": queries,
            "query_embeddings": query_embeddings
        }


class SourceCredibilityEvaluator(nn.Module):
    """
    来源可信度评估模块
    从权威性、专业性、技术性和一致性四个维度评估
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        authority_weight: float = 0.35,
        expertise_weight: float = 0.25,
        technical_weight: float = 0.20,
        consistency_weight: float = 0.20
    ):
        super().__init__()
        self.authority_weight = authority_weight
        self.expertise_weight = expertise_weight
        self.technical_weight = technical_weight
        self.consistency_weight = consistency_weight
        
        # 权威域名得分
        self.authority_domains = {
            "gov.cn": 1.0,
            "xinhuanet.com": 0.9,
            "people.com.cn": 0.9,
            "cctv.com": 0.9,
            "chinanews.com": 0.85,
            "thepaper.cn": 0.8,
            "weibo.com": 0.6,
            "zhihu.com": 0.5,
            "baidu.com": 0.4
        }
        
        # 专业性评估网络
        self.expertise_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 一致性评估网络
        self.consistency_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 虚假信息检测
        self.fake_detector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def evaluate_authority(self, domain: str) -> float:
        """
        评估来源权威性
        
        Args:
            domain: 域名
            
        Returns:
            权威性得分
        """
        for auth_domain, score in self.authority_domains.items():
            if auth_domain in domain:
                return score
        return 0.3  # 默认得分
    
    def evaluate_technical(self, url: str) -> float:
        """
        评估技术性指标
        
        Args:
            url: URL
            
        Returns:
            技术性得分
        """
        score = 0.5
        
        # HTTPS加分
        if url.startswith("https"):
            score += 0.2
        
        # 短链接减分
        if len(url) < 30:
            score -= 0.1
        
        return min(max(score, 0), 1)
    
    def forward(
        self,
        result_embeddings: torch.Tensor,
        claim_embedding: torch.Tensor,
        results: List[SearchResult],
        high_credibility_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        评估搜索结果的可信度
        
        Args:
            result_embeddings: 搜索结果嵌入 [num_results, dim]
            claim_embedding: 声明嵌入 [dim]
            results: 搜索结果列表
            high_credibility_embeddings: 高可信来源嵌入（用于一致性评估）
            
        Returns:
            可信度评估结果
        """
        num_results = len(results)
        device = result_embeddings.device
        
        # 1. 权威性评估
        authority_scores = torch.tensor(
            [self.evaluate_authority(r.domain) for r in results],
            device=device
        )
        
        # 2. 专业性评估
        claim_expanded = claim_embedding.unsqueeze(0).expand(num_results, -1)
        expertise_input = torch.cat([result_embeddings, claim_expanded], dim=-1)
        expertise_scores = self.expertise_scorer(expertise_input).squeeze(-1)
        
        # 3. 技术性评估
        technical_scores = torch.tensor(
            [self.evaluate_technical(r.url) for r in results],
            device=device
        )
        
        # 4. 一致性评估
        if high_credibility_embeddings is not None:
            # 与高可信来源的一致性
            consistency_input = torch.cat([
                result_embeddings,
                high_credibility_embeddings.mean(dim=0, keepdim=True).expand(num_results, -1)
            ], dim=-1)
            consistency_scores = self.consistency_scorer(consistency_input).squeeze(-1)
        else:
            consistency_scores = torch.ones(num_results, device=device) * 0.5
        
        # 5. 虚假信息概率
        fake_probs = self.fake_detector(result_embeddings).squeeze(-1)
        
        # 综合可信度
        credibility = (
            self.authority_weight * authority_scores +
            self.expertise_weight * expertise_scores +
            self.technical_weight * technical_scores +
            self.consistency_weight * consistency_scores
        ) * (1 - fake_probs)
        
        return {
            "credibility": credibility,
            "authority_scores": authority_scores,
            "expertise_scores": expertise_scores,
            "technical_scores": technical_scores,
            "consistency_scores": consistency_scores,
            "fake_probs": fake_probs
        }


class EvidenceAggregator(nn.Module):
    """
    证据聚合模块
    不确定性感知的加权聚合
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # 证据编码器
        self.evidence_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 不确定性估计
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # 确保正值
        )
        
        # 注意力聚合
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(
        self,
        evidence_embeddings: torch.Tensor,
        credibility_scores: torch.Tensor,
        claim_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        聚合搜索证据
        
        Args:
            evidence_embeddings: 证据嵌入 [num_evidence, dim]
            credibility_scores: 可信度得分 [num_evidence]
            claim_embedding: 声明嵌入 [dim]
            
        Returns:
            聚合结果字典
        """
        # 编码证据
        encoded = self.evidence_encoder(evidence_embeddings)
        
        # 估计不确定性
        uncertainty = self.uncertainty_estimator(encoded).squeeze(-1)
        
        # 可信度加权
        weights = credibility_scores / (uncertainty + 1e-6)
        weights = F.softmax(weights, dim=0)
        
        # 加权聚合
        weighted_evidence = (encoded * weights.unsqueeze(-1)).sum(dim=0)
        
        # 注意力增强
        if claim_embedding.dim() == 1:
            claim_embedding = claim_embedding.unsqueeze(0)
        
        encoded_expanded = encoded.unsqueeze(0)  # [1, num_evidence, dim]
        claim_expanded = claim_embedding.unsqueeze(0)  # [1, 1, dim]
        
        attended, attn_weights = self.attention(
            claim_expanded, encoded_expanded, encoded_expanded
        )
        attended = attended.squeeze(0).squeeze(0)
        
        # 融合
        aggregated = (weighted_evidence + attended) / 2
        
        # 计算整体不确定性
        overall_uncertainty = (uncertainty * weights).sum()
        
        return {
            "aggregated_evidence": aggregated,
            "evidence_weights": weights,
            "uncertainty": overall_uncertainty,
            "attention_weights": attn_weights.squeeze()
        }


class DynamicSearchModule(nn.Module):
    """
    动态搜索增强模块
    整合搜索触发、查询生成、可信度评估和证据聚合
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        search_threshold: float = 0.4,
        credibility_threshold: float = 0.5
    ):
        super().__init__()
        self.search_threshold = search_threshold
        self.credibility_threshold = credibility_threshold
        
        # 子模块
        self.trigger_decision = SearchTriggerDecision(
            embedding_dim=embedding_dim,
            threshold=search_threshold
        )
        self.query_generator = QueryGenerator(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        self.credibility_evaluator = SourceCredibilityEvaluator(
            embedding_dim=embedding_dim
        )
        self.evidence_aggregator = EvidenceAggregator(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        
        # 搜索结果编码器
        self.result_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(
        self,
        text: str,
        text_embedding: torch.Tensor,
        kg_coverage: torch.Tensor,
        entities: List[str] = None,
        search_results: List[SearchResult] = None,
        result_embeddings: torch.Tensor = None,
        hot_topics: List[str] = None
    ) -> Dict[str, any]:
        """
        动态搜索增强前向传播
        
        Args:
            text: 原始文本
            text_embedding: 文本嵌入
            kg_coverage: 知识图谱覆盖率
            entities: 识别的实体
            search_results: 搜索结果（如果已执行搜索）
            result_embeddings: 搜索结果嵌入
            hot_topics: 热点话题
            
        Returns:
            搜索增强结果
        """
        results = {}
        
        # 1. 搜索触发决策
        trigger_result = self.trigger_decision(
            text_embedding, kg_coverage, text, hot_topics
        )
        results["trigger_decision"] = trigger_result
        
        if not trigger_result["trigger"].item():
            # 不触发搜索，返回空的搜索表示
            results["search_embedding"] = torch.zeros_like(text_embedding)
            results["search_triggered"] = False
            return results
        
        results["search_triggered"] = True
        
        # 2. 生成查询
        query_result = self.query_generator(text, text_embedding, entities)
        results["queries"] = query_result["queries"]
        
        # 3. 如果有搜索结果，进行可信度评估和证据聚合
        if search_results is not None and result_embeddings is not None:
            # 编码搜索结果
            encoded_results = self.result_encoder(result_embeddings)
            
            # 可信度评估
            credibility_result = self.credibility_evaluator(
                encoded_results,
                text_embedding.squeeze(0) if text_embedding.dim() > 1 else text_embedding,
                search_results
            )
            results["credibility"] = credibility_result
            
            # 过滤低可信度结果
            valid_mask = credibility_result["credibility"] > self.credibility_threshold
            if valid_mask.sum() > 0:
                valid_embeddings = encoded_results[valid_mask]
                valid_credibility = credibility_result["credibility"][valid_mask]
                
                # 证据聚合
                aggregation_result = self.evidence_aggregator(
                    valid_embeddings,
                    valid_credibility,
                    text_embedding
                )
                results["aggregation"] = aggregation_result
                results["search_embedding"] = aggregation_result["aggregated_evidence"]
                results["num_valid_results"] = valid_mask.sum().item()
            else:
                results["search_embedding"] = torch.zeros_like(text_embedding.squeeze(0))
                results["num_valid_results"] = 0
        else:
            results["search_embedding"] = torch.zeros_like(text_embedding.squeeze(0))
        
        return results
