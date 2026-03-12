"""
多维一致性验证模块
从实体属性、关系结构、时序逻辑、常识符合度四个层次验证声明可信度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class EntityAttributeVerifier(nn.Module):
    """
    实体属性一致性验证
    验证声明中的实体属性是否与知识图谱一致
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # 属性匹配网络
        self.attr_matcher = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 属性类型权重
        self.attr_type_weights = nn.Parameter(torch.ones(50))  # 假设50种属性类型
    
    def forward(
        self,
        claim_attr_emb: torch.Tensor,
        kg_attr_emb: torch.Tensor,
        attr_types: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        验证属性一致性
        
        Args:
            claim_attr_emb: 声明中的属性嵌入 [batch, num_attrs, dim]
            kg_attr_emb: 知识图谱中的属性嵌入 [batch, num_attrs, dim]
            attr_types: 属性类型 [batch, num_attrs]
            
        Returns:
            验证结果字典
        """
        # 计算属性匹配得分
        combined = torch.cat([claim_attr_emb, kg_attr_emb], dim=-1)
        match_scores = self.attr_matcher(combined).squeeze(-1)  # [batch, num_attrs]
        
        # 应用属性类型权重
        type_weights = self.attr_type_weights[attr_types]
        weighted_scores = match_scores * type_weights
        
        # 聚合得分
        consistency_score = torch.sigmoid(weighted_scores.mean(dim=-1))
        
        # 检测矛盾
        contradictions = (match_scores < 0).float()
        contradiction_ratio = contradictions.mean(dim=-1)
        
        return {
            "attr_consistency": consistency_score,
            "attr_match_scores": match_scores,
            "contradiction_ratio": contradiction_ratio
        }


class RelationStructureVerifier(nn.Module):
    """
    关系结构一致性验证
    验证声明中的实体关系是否与知识图谱结构一致
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_relations: int = 100
    ):
        super().__init__()
        
        # 关系嵌入
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        
        # 三元组验证网络
        self.triple_verifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # 路径验证网络（验证多跳关系）
        self.path_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.path_verifier = nn.Linear(hidden_dim, 1)
    
    def verify_triple(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        验证单个三元组
        
        Args:
            head_emb: 头实体嵌入 [batch, dim]
            relation_id: 关系ID [batch]
            tail_emb: 尾实体嵌入 [batch, dim]
            
        Returns:
            验证得分 [batch]
        """
        relation_emb = self.relation_embedding(relation_id)
        combined = torch.cat([head_emb, relation_emb, tail_emb], dim=-1)
        score = self.triple_verifier(combined).squeeze(-1)
        return torch.sigmoid(score)
    
    def verify_path(
        self,
        path_embeddings: torch.Tensor,
        path_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        验证关系路径
        
        Args:
            path_embeddings: 路径嵌入序列 [batch, max_len, dim]
            path_lengths: 路径长度 [batch]
            
        Returns:
            路径验证得分 [batch]
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            path_embeddings, path_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.path_encoder(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        score = self.path_verifier(hidden).squeeze(-1)
        return torch.sigmoid(score)
    
    def forward(
        self,
        claim_triples: Dict[str, torch.Tensor],
        kg_structure: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        验证关系结构一致性
        
        Args:
            claim_triples: 声明中的三元组信息
            kg_structure: 知识图谱结构信息
            
        Returns:
            验证结果字典
        """
        # 验证直接三元组
        triple_scores = self.verify_triple(
            claim_triples["head_emb"],
            claim_triples["relation_ids"],
            claim_triples["tail_emb"]
        )
        
        # 验证推理路径（如果存在）
        if "path_embeddings" in kg_structure:
            path_scores = self.verify_path(
                kg_structure["path_embeddings"],
                kg_structure["path_lengths"]
            )
        else:
            path_scores = torch.ones_like(triple_scores)
        
        # 综合得分
        structure_consistency = (triple_scores + path_scores) / 2
        
        return {
            "structure_consistency": structure_consistency,
            "triple_scores": triple_scores,
            "path_scores": path_scores
        }


class TemporalLogicVerifier(nn.Module):
    """
    时序逻辑一致性验证
    验证声明中的时间信息是否符合逻辑
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # 时间编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),  # year, month, day, hour, minute, second
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # 时序关系验证
        self.temporal_verifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # before, after, concurrent
        )
        
        # 时效性评估
        self.freshness_scorer = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def encode_time(self, time_features: torch.Tensor) -> torch.Tensor:
        """
        编码时间特征
        
        Args:
            time_features: 时间特征 [batch, 6]
            
        Returns:
            时间嵌入 [batch, embedding_dim]
        """
        return self.time_encoder(time_features)
    
    def verify_temporal_order(
        self,
        event1_emb: torch.Tensor,
        event2_emb: torch.Tensor,
        time1_emb: torch.Tensor,
        time2_emb: torch.Tensor,
        expected_order: torch.Tensor
    ) -> torch.Tensor:
        """
        验证时序顺序
        
        Args:
            event1_emb: 事件1嵌入
            event2_emb: 事件2嵌入
            time1_emb: 时间1嵌入
            time2_emb: 时间2嵌入
            expected_order: 期望的时序关系 (0: before, 1: after, 2: concurrent)
            
        Returns:
            验证得分
        """
        combined = torch.cat([
            event1_emb + time1_emb,
            event2_emb + time2_emb,
            time1_emb - time2_emb
        ], dim=-1)
        
        logits = self.temporal_verifier(combined)
        probs = F.softmax(logits, dim=-1)
        
        # 获取期望顺序的概率
        batch_indices = torch.arange(expected_order.size(0), device=expected_order.device)
        order_prob = probs[batch_indices, expected_order]
        
        return order_prob
    
    def forward(
        self,
        claim_events: Dict[str, torch.Tensor],
        kg_events: Dict[str, torch.Tensor],
        current_time: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        验证时序逻辑一致性
        
        Args:
            claim_events: 声明中的事件信息
            kg_events: 知识图谱中的事件信息
            current_time: 当前时间特征
            
        Returns:
            验证结果字典
        """
        # 编码时间
        claim_time_emb = self.encode_time(claim_events["time_features"])
        kg_time_emb = self.encode_time(kg_events["time_features"])
        current_time_emb = self.encode_time(current_time)
        
        # 验证时序顺序
        if "expected_order" in claim_events:
            order_score = self.verify_temporal_order(
                claim_events["event_emb"],
                kg_events["event_emb"],
                claim_time_emb,
                kg_time_emb,
                claim_events["expected_order"]
            )
        else:
            order_score = torch.ones(claim_time_emb.size(0), device=claim_time_emb.device)
        
        # 评估时效性
        time_diff = (current_time_emb - claim_time_emb).norm(dim=-1, keepdim=True)
        freshness_input = torch.cat([claim_events["event_emb"], time_diff], dim=-1)
        freshness_score = self.freshness_scorer(freshness_input).squeeze(-1)
        
        # 综合得分
        temporal_consistency = (order_score + freshness_score) / 2
        
        return {
            "temporal_consistency": temporal_consistency,
            "order_score": order_score,
            "freshness_score": freshness_score
        }


class CommonsenseVerifier(nn.Module):
    """
    常识符合度验证
    验证声明是否符合常识推理
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # 常识推理网络
        self.reasoning_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 因果关系验证
        self.causal_verifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # cause, effect, no_relation
        )
        
        # 物理规律验证
        self.physics_verifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def verify_commonsense(
        self,
        claim_emb: torch.Tensor,
        commonsense_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        验证常识符合度
        
        Args:
            claim_emb: 声明嵌入
            commonsense_emb: 常识知识嵌入
            
        Returns:
            常识符合度得分
        """
        combined = torch.cat([claim_emb, commonsense_emb], dim=-1)
        score = self.reasoning_network(combined).squeeze(-1)
        return torch.sigmoid(score)
    
    def verify_causality(
        self,
        cause_emb: torch.Tensor,
        effect_emb: torch.Tensor,
        expected_relation: torch.Tensor
    ) -> torch.Tensor:
        """
        验证因果关系
        
        Args:
            cause_emb: 原因嵌入
            effect_emb: 结果嵌入
            expected_relation: 期望的因果关系
            
        Returns:
            因果验证得分
        """
        combined = torch.cat([cause_emb, effect_emb], dim=-1)
        logits = self.causal_verifier(combined)
        probs = F.softmax(logits, dim=-1)
        
        batch_indices = torch.arange(expected_relation.size(0), device=expected_relation.device)
        relation_prob = probs[batch_indices, expected_relation]
        
        return relation_prob
    
    def forward(
        self,
        claim_embedding: torch.Tensor,
        commonsense_knowledge: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        验证常识符合度
        
        Args:
            claim_embedding: 声明嵌入
            commonsense_knowledge: 常识知识信息
            
        Returns:
            验证结果字典
        """
        # 常识推理验证
        commonsense_score = self.verify_commonsense(
            claim_embedding,
            commonsense_knowledge["knowledge_emb"]
        )
        
        # 因果关系验证（如果存在）
        if "cause_emb" in commonsense_knowledge:
            causal_score = self.verify_causality(
                commonsense_knowledge["cause_emb"],
                commonsense_knowledge["effect_emb"],
                commonsense_knowledge["causal_relation"]
            )
        else:
            causal_score = torch.ones_like(commonsense_score)
        
        # 物理规律验证
        physics_score = self.physics_verifier(claim_embedding).squeeze(-1)
        
        # 综合得分
        commonsense_consistency = (commonsense_score + causal_score + physics_score) / 3
        
        return {
            "commonsense_consistency": commonsense_consistency,
            "reasoning_score": commonsense_score,
            "causal_score": causal_score,
            "physics_score": physics_score
        }


class MultiDimensionalConsistencyVerifier(nn.Module):
    """
    多维一致性验证框架
    整合四个层次的验证结果
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_relations: int = 100
    ):
        super().__init__()
        
        # 四个验证模块
        self.attr_verifier = EntityAttributeVerifier(embedding_dim, hidden_dim)
        self.struct_verifier = RelationStructureVerifier(embedding_dim, hidden_dim, num_relations)
        self.temporal_verifier = TemporalLogicVerifier(embedding_dim, hidden_dim)
        self.commonsense_verifier = CommonsenseVerifier(embedding_dim, hidden_dim)
        
        # 验证结果融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 可学习的层次权重
        self.layer_weights = nn.Parameter(torch.ones(4) / 4)
    
    def forward(
        self,
        claim_info: Dict[str, torch.Tensor],
        kg_info: Dict[str, torch.Tensor],
        current_time: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        多维一致性验证
        
        Args:
            claim_info: 声明信息字典
            kg_info: 知识图谱信息字典
            current_time: 当前时间（可选）
            
        Returns:
            验证结果字典
        """
        results = {}
        scores = []
        
        # 1. 实体属性验证
        if "attr_emb" in claim_info and "attr_emb" in kg_info:
            attr_result = self.attr_verifier(
                claim_info["attr_emb"],
                kg_info["attr_emb"],
                claim_info.get("attr_types", torch.zeros(1).long())
            )
            results["attr_verification"] = attr_result
            scores.append(attr_result["attr_consistency"])
        else:
            scores.append(torch.tensor([0.5]))
        
        # 2. 关系结构验证
        if "triples" in claim_info and "structure" in kg_info:
            struct_result = self.struct_verifier(
                claim_info["triples"],
                kg_info["structure"]
            )
            results["struct_verification"] = struct_result
            scores.append(struct_result["structure_consistency"])
        else:
            scores.append(torch.tensor([0.5]))
        
        # 3. 时序逻辑验证
        if "events" in claim_info and "events" in kg_info:
            if current_time is None:
                current_time = torch.zeros(claim_info["events"]["event_emb"].size(0), 6)
            temporal_result = self.temporal_verifier(
                claim_info["events"],
                kg_info["events"],
                current_time
            )
            results["temporal_verification"] = temporal_result
            scores.append(temporal_result["temporal_consistency"])
        else:
            scores.append(torch.tensor([0.5]))
        
        # 4. 常识符合度验证
        if "embedding" in claim_info and "commonsense" in kg_info:
            commonsense_result = self.commonsense_verifier(
                claim_info["embedding"],
                kg_info["commonsense"]
            )
            results["commonsense_verification"] = commonsense_result
            scores.append(commonsense_result["commonsense_consistency"])
        else:
            scores.append(torch.tensor([0.5]))
        
        # 融合四个层次的得分
        scores_tensor = torch.stack([s.mean() if s.dim() > 0 else s for s in scores])
        weights = F.softmax(self.layer_weights, dim=0)
        weighted_score = (scores_tensor * weights).sum()
        
        # 通过融合层
        fusion_input = scores_tensor.unsqueeze(0)
        final_score = self.fusion_layer(fusion_input).squeeze()
        
        results["consistency_scores"] = scores_tensor
        results["layer_weights"] = weights
        results["weighted_consistency"] = weighted_score
        results["final_consistency"] = final_score
        
        return results
