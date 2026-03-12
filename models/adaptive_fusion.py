"""
自适应知识协同融合模块
动态融合静态多源知识图谱和动态搜索证据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class KnowledgeQualityAssessor(nn.Module):
    """
    知识质量评估模块
    评估静态知识和动态证据的质量
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # 静态知识质量评估
        self.static_quality_net = nn.Sequential(
            nn.Linear(embedding_dim + 3, hidden_dim),  # +3 for coverage, consistency, staleness
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 动态证据质量评估
        self.dynamic_quality_net = nn.Sequential(
            nn.Linear(embedding_dim + 3, hidden_dim),  # +3 for effective_ratio, avg_auth, conflict
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def assess_static_quality(
        self,
        kg_embedding: torch.Tensor,
        coverage: torch.Tensor,
        consistency: torch.Tensor,
        staleness: torch.Tensor
    ) -> torch.Tensor:
        """
        评估静态知识质量
        Q_static = coverage * consistency * (1 - staleness)
        
        Args:
            kg_embedding: 知识图谱嵌入
            coverage: 覆盖率
            consistency: 一致性得分
            staleness: 过时程度
            
        Returns:
            静态知识质量得分
        """
        # 确保维度正确
        if kg_embedding.dim() == 1:
            kg_embedding = kg_embedding.unsqueeze(0)
        
        features = torch.cat([
            kg_embedding,
            coverage.view(-1, 1),
            consistency.view(-1, 1),
            staleness.view(-1, 1)
        ], dim=-1)
        
        quality = self.static_quality_net(features).squeeze(-1)
        return quality
    
    def assess_dynamic_quality(
        self,
        search_embedding: torch.Tensor,
        effective_ratio: torch.Tensor,
        avg_credibility: torch.Tensor,
        conflict_ratio: torch.Tensor
    ) -> torch.Tensor:
        """
        评估动态证据质量
        Q_dynamic = effective_ratio * avg_credibility * (1 - conflict_ratio)
        
        Args:
            search_embedding: 搜索证据嵌入
            effective_ratio: 有效结果比例
            avg_credibility: 平均可信度
            conflict_ratio: 冲突比例
            
        Returns:
            动态证据质量得分
        """
        if search_embedding.dim() == 1:
            search_embedding = search_embedding.unsqueeze(0)
        
        features = torch.cat([
            search_embedding,
            effective_ratio.view(-1, 1),
            avg_credibility.view(-1, 1),
            conflict_ratio.view(-1, 1)
        ], dim=-1)
        
        quality = self.dynamic_quality_net(features).squeeze(-1)
        return quality


class GatedFusionMechanism(nn.Module):
    """
    门控融合机制
    通过可学习的门控动态平衡静态和动态知识
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # 全局门控（标量权重）
        self.global_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 2, hidden_dim),  # +2 for quality scores
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 元素级门控（向量权重）
        self.element_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Sigmoid()
        )
        
        # 融合后的变换
        self.fusion_transform = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(
        self,
        static_embedding: torch.Tensor,
        dynamic_embedding: torch.Tensor,
        static_quality: torch.Tensor,
        dynamic_quality: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        门控融合
        
        Args:
            static_embedding: 静态知识嵌入 [batch, dim]
            dynamic_embedding: 动态证据嵌入 [batch, dim]
            static_quality: 静态知识质量 [batch]
            dynamic_quality: 动态证据质量 [batch]
            
        Returns:
            融合结果字典
        """
        # 确保维度正确
        if static_embedding.dim() == 1:
            static_embedding = static_embedding.unsqueeze(0)
        if dynamic_embedding.dim() == 1:
            dynamic_embedding = dynamic_embedding.unsqueeze(0)
        
        # 计算全局门控权重 β
        global_input = torch.cat([
            static_embedding,
            dynamic_embedding,
            static_quality.view(-1, 1),
            dynamic_quality.view(-1, 1)
        ], dim=-1)
        beta = self.global_gate(global_input)  # [batch, 1]
        
        # 计算元素级门控向量 g
        element_input = torch.cat([static_embedding, dynamic_embedding], dim=-1)
        g = self.element_gate(element_input)  # [batch, dim]
        
        # 融合: v* = g ⊙ v_static + (1-g) ⊙ v_dynamic
        fused = g * static_embedding + (1 - g) * dynamic_embedding
        
        # 应用全局权重调整
        fused = beta * fused + (1 - beta) * static_embedding
        
        # 变换
        output = self.fusion_transform(fused)
        
        return {
            "fused_embedding": output,
            "global_gate": beta.squeeze(-1),
            "element_gate": g,
            "raw_fusion": fused
        }


class ConflictDetector(nn.Module):
    """
    冲突检测模块
    检测静态知识和动态证据之间的冲突
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # 冲突检测网络
        self.conflict_detector = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 冲突类型分类
        self.conflict_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # 5种冲突类型
        )
        
        # 冲突类型
        self.conflict_types = [
            "no_conflict",
            "factual_contradiction",
            "temporal_inconsistency",
            "source_disagreement",
            "partial_conflict"
        ]
    
    def forward(
        self,
        static_embedding: torch.Tensor,
        dynamic_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        检测冲突
        
        Args:
            static_embedding: 静态知识嵌入
            dynamic_embedding: 动态证据嵌入
            
        Returns:
            冲突检测结果
        """
        if static_embedding.dim() == 1:
            static_embedding = static_embedding.unsqueeze(0)
        if dynamic_embedding.dim() == 1:
            dynamic_embedding = dynamic_embedding.unsqueeze(0)
        
        combined = torch.cat([static_embedding, dynamic_embedding], dim=-1)
        
        # 检测是否存在冲突
        conflict_prob = self.conflict_detector(combined).squeeze(-1)
        
        # 分类冲突类型
        conflict_logits = self.conflict_classifier(combined)
        conflict_type_probs = F.softmax(conflict_logits, dim=-1)
        conflict_type = conflict_logits.argmax(dim=-1)
        
        return {
            "conflict_probability": conflict_prob,
            "conflict_type": conflict_type,
            "conflict_type_probs": conflict_type_probs,
            "has_conflict": conflict_prob > 0.5
        }


class ConflictResolver(nn.Module):
    """
    冲突解决模块
    根据质量差异解决静态和动态知识的冲突
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # 冲突解决网络
        self.resolver = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 2, hidden_dim),  # +2 for quality scores
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 置信度估计
        self.confidence_estimator = nn.Sequential(
            nn.Linear(embedding_dim + 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        static_embedding: torch.Tensor,
        dynamic_embedding: torch.Tensor,
        static_quality: torch.Tensor,
        dynamic_quality: torch.Tensor,
        conflict_info: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        解决冲突
        
        Args:
            static_embedding: 静态知识嵌入
            dynamic_embedding: 动态证据嵌入
            static_quality: 静态知识质量
            dynamic_quality: 动态证据质量
            conflict_info: 冲突检测信息
            
        Returns:
            解决后的结果
        """
        if static_embedding.dim() == 1:
            static_embedding = static_embedding.unsqueeze(0)
        if dynamic_embedding.dim() == 1:
            dynamic_embedding = dynamic_embedding.unsqueeze(0)
        
        # 根据质量选择主要来源
        quality_diff = static_quality - dynamic_quality
        
        # 解决冲突
        resolver_input = torch.cat([
            static_embedding,
            dynamic_embedding,
            static_quality.view(-1, 1),
            dynamic_quality.view(-1, 1)
        ], dim=-1)
        
        resolved = self.resolver(resolver_input)
        
        # 估计解决后的置信度
        confidence_input = torch.cat([
            resolved,
            static_quality.view(-1, 1),
            dynamic_quality.view(-1, 1)
        ], dim=-1)
        confidence = self.confidence_estimator(confidence_input).squeeze(-1)
        
        # 确定主要来源
        primary_source = torch.where(
            quality_diff > 0,
            torch.zeros_like(quality_diff),  # 0: static
            torch.ones_like(quality_diff)    # 1: dynamic
        )
        
        return {
            "resolved_embedding": resolved,
            "confidence": confidence,
            "primary_source": primary_source,
            "quality_difference": quality_diff
        }


class AdaptiveKnowledgeFusion(nn.Module):
    """
    自适应知识协同融合模块
    整合质量评估、门控融合、冲突检测和解决
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 子模块
        self.quality_assessor = KnowledgeQualityAssessor(embedding_dim, hidden_dim)
        self.gated_fusion = GatedFusionMechanism(embedding_dim, hidden_dim)
        self.conflict_detector = ConflictDetector(embedding_dim, hidden_dim)
        self.conflict_resolver = ConflictResolver(embedding_dim, hidden_dim)
        
        # 最终输出层
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(
        self,
        static_embedding: torch.Tensor,
        dynamic_embedding: torch.Tensor,
        static_metrics: Dict[str, torch.Tensor],
        dynamic_metrics: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        自适应知识融合
        
        Args:
            static_embedding: 静态知识嵌入 (v_MKG)
            dynamic_embedding: 动态证据嵌入 (v_Search)
            static_metrics: 静态知识指标 {coverage, consistency, staleness}
            dynamic_metrics: 动态证据指标 {effective_ratio, avg_credibility, conflict_ratio}
            
        Returns:
            融合结果字典
        """
        results = {}
        
        # 1. 质量评估
        static_quality = self.quality_assessor.assess_static_quality(
            static_embedding,
            static_metrics.get("coverage", torch.tensor([0.5])),
            static_metrics.get("consistency", torch.tensor([0.5])),
            static_metrics.get("staleness", torch.tensor([0.0]))
        )
        
        dynamic_quality = self.quality_assessor.assess_dynamic_quality(
            dynamic_embedding,
            dynamic_metrics.get("effective_ratio", torch.tensor([0.5])),
            dynamic_metrics.get("avg_credibility", torch.tensor([0.5])),
            dynamic_metrics.get("conflict_ratio", torch.tensor([0.0]))
        )
        
        results["static_quality"] = static_quality
        results["dynamic_quality"] = dynamic_quality
        
        # 2. 冲突检测
        conflict_info = self.conflict_detector(static_embedding, dynamic_embedding)
        results["conflict_info"] = conflict_info
        
        # 3. 根据是否有冲突选择融合策略
        if conflict_info["has_conflict"].any():
            # 存在冲突，使用冲突解决
            resolved = self.conflict_resolver(
                static_embedding,
                dynamic_embedding,
                static_quality,
                dynamic_quality,
                conflict_info
            )
            results["conflict_resolution"] = resolved
            fused_embedding = resolved["resolved_embedding"]
        else:
            # 无冲突，使用门控融合
            fusion_result = self.gated_fusion(
                static_embedding,
                dynamic_embedding,
                static_quality,
                dynamic_quality
            )
            results["fusion_result"] = fusion_result
            fused_embedding = fusion_result["fused_embedding"]
        
        # 4. 最终输出
        output = self.output_layer(fused_embedding)
        results["knowledge_embedding"] = output
        
        return results


class CrossSourceConsistencyVerifier(nn.Module):
    """
    跨源一致性验证模块
    验证静态知识和动态证据的一致性
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # 一致性验证网络
        self.consistency_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 支持/反驳分类
        self.stance_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # support, refute, neutral
        )
    
    def forward(
        self,
        static_embedding: torch.Tensor,
        dynamic_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        验证跨源一致性
        
        Args:
            static_embedding: 静态知识嵌入
            dynamic_embedding: 动态证据嵌入
            
        Returns:
            一致性验证结果
        """
        if static_embedding.dim() == 1:
            static_embedding = static_embedding.unsqueeze(0)
        if dynamic_embedding.dim() == 1:
            dynamic_embedding = dynamic_embedding.unsqueeze(0)
        
        combined = torch.cat([static_embedding, dynamic_embedding], dim=-1)
        
        # 一致性得分
        consistency = self.consistency_net(combined).squeeze(-1)
        
        # 立场分类
        stance_logits = self.stance_classifier(combined)
        stance_probs = F.softmax(stance_logits, dim=-1)
        stance = stance_logits.argmax(dim=-1)
        
        return {
            "cross_source_consistency": consistency,
            "stance": stance,  # 0: support, 1: refute, 2: neutral
            "stance_probs": stance_probs
        }
