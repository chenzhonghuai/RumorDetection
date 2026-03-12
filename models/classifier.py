"""
谣言分类器模块
整合多源特征进行最终分类决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class FeatureFusion(nn.Module):
    """
    四阶段特征融合模块
    融合文本语义、知识增强、一致性得分和不确定性信息
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        knowledge_dim: int = 256,
        consistency_dim: int = 4,
        uncertainty_dim: int = 1,
        hidden_dim: int = 512,
        output_dim: int = 256
    ):
        super().__init__()
        
        # 阶段1: 文本特征投影
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 阶段2: 知识特征投影
        self.knowledge_projection = nn.Sequential(
            nn.Linear(knowledge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 阶段3: 一致性特征投影
        self.consistency_projection = nn.Sequential(
            nn.Linear(consistency_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 阶段4: 不确定性特征投影
        self.uncertainty_projection = nn.Sequential(
            nn.Linear(uncertainty_dim, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, hidden_dim)
        )
        
        # 特征融合注意力
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        text_features: torch.Tensor,
        knowledge_features: torch.Tensor,
        consistency_scores: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        四阶段特征融合
        
        Args:
            text_features: 文本语义特征 [batch, text_dim]
            knowledge_features: 知识增强特征 [batch, knowledge_dim]
            consistency_scores: 一致性得分 [batch, 4]
            uncertainty: 不确定性 [batch, 1]
            
        Returns:
            融合特征 [batch, output_dim]
        """
        # 各阶段投影
        text_proj = self.text_projection(text_features)
        knowledge_proj = self.knowledge_projection(knowledge_features)
        consistency_proj = self.consistency_projection(consistency_scores)
        uncertainty_proj = self.uncertainty_projection(uncertainty)
        
        # 堆叠为序列
        features = torch.stack([
            text_proj, knowledge_proj, consistency_proj, uncertainty_proj
        ], dim=1)  # [batch, 4, hidden]
        
        # 注意力融合
        attended, _ = self.fusion_attention(features, features, features)
        
        # 拼接并融合
        concat = attended.view(attended.size(0), -1)  # [batch, 4*hidden]
        output = self.final_fusion(concat)
        
        return output


class AttentionClassifier(nn.Module):
    """
    注意力增强分类器
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 置信度估计
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        分类前向传播
        
        Args:
            features: 输入特征 [batch, dim]
            
        Returns:
            分类结果字典
        """
        # 添加序列维度用于注意力
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        # 自注意力
        attended, attn_weights = self.self_attention(features, features, features)
        attended = attended.squeeze(1)
        
        # 分类
        logits = self.classifier(attended)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        
        # 置信度
        confidence = self.confidence_head(attended).squeeze(-1)
        
        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": predictions,
            "confidence": confidence,
            "attention_weights": attn_weights
        }


class ContrastiveLearningHead(nn.Module):
    """
    监督对比学习头
    增强分类器的判别能力
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        super().__init__()
        self.temperature = temperature
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算对比学习损失
        
        Args:
            features: 特征 [batch, dim]
            labels: 标签 [batch]
            
        Returns:
            对比学习结果
        """
        # 投影
        projections = self.projector(features)
        projections = F.normalize(projections, p=2, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(projections, projections.T) / self.temperature
        
        # 创建标签掩码
        batch_size = labels.size(0)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # 移除对角线
        mask = torch.eye(batch_size, device=labels.device).bool()
        labels_equal = labels_equal & ~mask
        
        # 计算对比损失
        exp_sim = torch.exp(similarity)
        
        # 正样本对
        pos_sim = exp_sim * labels_equal.float()
        
        # 所有样本对（除自身）
        all_sim = exp_sim * (~mask).float()
        
        # 对比损失
        loss = -torch.log(
            pos_sim.sum(dim=1) / (all_sim.sum(dim=1) + 1e-8) + 1e-8
        )
        loss = loss[labels_equal.sum(dim=1) > 0].mean()
        
        return {
            "contrastive_loss": loss,
            "projections": projections,
            "similarity_matrix": similarity
        }


class RumorClassifier(nn.Module):
    """
    谣言分类器
    整合特征融合、注意力分类和对比学习
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        knowledge_dim: int = 256,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_contrastive: bool = True,
        contrastive_weight: float = 0.1
    ):
        super().__init__()
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        
        # 特征融合
        self.feature_fusion = FeatureFusion(
            text_dim=text_dim,
            knowledge_dim=knowledge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim // 2
        )
        
        # 注意力分类器
        self.classifier = AttentionClassifier(
            input_dim=hidden_dim // 2,
            hidden_dim=hidden_dim // 2,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # 对比学习头
        if use_contrastive:
            self.contrastive_head = ContrastiveLearningHead(
                input_dim=hidden_dim // 2
            )
        
        # 判定依据生成
        self.explanation_generator = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 4, hidden_dim // 4),  # +4 for consistency scores
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 5)  # 5种矛盾类型
        )
        
        # 矛盾类型
        self.contradiction_types = [
            "无矛盾",
            "实体属性矛盾",
            "关系结构矛盾",
            "时序逻辑矛盾",
            "常识违背"
        ]
    
    def forward(
        self,
        text_features: torch.Tensor,
        knowledge_features: torch.Tensor,
        consistency_scores: torch.Tensor,
        uncertainty: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        分类前向传播
        
        Args:
            text_features: 文本特征 [batch, text_dim]
            knowledge_features: 知识特征 [batch, knowledge_dim]
            consistency_scores: 一致性得分 [batch, 4]
            uncertainty: 不确定性 [batch, 1]
            labels: 标签（训练时使用）[batch]
            
        Returns:
            分类结果字典
        """
        results = {}
        
        # 1. 特征融合
        fused_features = self.feature_fusion(
            text_features,
            knowledge_features,
            consistency_scores,
            uncertainty
        )
        results["fused_features"] = fused_features
        
        # 2. 分类
        classification_result = self.classifier(fused_features)
        results.update(classification_result)
        
        # 3. 对比学习（训练时）
        if self.use_contrastive and labels is not None:
            contrastive_result = self.contrastive_head(fused_features, labels)
            results["contrastive_loss"] = contrastive_result["contrastive_loss"]
        
        # 4. 生成判定依据
        explanation_input = torch.cat([fused_features, consistency_scores], dim=-1)
        contradiction_logits = self.explanation_generator(explanation_input)
        contradiction_probs = F.softmax(contradiction_logits, dim=-1)
        contradiction_type = contradiction_logits.argmax(dim=-1)
        
        results["contradiction_logits"] = contradiction_logits
        results["contradiction_probs"] = contradiction_probs
        results["contradiction_type"] = contradiction_type
        
        return results
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            outputs: 模型输出
            labels: 真实标签
            
        Returns:
            损失字典
        """
        # 分类损失
        ce_loss = F.cross_entropy(outputs["logits"], labels)
        
        # 总损失
        total_loss = ce_loss
        
        # 对比学习损失
        if "contrastive_loss" in outputs:
            contrastive_loss = outputs["contrastive_loss"]
            if not torch.isnan(contrastive_loss):
                total_loss = total_loss + self.contrastive_weight * contrastive_loss
        
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "contrastive_loss": outputs.get("contrastive_loss", torch.tensor(0.0))
        }
    
    def predict(
        self,
        text_features: torch.Tensor,
        knowledge_features: torch.Tensor,
        consistency_scores: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> Dict[str, any]:
        """
        预测接口
        
        Args:
            text_features: 文本特征
            knowledge_features: 知识特征
            consistency_scores: 一致性得分
            uncertainty: 不确定性
            
        Returns:
            预测结果
        """
        with torch.no_grad():
            outputs = self.forward(
                text_features,
                knowledge_features,
                consistency_scores,
                uncertainty
            )
        
        # 转换为可读结果
        predictions = outputs["predictions"].cpu().numpy()
        probabilities = outputs["probabilities"].cpu().numpy()
        confidence = outputs["confidence"].cpu().numpy()
        contradiction_types = outputs["contradiction_type"].cpu().numpy()
        
        results = []
        for i in range(len(predictions)):
            result = {
                "prediction": "rumor" if predictions[i] == 1 else "real",
                "probability": float(probabilities[i][predictions[i]]),
                "confidence": float(confidence[i]),
                "contradiction_type": self.contradiction_types[contradiction_types[i]],
                "consistency_scores": {
                    "entity_attr": float(consistency_scores[i][0]),
                    "relation_struct": float(consistency_scores[i][1]),
                    "temporal_logic": float(consistency_scores[i][2]),
                    "commonsense": float(consistency_scores[i][3])
                }
            }
            results.append(result)
        
        return results
