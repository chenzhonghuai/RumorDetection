"""
MKGF-RD: 基于多源知识图谱融合的谣言检测方法
Multi-source Knowledge Graph Fusion for Rumor Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .text_encoder import TextEncoder, EntityMentionEncoder
from .knowledge_graph import MultiSourceKGFusion, KnowledgeGraphModule
from .entity_alignment import CrossGraphEntityAlignment
from .consistency_verification import MultiDimensionalConsistencyVerifier
from .classifier import RumorClassifier


class MKGF_RD(nn.Module):
    """
    MKGF-RD: 基于多源知识图谱融合的谣言检测模型
    
    主要组件:
    1. 文本编码器 - 基于BERT的语义编码
    2. 多源知识图谱融合 - 统一知识表示空间(UKRS)
    3. 跨图谱实体对齐 - 多视图学习
    4. 多维一致性验证 - 四层递进验证
    5. 谣言分类器 - 特征融合与分类
    """
    
    def __init__(
        self,
        bert_model_name: str = "hfl/chinese-roberta-wwm-ext",
        bert_hidden_size: int = 768,
        kg_embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_classes: int = 2,
        kg_configs: Dict = None,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.bert_hidden_size = bert_hidden_size
        self.kg_embedding_dim = kg_embedding_dim
        
        # 默认知识图谱配置
        if kg_configs is None:
            kg_configs = {
                "medical": {"num_entities": 100000, "num_relations": 50},
                "commonsense": {"num_entities": 200000, "num_relations": 100},
                "event": {"num_entities": 50000, "num_relations": 30},
                "geo": {"num_entities": 80000, "num_relations": 40}
            }
        
        # 1. 文本编码器
        self.text_encoder = TextEncoder(
            model_name=bert_model_name,
            hidden_size=bert_hidden_size
        )
        
        # 2. 实体提及编码器
        self.entity_encoder = EntityMentionEncoder(
            hidden_size=bert_hidden_size,
            entity_type_size=32
        )
        
        # 3. 多源知识图谱融合模块
        self.mkg_fusion = MultiSourceKGFusion(
            kg_configs=kg_configs,
            embedding_dim=kg_embedding_dim,
            hidden_dim=hidden_dim
        )
        
        # 4. 跨图谱实体对齐模块
        self.entity_alignment = CrossGraphEntityAlignment(
            embedding_dim=kg_embedding_dim
        )
        
        # 5. 多维一致性验证模块
        self.consistency_verifier = MultiDimensionalConsistencyVerifier(
            embedding_dim=kg_embedding_dim,
            hidden_dim=hidden_dim
        )
        
        # 6. 文本到知识空间的投影
        self.text_to_kg_projection = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, kg_embedding_dim)
        )
        
        # 7. 谣言分类器
        self.classifier = RumorClassifier(
            text_dim=bert_hidden_size,
            knowledge_dim=kg_embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_contrastive=True
        )
        
        # 8. 知识图谱覆盖率计算
        self.coverage_estimator = nn.Sequential(
            nn.Linear(kg_embedding_dim * 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def compute_kg_coverage(
        self,
        entity_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算多源知识图谱覆盖率
        
        Args:
            entity_embeddings: 各知识图谱的实体嵌入
            
        Returns:
            覆盖率得分
        """
        # 收集各KG的表示
        kg_reprs = []
        for kg_type in ["medical", "commonsense", "event", "geo"]:
            if kg_type in entity_embeddings and entity_embeddings[kg_type].numel() > 0:
                kg_reprs.append(entity_embeddings[kg_type].mean(dim=0))
            else:
                kg_reprs.append(torch.zeros(self.kg_embedding_dim, 
                                           device=next(self.parameters()).device))
        
        # 拼接并计算覆盖率
        concat = torch.cat(kg_reprs, dim=-1)
        if concat.dim() == 1:
            concat = concat.unsqueeze(0)
        coverage = self.coverage_estimator(concat).squeeze(-1)
        
        return coverage
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_ids: Optional[Dict[str, torch.Tensor]] = None,
        entity_types: Optional[torch.Tensor] = None,
        kg_edge_indices: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        MKGF-RD前向传播
        
        Args:
            input_ids: 输入token IDs [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            entity_ids: 各知识图谱的实体ID字典
            entity_types: 实体类型 [num_entities]
            kg_edge_indices: 各知识图谱的边索引
            labels: 标签（训练时使用）
            
        Returns:
            模型输出字典
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        results = {}
        
        # 1. 文本编码
        text_outputs = self.text_encoder(input_ids, attention_mask)
        text_embedding = text_outputs["text_embedding"]  # [batch, bert_hidden]
        results["text_embedding"] = text_embedding
        
        # 2. 投影到知识空间
        text_kg_embedding = self.text_to_kg_projection(text_embedding)
        
        # 3. 知识图谱处理
        if entity_ids is not None:
            # 获取各知识图谱的实体嵌入
            entity_embeddings = {}
            for kg_type, ids in entity_ids.items():
                if kg_type in self.mkg_fusion.kg_modules and ids.numel() > 0:
                    kg_module = self.mkg_fusion.kg_modules[kg_type]
                    edge_index = kg_edge_indices.get(kg_type, torch.empty(2, 0, dtype=torch.long, device=device))
                    entity_emb = kg_module.get_knowledge_representation(ids, edge_index)
                    entity_embeddings[kg_type] = entity_emb
            
            # 4. 多源知识图谱融合
            fusion_result = self.mkg_fusion(entity_embeddings, text_kg_embedding)
            mkg_embedding = fusion_result["mkg_embedding"]
            results["mkg_embedding"] = mkg_embedding
            results["kg_representations"] = fusion_result["kg_representations"]
            
            # 5. 计算覆盖率
            coverage = self.compute_kg_coverage(entity_embeddings)
            results["kg_coverage"] = coverage
            
            # 6. 多维一致性验证
            # 构建验证所需的信息
            claim_info = {
                "embedding": text_kg_embedding
            }
            kg_info = {
                "commonsense": {
                    "knowledge_emb": mkg_embedding
                }
            }
            
            consistency_result = self.consistency_verifier(claim_info, kg_info)
            consistency_scores = consistency_result["consistency_scores"]
            results["consistency_result"] = consistency_result
        else:
            # 无知识图谱信息时使用默认值
            mkg_embedding = text_kg_embedding
            consistency_scores = torch.ones(4, device=device) * 0.5
            results["kg_coverage"] = torch.tensor([0.0], device=device)
        
        # 7. 准备分类器输入
        if consistency_scores.dim() == 1:
            consistency_scores = consistency_scores.unsqueeze(0).expand(batch_size, -1)
        
        uncertainty = torch.ones(batch_size, 1, device=device) * 0.5
        
        # 8. 分类
        classification_result = self.classifier(
            text_features=text_embedding,
            knowledge_features=mkg_embedding,
            consistency_scores=consistency_scores,
            uncertainty=uncertainty,
            labels=labels
        )
        results.update(classification_result)
        
        # 9. 计算损失（训练时）
        if labels is not None:
            loss_dict = self.classifier.compute_loss(classification_result, labels)
            results.update(loss_dict)
        
        return results
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_ids: Optional[Dict[str, torch.Tensor]] = None,
        kg_edge_indices: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[Dict]:
        """
        预测接口
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            entity_ids: 实体ID字典
            kg_edge_indices: 边索引字典
            
        Returns:
            预测结果列表
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_ids=entity_ids,
                kg_edge_indices=kg_edge_indices
            )
        
        # 转换结果
        predictions = outputs["predictions"].cpu().numpy()
        probabilities = outputs["probabilities"].cpu().numpy()
        confidence = outputs["confidence"].cpu().numpy()
        
        results = []
        for i in range(len(predictions)):
            result = {
                "prediction": "rumor" if predictions[i] == 1 else "real",
                "probability": float(probabilities[i][predictions[i]]),
                "confidence": float(confidence[i]),
                "kg_coverage": float(outputs["kg_coverage"][i]) if outputs["kg_coverage"].numel() > i else 0.0
            }
            results.append(result)
        
        return results


class MKGF_RD_Loss(nn.Module):
    """
    MKGF-RD联合训练损失
    包含分类损失、知识图谱嵌入损失、对齐损失和类型约束损失
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        kg_weight: float = 0.3,
        align_weight: float = 0.2,
        type_weight: float = 0.1,
        contrastive_weight: float = 0.1
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.kg_weight = kg_weight
        self.align_weight = align_weight
        self.type_weight = type_weight
        self.contrastive_weight = contrastive_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        kg_triples: Optional[Dict] = None,
        alignment_pairs: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算联合损失
        
        Args:
            outputs: 模型输出
            labels: 真实标签
            kg_triples: 知识图谱三元组（用于KG嵌入损失）
            alignment_pairs: 对齐实体对（用于对齐损失）
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 1. 分类损失
        ce_loss = F.cross_entropy(outputs["logits"], labels)
        losses["ce_loss"] = ce_loss
        
        # 2. 对比学习损失
        if "contrastive_loss" in outputs:
            losses["contrastive_loss"] = outputs["contrastive_loss"]
        
        # 3. 知识图谱嵌入损失（如果提供三元组）
        if kg_triples is not None:
            kg_loss = self._compute_kg_loss(outputs, kg_triples)
            losses["kg_loss"] = kg_loss
        else:
            losses["kg_loss"] = torch.tensor(0.0, device=labels.device)
        
        # 4. 对齐损失（如果提供对齐对）
        if alignment_pairs is not None:
            align_loss = self._compute_alignment_loss(outputs, alignment_pairs)
            losses["align_loss"] = align_loss
        else:
            losses["align_loss"] = torch.tensor(0.0, device=labels.device)
        
        # 5. 总损失
        total_loss = (
            self.ce_weight * ce_loss +
            self.contrastive_weight * losses.get("contrastive_loss", torch.tensor(0.0)) +
            self.kg_weight * losses["kg_loss"] +
            self.align_weight * losses["align_loss"]
        )
        losses["total_loss"] = total_loss
        
        return losses
    
    def _compute_kg_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        kg_triples: Dict
    ) -> torch.Tensor:
        """计算知识图谱嵌入损失（TransE风格）"""
        # 简化实现
        return torch.tensor(0.0, device=outputs["logits"].device)
    
    def _compute_alignment_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        alignment_pairs: Dict
    ) -> torch.Tensor:
        """计算跨图谱对齐损失"""
        # 简化实现
        return torch.tensor(0.0, device=outputs["logits"].device)
