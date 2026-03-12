"""
DSMK-RD: 融合动态搜索的多源知识增强谣言检测方法
Dynamic Search enhanced Multi-source Knowledge for Rumor Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .mkgf_rd import MKGF_RD
from .dynamic_search import DynamicSearchModule, SearchResult
from .adaptive_fusion import AdaptiveKnowledgeFusion, CrossSourceConsistencyVerifier
from .classifier import RumorClassifier


class DSMK_RD(nn.Module):
    """
    DSMK-RD: 融合动态搜索的多源知识增强谣言检测模型
    
    在MKGF-RD基础上增加:
    1. 智能搜索触发决策
    2. 多策略查询生成
    3. 来源可信度评估
    4. 自适应知识协同融合
    5. 跨源一致性验证
    """
    
    def __init__(
        self,
        bert_model_name: str = "hfl/chinese-roberta-wwm-ext",
        bert_hidden_size: int = 768,
        kg_embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_classes: int = 2,
        kg_configs: Dict = None,
        search_threshold: float = 0.4,
        credibility_threshold: float = 0.5,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.bert_hidden_size = bert_hidden_size
        self.kg_embedding_dim = kg_embedding_dim
        self.search_threshold = search_threshold
        
        # 1. 基础MKGF-RD模型
        self.mkgf_rd = MKGF_RD(
            bert_model_name=bert_model_name,
            bert_hidden_size=bert_hidden_size,
            kg_embedding_dim=kg_embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            kg_configs=kg_configs,
            dropout=dropout
        )
        
        # 2. 动态搜索模块
        self.dynamic_search = DynamicSearchModule(
            embedding_dim=kg_embedding_dim,
            hidden_dim=hidden_dim,
            search_threshold=search_threshold,
            credibility_threshold=credibility_threshold
        )
        
        # 3. 自适应知识融合模块
        self.adaptive_fusion = AdaptiveKnowledgeFusion(
            embedding_dim=kg_embedding_dim,
            hidden_dim=hidden_dim
        )
        
        # 4. 跨源一致性验证
        self.cross_source_verifier = CrossSourceConsistencyVerifier(
            embedding_dim=kg_embedding_dim
        )
        
        # 5. 增强的分类器（考虑动态搜索信息）
        self.enhanced_classifier = RumorClassifier(
            text_dim=bert_hidden_size,
            knowledge_dim=kg_embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_contrastive=True
        )
        
        # 6. 搜索结果编码器
        self.search_result_encoder = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, kg_embedding_dim)
        )
        
        # 7. 一致性得分扩展（增加跨源一致性）
        self.consistency_expansion = nn.Linear(4, 5)  # 4维扩展到5维
    
    def encode_search_results(
        self,
        search_results: List[SearchResult],
        text_encoder_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        编码搜索结果
        
        Args:
            search_results: 搜索结果列表
            text_encoder_outputs: 文本编码器输出（用于获取编码器）
            
        Returns:
            搜索结果嵌入 [num_results, kg_embedding_dim]
        """
        if not search_results:
            return torch.zeros(1, self.kg_embedding_dim, 
                             device=next(self.parameters()).device)
        
        # 简化实现：使用文本编码器编码搜索结果的snippet
        # 实际应用中应该批量编码
        device = next(self.parameters()).device
        result_embeddings = []
        
        for result in search_results:
            # 这里简化处理，实际应该用text_encoder编码
            # 使用随机初始化模拟
            emb = torch.randn(self.kg_embedding_dim, device=device)
            result_embeddings.append(emb)
        
        return torch.stack(result_embeddings)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text: Optional[str] = None,
        entity_ids: Optional[Dict[str, torch.Tensor]] = None,
        entity_types: Optional[torch.Tensor] = None,
        kg_edge_indices: Optional[Dict[str, torch.Tensor]] = None,
        entities: Optional[List[str]] = None,
        search_results: Optional[List[SearchResult]] = None,
        hot_topics: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        DSMK-RD前向传播
        
        Args:
            input_ids: 输入token IDs [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            text: 原始文本（用于搜索触发决策）
            entity_ids: 各知识图谱的实体ID字典
            entity_types: 实体类型
            kg_edge_indices: 各知识图谱的边索引
            entities: 识别的实体列表
            search_results: 搜索结果（如果已执行搜索）
            hot_topics: 热点话题列表
            labels: 标签（训练时使用）
            
        Returns:
            模型输出字典
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        results = {}
        
        # 1. 获取MKGF-RD的基础输出
        mkgf_outputs = self.mkgf_rd(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_ids=entity_ids,
            entity_types=entity_types,
            kg_edge_indices=kg_edge_indices
        )
        
        text_embedding = mkgf_outputs["text_embedding"]
        mkg_embedding = mkgf_outputs.get("mkg_embedding", 
                                         self.mkgf_rd.text_to_kg_projection(text_embedding))
        kg_coverage = mkgf_outputs.get("kg_coverage", torch.tensor([0.5], device=device))
        
        results["mkgf_outputs"] = mkgf_outputs
        results["text_embedding"] = text_embedding
        results["mkg_embedding"] = mkg_embedding
        results["kg_coverage"] = kg_coverage
        
        # 2. 动态搜索处理
        if text is not None:
            # 投影文本嵌入到知识空间
            text_kg_emb = self.mkgf_rd.text_to_kg_projection(text_embedding)
            
            # 编码搜索结果（如果有）
            if search_results is not None:
                result_embeddings = self.encode_search_results(
                    search_results, mkgf_outputs
                )
            else:
                result_embeddings = None
            
            # 动态搜索模块
            search_output = self.dynamic_search(
                text=text,
                text_embedding=text_kg_emb,
                kg_coverage=kg_coverage,
                entities=entities,
                search_results=search_results,
                result_embeddings=result_embeddings,
                hot_topics=hot_topics
            )
            
            results["search_output"] = search_output
            results["search_triggered"] = search_output["search_triggered"]
            
            if search_output["search_triggered"]:
                search_embedding = search_output["search_embedding"]
                results["search_embedding"] = search_embedding
                
                # 3. 自适应知识融合
                # 准备静态知识指标
                static_metrics = {
                    "coverage": kg_coverage,
                    "consistency": mkgf_outputs.get("consistency_result", {}).get(
                        "final_consistency", torch.tensor([0.5], device=device)
                    ),
                    "staleness": torch.tensor([0.1], device=device)  # 假设值
                }
                
                # 准备动态证据指标
                dynamic_metrics = {
                    "effective_ratio": torch.tensor(
                        [search_output.get("num_valid_results", 0) / 10], device=device
                    ),
                    "avg_credibility": search_output.get("credibility", {}).get(
                        "credibility", torch.tensor([0.5], device=device)
                    ).mean() if "credibility" in search_output else torch.tensor([0.5], device=device),
                    "conflict_ratio": torch.tensor([0.0], device=device)
                }
                
                # 自适应融合
                fusion_output = self.adaptive_fusion(
                    static_embedding=mkg_embedding,
                    dynamic_embedding=search_embedding,
                    static_metrics=static_metrics,
                    dynamic_metrics=dynamic_metrics
                )
                
                results["fusion_output"] = fusion_output
                knowledge_embedding = fusion_output["knowledge_embedding"]
                
                # 4. 跨源一致性验证
                cross_source_result = self.cross_source_verifier(
                    mkg_embedding, search_embedding
                )
                results["cross_source_consistency"] = cross_source_result
            else:
                knowledge_embedding = mkg_embedding
                results["search_embedding"] = torch.zeros_like(mkg_embedding)
        else:
            knowledge_embedding = mkg_embedding
            results["search_triggered"] = False
        
        results["knowledge_embedding"] = knowledge_embedding
        
        # 5. 准备一致性得分（扩展到5维，包含跨源一致性）
        base_consistency = mkgf_outputs.get("consistency_result", {}).get(
            "consistency_scores", torch.ones(4, device=device) * 0.5
        )
        if base_consistency.dim() == 1:
            base_consistency = base_consistency.unsqueeze(0)
        
        # 扩展一致性得分
        expanded_consistency = self.consistency_expansion(base_consistency)
        
        # 如果有跨源一致性，更新第5维
        if "cross_source_consistency" in results:
            cross_consistency = results["cross_source_consistency"]["cross_source_consistency"]
            if cross_consistency.dim() == 0:
                cross_consistency = cross_consistency.unsqueeze(0)
            expanded_consistency[:, 4] = cross_consistency
        
        # 取前4维用于分类器（保持兼容性）
        consistency_scores = expanded_consistency[:, :4]
        
        # 6. 计算不确定性
        if "fusion_output" in results and "conflict_info" in results["fusion_output"]:
            uncertainty = results["fusion_output"]["conflict_info"]["conflict_probability"]
        else:
            uncertainty = torch.ones(batch_size, device=device) * 0.3
        
        if uncertainty.dim() == 0:
            uncertainty = uncertainty.unsqueeze(0)
        uncertainty = uncertainty.unsqueeze(-1)
        
        # 7. 增强分类
        classification_result = self.enhanced_classifier(
            text_features=text_embedding,
            knowledge_features=knowledge_embedding,
            consistency_scores=consistency_scores,
            uncertainty=uncertainty,
            labels=labels
        )
        results.update(classification_result)
        
        # 8. 计算损失（训练时）
        if labels is not None:
            loss_dict = self.enhanced_classifier.compute_loss(classification_result, labels)
            results.update(loss_dict)
        
        return results
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text: str,
        entity_ids: Optional[Dict[str, torch.Tensor]] = None,
        kg_edge_indices: Optional[Dict[str, torch.Tensor]] = None,
        entities: Optional[List[str]] = None,
        search_results: Optional[List[SearchResult]] = None,
        hot_topics: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        预测接口
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            text: 原始文本
            entity_ids: 实体ID字典
            kg_edge_indices: 边索引字典
            entities: 实体列表
            search_results: 搜索结果
            hot_topics: 热点话题
            
        Returns:
            预测结果列表
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                text=text,
                entity_ids=entity_ids,
                kg_edge_indices=kg_edge_indices,
                entities=entities,
                search_results=search_results,
                hot_topics=hot_topics
            )
        
        # 转换结果
        predictions = outputs["predictions"].cpu().numpy()
        probabilities = outputs["probabilities"].cpu().numpy()
        confidence = outputs["confidence"].cpu().numpy()
        contradiction_types = outputs["contradiction_type"].cpu().numpy()
        
        # 矛盾类型映射
        contradiction_type_names = [
            "无矛盾",
            "实体属性矛盾",
            "关系结构矛盾",
            "时序逻辑矛盾",
            "常识违背"
        ]
        
        results = []
        for i in range(len(predictions)):
            result = {
                "prediction": "rumor" if predictions[i] == 1 else "real",
                "probability": float(probabilities[i][predictions[i]]),
                "confidence": float(confidence[i]),
                "contradiction_type": contradiction_type_names[contradiction_types[i]],
                "search_triggered": outputs["search_triggered"],
                "kg_coverage": float(outputs["kg_coverage"][i]) if outputs["kg_coverage"].numel() > i else 0.0
            }
            
            # 添加跨源一致性（如果有）
            if "cross_source_consistency" in outputs:
                result["cross_source_consistency"] = float(
                    outputs["cross_source_consistency"]["cross_source_consistency"]
                )
            
            results.append(result)
        
        return results
    
    def get_search_queries(self, text: str, entities: List[str] = None) -> Dict[str, str]:
        """
        获取搜索查询（用于外部搜索）
        
        Args:
            text: 原始文本
            entities: 实体列表
            
        Returns:
            查询字典
        """
        # 使用查询生成器
        dummy_embedding = torch.zeros(1, self.kg_embedding_dim, 
                                     device=next(self.parameters()).device)
        query_result = self.dynamic_search.query_generator(
            text, dummy_embedding, entities
        )
        return query_result["queries"]


class DSMK_RD_Loss(nn.Module):
    """
    DSMK-RD联合训练损失
    在MKGF-RD损失基础上增加搜索相关损失
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        kg_weight: float = 0.3,
        align_weight: float = 0.2,
        contrastive_weight: float = 0.1,
        search_weight: float = 0.1,
        fusion_weight: float = 0.1
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.kg_weight = kg_weight
        self.align_weight = align_weight
        self.contrastive_weight = contrastive_weight
        self.search_weight = search_weight
        self.fusion_weight = fusion_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算联合损失
        
        Args:
            outputs: 模型输出
            labels: 真实标签
            
        Returns:
            损失字典
        """
        losses = {}
        device = labels.device
        
        # 1. 分类损失
        ce_loss = F.cross_entropy(outputs["logits"], labels)
        losses["ce_loss"] = ce_loss
        
        # 2. 对比学习损失
        if "contrastive_loss" in outputs:
            losses["contrastive_loss"] = outputs["contrastive_loss"]
        else:
            losses["contrastive_loss"] = torch.tensor(0.0, device=device)
        
        # 3. 搜索触发损失（鼓励在需要时触发搜索）
        # 简化实现
        losses["search_loss"] = torch.tensor(0.0, device=device)
        
        # 4. 融合一致性损失
        if "fusion_output" in outputs and "conflict_info" in outputs["fusion_output"]:
            conflict_prob = outputs["fusion_output"]["conflict_info"]["conflict_probability"]
            # 鼓励减少冲突
            fusion_loss = conflict_prob.mean()
            losses["fusion_loss"] = fusion_loss
        else:
            losses["fusion_loss"] = torch.tensor(0.0, device=device)
        
        # 5. 总损失
        total_loss = (
            self.ce_weight * ce_loss +
            self.contrastive_weight * losses["contrastive_loss"] +
            self.search_weight * losses["search_loss"] +
            self.fusion_weight * losses["fusion_loss"]
        )
        losses["total_loss"] = total_loss
        
        return losses
