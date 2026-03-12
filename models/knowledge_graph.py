"""
知识图谱模块 - 多源知识图谱表示与融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from typing import Dict, List, Tuple, Optional
import numpy as np


class KnowledgeGraphEmbedding(nn.Module):
    """
    知识图谱嵌入模块
    使用TransE/RotatE风格的嵌入学习
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        margin: float = 1.0
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # 实体和关系嵌入
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
    
    def forward(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        计算三元组得分 (TransE: h + r ≈ t)
        
        Args:
            head_ids: 头实体ID
            relation_ids: 关系ID
            tail_ids: 尾实体ID
            
        Returns:
            三元组得分（越小越好）
        """
        head_emb = self.entity_embedding(head_ids)
        relation_emb = self.relation_embedding(relation_ids)
        tail_emb = self.entity_embedding(tail_ids)
        
        # TransE得分函数
        score = torch.norm(head_emb + relation_emb - tail_emb, p=2, dim=-1)
        return score
    
    def get_entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """获取实体嵌入"""
        return self.entity_embedding(entity_ids)
    
    def get_relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """获取关系嵌入"""
        return self.relation_embedding(relation_ids)


class GraphAttentionNetwork(nn.Module):
    """
    图注意力网络
    用于聚合知识图谱中的邻域信息
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 第一层
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(hidden_channels * num_heads))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * num_heads, hidden_channels, 
                       heads=num_heads, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_channels * num_heads))
        
        # 最后一层
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_channels * num_heads, out_channels, 
                       heads=1, concat=False, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(out_channels))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边属性（可选）
            
        Returns:
            更新后的节点特征 [num_nodes, out_channels]
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.elu(x_new)
            x_new = self.dropout(x_new)
            
            # 残差连接（如果维度匹配）
            if x.shape[-1] == x_new.shape[-1]:
                x = x + x_new
            else:
                x = x_new
        
        return x


class KnowledgeGraphModule(nn.Module):
    """
    单个知识图谱处理模块
    包含实体链接、邻域聚合和知识表示
    """
    
    def __init__(
        self,
        kg_type: str,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_gat_heads: int = 8,
        num_gat_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.kg_type = kg_type
        self.embedding_dim = embedding_dim
        
        # 知识图谱嵌入
        self.kg_embedding = KnowledgeGraphEmbedding(
            num_entities, num_relations, embedding_dim
        )
        
        # 图注意力网络
        self.gat = GraphAttentionNetwork(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            out_channels=embedding_dim,
            num_heads=num_gat_heads,
            num_layers=num_gat_layers,
            dropout=dropout
        )
        
        # 实体链接得分计算
        self.link_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def compute_link_score(
        self,
        mention_embedding: torch.Tensor,
        candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        计算实体链接得分
        
        Args:
            mention_embedding: 实体提及嵌入 [batch, hidden]
            candidate_ids: 候选实体ID [batch, num_candidates]
            
        Returns:
            链接得分 [batch, num_candidates]
        """
        # 获取候选实体嵌入
        candidate_emb = self.kg_embedding.get_entity_embedding(candidate_ids)
        
        # 扩展mention嵌入
        mention_expanded = mention_embedding.unsqueeze(1).expand_as(candidate_emb)
        
        # 计算得分
        combined = torch.cat([mention_expanded, candidate_emb], dim=-1)
        scores = self.link_scorer(combined).squeeze(-1)
        
        return scores
    
    def get_knowledge_representation(
        self,
        entity_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取知识增强的实体表示
        
        Args:
            entity_ids: 实体ID [num_entities]
            edge_index: 子图边索引 [2, num_edges]
            edge_type: 边类型（可选）
            
        Returns:
            知识增强表示 [num_entities, embedding_dim]
        """
        # 获取初始嵌入
        entity_emb = self.kg_embedding.get_entity_embedding(entity_ids)
        
        # GAT聚合邻域信息
        if edge_index.numel() > 0:
            enhanced_emb = self.gat(entity_emb, edge_index)
        else:
            enhanced_emb = entity_emb
        
        return enhanced_emb


class MultiSourceKGFusion(nn.Module):
    """
    多源知识图谱融合模块
    实现统一知识表示空间(UKRS)和层次化融合策略
    """
    
    def __init__(
        self,
        kg_configs: Dict[str, Dict],
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_kg_types: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_kg_types = num_kg_types
        
        # 各知识图谱模块
        self.kg_modules = nn.ModuleDict()
        for kg_type, config in kg_configs.items():
            self.kg_modules[kg_type] = KnowledgeGraphModule(
                kg_type=kg_type,
                num_entities=config["num_entities"],
                num_relations=config["num_relations"],
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        
        # 统一知识表示空间投影
        self.ukrs_projections = nn.ModuleDict()
        for kg_type in kg_configs.keys():
            self.ukrs_projections[kg_type] = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim)
            )
        
        # 知识图谱类型嵌入
        self.kg_type_embedding = nn.Embedding(num_kg_types, embedding_dim)
        
        # 领域内融合注意力
        self.intra_domain_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 跨领域融合注意力
        self.cross_domain_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 残差增强门控
        self.residual_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(embedding_dim * num_kg_types, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def project_to_ukrs(
        self,
        kg_type: str,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        将知识图谱嵌入投影到统一表示空间
        
        Args:
            kg_type: 知识图谱类型
            embeddings: 原始嵌入 [batch, embedding_dim]
            
        Returns:
            统一空间嵌入 [batch, embedding_dim]
        """
        return self.ukrs_projections[kg_type](embeddings)
    
    def intra_domain_fusion(
        self,
        kg_embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        领域内融合 - 同一知识图谱内的实体融合
        
        Args:
            kg_embeddings: 各知识图谱的实体嵌入字典
            
        Returns:
            融合后的嵌入字典
        """
        fused_embeddings = {}
        
        for kg_type, embeddings in kg_embeddings.items():
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(0)
            
            # 自注意力融合
            fused, _ = self.intra_domain_attention(
                embeddings, embeddings, embeddings
            )
            fused_embeddings[kg_type] = fused.squeeze(0)
        
        return fused_embeddings
    
    def cross_domain_fusion(
        self,
        kg_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        跨领域融合 - 不同知识图谱间的知识整合
        
        Args:
            kg_embeddings: 各知识图谱的实体嵌入字典
            
        Returns:
            跨领域融合表示 [batch, embedding_dim]
        """
        # 收集所有知识图谱的表示
        all_embeddings = []
        kg_type_ids = []
        
        kg_type_map = {"medical": 0, "commonsense": 1, "event": 2, "geo": 3}
        
        for kg_type, embeddings in kg_embeddings.items():
            if embeddings.numel() > 0:
                # 对每个KG取平均表示
                if embeddings.dim() == 2:
                    kg_repr = embeddings.mean(dim=0, keepdim=True)
                else:
                    kg_repr = embeddings.mean(dim=1)
                all_embeddings.append(kg_repr)
                kg_type_ids.append(kg_type_map.get(kg_type, 0))
        
        if not all_embeddings:
            return torch.zeros(1, self.embedding_dim)
        
        # 堆叠并添加类型嵌入
        stacked = torch.stack(all_embeddings, dim=1)  # [batch, num_kgs, dim]
        type_ids = torch.tensor(kg_type_ids, device=stacked.device)
        type_emb = self.kg_type_embedding(type_ids)  # [num_kgs, dim]
        stacked = stacked + type_emb.unsqueeze(0)
        
        # 跨领域注意力
        fused, _ = self.cross_domain_attention(stacked, stacked, stacked)
        
        return fused.mean(dim=1)  # [batch, dim]
    
    def forward(
        self,
        entity_embeddings: Dict[str, torch.Tensor],
        text_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        多源知识图谱融合前向传播
        
        Args:
            entity_embeddings: 各知识图谱的实体嵌入
            text_embedding: 文本嵌入 [batch, hidden]
            
        Returns:
            融合结果字典
        """
        # 1. 投影到统一表示空间
        ukrs_embeddings = {}
        for kg_type, embeddings in entity_embeddings.items():
            if kg_type in self.ukrs_projections and embeddings.numel() > 0:
                ukrs_embeddings[kg_type] = self.project_to_ukrs(kg_type, embeddings)
        
        # 2. 领域内融合
        intra_fused = self.intra_domain_fusion(ukrs_embeddings)
        
        # 3. 跨领域融合
        cross_fused = self.cross_domain_fusion(intra_fused)
        
        # 4. 残差增强
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
        
        gate_input = torch.cat([cross_fused, text_embedding], dim=-1)
        gate = self.residual_gate(gate_input)
        enhanced = gate * cross_fused + (1 - gate) * text_embedding
        
        # 5. 收集各KG的平均表示用于最终融合
        kg_reprs = []
        for kg_type in ["medical", "commonsense", "event", "geo"]:
            if kg_type in intra_fused and intra_fused[kg_type].numel() > 0:
                kg_reprs.append(intra_fused[kg_type].mean(dim=0))
            else:
                kg_reprs.append(torch.zeros(self.embedding_dim, device=enhanced.device))
        
        # 拼接并融合
        concat_repr = torch.cat(kg_reprs, dim=-1)
        if concat_repr.dim() == 1:
            concat_repr = concat_repr.unsqueeze(0)
        final_kg_repr = self.final_fusion(concat_repr)
        
        return {
            "mkg_embedding": enhanced,  # 多源知识增强表示
            "kg_representations": intra_fused,  # 各KG的表示
            "cross_domain_repr": cross_fused,  # 跨领域融合表示
            "final_kg_repr": final_kg_repr  # 最终KG表示
        }
