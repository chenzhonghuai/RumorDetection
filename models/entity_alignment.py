"""
跨图谱实体对齐模块
基于多视图学习的实体对齐方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class NameViewEncoder(nn.Module):
    """
    名称视图编码器
    基于字符级和词级特征的实体名称编码
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 256
    ):
        super().__init__()
        
        # 字符级嵌入
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 双向LSTM编码
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # 输出投影
        self.projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, char_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        编码实体名称
        
        Args:
            char_ids: 字符ID序列 [batch, max_len]
            lengths: 实际长度 [batch]
            
        Returns:
            名称嵌入 [batch, output_dim]
        """
        # 字符嵌入
        char_emb = self.char_embedding(char_ids)
        
        # LSTM编码
        packed = nn.utils.rnn.pack_padded_sequence(
            char_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        
        # 拼接双向隐藏状态
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        # 投影
        output = self.projection(hidden)
        return output


class AttributeViewEncoder(nn.Module):
    """
    属性视图编码器
    编码实体的属性信息
    """
    
    def __init__(
        self,
        num_attr_types: int = 100,
        attr_embedding_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 256
    ):
        super().__init__()
        
        # 属性类型嵌入
        self.attr_type_embedding = nn.Embedding(num_attr_types, attr_embedding_dim)
        
        # 属性值编码器（简化版，实际可用更复杂的编码）
        self.value_encoder = nn.Sequential(
            nn.Linear(attr_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 属性聚合注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 输出投影
        self.projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        attr_types: torch.Tensor,
        attr_values: torch.Tensor,
        attr_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        编码实体属性
        
        Args:
            attr_types: 属性类型ID [batch, num_attrs]
            attr_values: 属性值嵌入 [batch, num_attrs, value_dim]
            attr_mask: 属性掩码 [batch, num_attrs]
            
        Returns:
            属性视图嵌入 [batch, output_dim]
        """
        # 属性类型嵌入
        type_emb = self.attr_type_embedding(attr_types)
        
        # 融合类型和值
        combined = type_emb + attr_values[:, :, :type_emb.size(-1)]
        encoded = self.value_encoder(combined)
        
        # 注意力聚合
        # 创建注意力掩码
        attn_mask = ~attr_mask.bool()
        aggregated, _ = self.attention(
            encoded, encoded, encoded,
            key_padding_mask=attn_mask
        )
        
        # 平均池化
        mask_expanded = attr_mask.unsqueeze(-1).float()
        pooled = (aggregated * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        # 投影
        output = self.projection(pooled)
        return output


class StructureViewEncoder(nn.Module):
    """
    结构视图编码器
    编码实体在图中的结构信息
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        
        # GCN层
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.gcn_layers.append(nn.Linear(in_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # 输出投影
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        编码图结构
        
        Args:
            node_features: 节点特征 [num_nodes, input_dim]
            adj_matrix: 邻接矩阵 [num_nodes, num_nodes]
            
        Returns:
            结构视图嵌入 [num_nodes, output_dim]
        """
        x = node_features
        
        # 归一化邻接矩阵
        degree = adj_matrix.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj_matrix / degree
        
        for gcn, norm in zip(self.gcn_layers, self.norms):
            # 消息传递
            x = torch.matmul(adj_norm, x)
            x = gcn(x)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        output = self.projection(x)
        return output


class CrossGraphEntityAlignment(nn.Module):
    """
    跨图谱实体对齐模块
    综合名称、属性和结构三个视图进行实体匹配
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        name_weight: float = 0.4,
        attr_weight: float = 0.3,
        struct_weight: float = 0.3,
        align_threshold: float = 0.7
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.name_weight = name_weight
        self.attr_weight = attr_weight
        self.struct_weight = struct_weight
        self.align_threshold = align_threshold
        
        # 三视图编码器
        self.name_encoder = NameViewEncoder(output_dim=embedding_dim)
        self.attr_encoder = AttributeViewEncoder(output_dim=embedding_dim)
        self.struct_encoder = StructureViewEncoder(
            input_dim=embedding_dim,
            output_dim=embedding_dim
        )
        
        # 视图融合
        self.view_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 对齐得分计算
        self.alignment_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def compute_view_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算两个嵌入的余弦相似度
        
        Args:
            emb1: 嵌入1 [batch1, dim]
            emb2: 嵌入2 [batch2, dim]
            
        Returns:
            相似度矩阵 [batch1, batch2]
        """
        emb1_norm = F.normalize(emb1, p=2, dim=-1)
        emb2_norm = F.normalize(emb2, p=2, dim=-1)
        similarity = torch.matmul(emb1_norm, emb2_norm.transpose(-2, -1))
        return similarity
    
    def forward(
        self,
        entities_kg1: Dict[str, torch.Tensor],
        entities_kg2: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算两个知识图谱间的实体对齐
        
        Args:
            entities_kg1: KG1的实体信息字典
            entities_kg2: KG2的实体信息字典
            
        Returns:
            对齐结果字典
        """
        # 编码名称视图
        name_emb1 = self.name_encoder(
            entities_kg1["name_chars"],
            entities_kg1["name_lengths"]
        )
        name_emb2 = self.name_encoder(
            entities_kg2["name_chars"],
            entities_kg2["name_lengths"]
        )
        name_sim = self.compute_view_similarity(name_emb1, name_emb2)
        
        # 编码属性视图
        attr_emb1 = self.attr_encoder(
            entities_kg1["attr_types"],
            entities_kg1["attr_values"],
            entities_kg1["attr_mask"]
        )
        attr_emb2 = self.attr_encoder(
            entities_kg2["attr_types"],
            entities_kg2["attr_values"],
            entities_kg2["attr_mask"]
        )
        attr_sim = self.compute_view_similarity(attr_emb1, attr_emb2)
        
        # 编码结构视图
        struct_emb1 = self.struct_encoder(
            entities_kg1["node_features"],
            entities_kg1["adj_matrix"]
        )
        struct_emb2 = self.struct_encoder(
            entities_kg2["node_features"],
            entities_kg2["adj_matrix"]
        )
        struct_sim = self.compute_view_similarity(struct_emb1, struct_emb2)
        
        # 加权融合三视图相似度
        combined_sim = (
            self.name_weight * name_sim +
            self.attr_weight * attr_sim +
            self.struct_weight * struct_sim
        )
        
        # 获取对齐对
        aligned_pairs = (combined_sim > self.align_threshold).nonzero(as_tuple=False)
        alignment_scores = combined_sim[aligned_pairs[:, 0], aligned_pairs[:, 1]]
        
        # 融合视图表示
        fused_emb1 = self.view_fusion(
            torch.cat([name_emb1, attr_emb1, struct_emb1], dim=-1)
        )
        fused_emb2 = self.view_fusion(
            torch.cat([name_emb2, attr_emb2, struct_emb2], dim=-1)
        )
        
        return {
            "similarity_matrix": combined_sim,
            "aligned_pairs": aligned_pairs,
            "alignment_scores": alignment_scores,
            "fused_embeddings_kg1": fused_emb1,
            "fused_embeddings_kg2": fused_emb2,
            "name_similarity": name_sim,
            "attr_similarity": attr_sim,
            "struct_similarity": struct_sim
        }
    
    def align_entities(
        self,
        similarity_matrix: torch.Tensor,
        strategy: str = "greedy"
    ) -> List[Tuple[int, int, float]]:
        """
        基于相似度矩阵进行实体对齐
        
        Args:
            similarity_matrix: 相似度矩阵 [n1, n2]
            strategy: 对齐策略 (greedy, hungarian)
            
        Returns:
            对齐结果列表 [(idx1, idx2, score), ...]
        """
        alignments = []
        sim = similarity_matrix.clone()
        
        if strategy == "greedy":
            # 贪心对齐
            while True:
                max_val = sim.max()
                if max_val < self.align_threshold:
                    break
                
                max_idx = (sim == max_val).nonzero(as_tuple=False)[0]
                i, j = max_idx[0].item(), max_idx[1].item()
                alignments.append((i, j, max_val.item()))
                
                # 移除已对齐的实体
                sim[i, :] = -1
                sim[:, j] = -1
        
        return alignments
