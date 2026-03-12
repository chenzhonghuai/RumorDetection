"""
文本编码器 - 基于BERT的语义编码
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Tuple, Optional


class TextEncoder(nn.Module):
    """
    基于Chinese-RoBERTa-wwm-ext的文本编码器
    用于获取微博文本的语义表示
    """
    
    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        hidden_size: int = 768,
        max_length: int = 256,
        pooling_strategy: str = "cls"  # cls, mean, max
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        
        # 加载预训练模型
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 投影层（可选）
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: token类型ids [batch_size, seq_len]
            return_all_hidden_states: 是否返回所有隐藏层状态
            
        Returns:
            包含文本表示的字典
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=return_all_hidden_states
        )
        
        # 获取序列输出
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # 池化策略
        if self.pooling_strategy == "cls":
            pooled_output = sequence_output[:, 0, :]  # [CLS] token
        elif self.pooling_strategy == "mean":
            # 平均池化（考虑attention mask）
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        elif self.pooling_strategy == "max":
            # 最大池化
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size())
            sequence_output[mask_expanded == 0] = -1e9
            pooled_output = torch.max(sequence_output, dim=1)[0]
        else:
            pooled_output = outputs.pooler_output
        
        # 投影和归一化
        projected = self.projection(pooled_output)
        projected = self.layer_norm(projected)
        projected = self.dropout(projected)
        
        result = {
            "text_embedding": projected,  # [batch, hidden]
            "sequence_output": sequence_output,  # [batch, seq_len, hidden]
            "pooled_output": pooled_output  # [batch, hidden]
        }
        
        if return_all_hidden_states:
            result["all_hidden_states"] = outputs.hidden_states
        
        return result
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        编码文本列表
        
        Args:
            texts: 文本列表
            
        Returns:
            文本嵌入 [batch_size, hidden_size]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移动到设备
        device = next(self.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
        
        return outputs["text_embedding"]
    
    def get_token_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_spans: List[List[Tuple[int, int]]]
    ) -> List[torch.Tensor]:
        """
        获取实体span对应的token嵌入
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            entity_spans: 每个样本的实体span列表 [(start, end), ...]
            
        Returns:
            每个样本的实体嵌入列表
        """
        outputs = self.forward(input_ids, attention_mask)
        sequence_output = outputs["sequence_output"]
        
        batch_entity_embeddings = []
        for i, spans in enumerate(entity_spans):
            entity_embeddings = []
            for start, end in spans:
                # 获取span内token的平均嵌入
                span_embedding = sequence_output[i, start:end, :].mean(dim=0)
                entity_embeddings.append(span_embedding)
            
            if entity_embeddings:
                batch_entity_embeddings.append(torch.stack(entity_embeddings))
            else:
                # 如果没有实体，返回零向量
                batch_entity_embeddings.append(
                    torch.zeros(1, self.hidden_size, device=sequence_output.device)
                )
        
        return batch_entity_embeddings


class EntityMentionEncoder(nn.Module):
    """
    实体提及编码器
    用于编码文本中识别出的实体提及
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        entity_type_size: int = 32,
        num_entity_types: int = 11  # 实体类型数量
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 实体类型嵌入
        self.entity_type_embedding = nn.Embedding(num_entity_types, entity_type_size)
        
        # 实体表示融合
        self.entity_fusion = nn.Sequential(
            nn.Linear(hidden_size + entity_type_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(
        self,
        entity_embeddings: torch.Tensor,
        entity_types: torch.Tensor
    ) -> torch.Tensor:
        """
        融合实体嵌入和类型信息
        
        Args:
            entity_embeddings: 实体token嵌入 [num_entities, hidden_size]
            entity_types: 实体类型 [num_entities]
            
        Returns:
            融合后的实体表示 [num_entities, hidden_size]
        """
        # 获取类型嵌入
        type_emb = self.entity_type_embedding(entity_types)
        
        # 拼接并融合
        combined = torch.cat([entity_embeddings, type_emb], dim=-1)
        fused = self.entity_fusion(combined)
        
        return fused
