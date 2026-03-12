"""
配置文件 - 微博谣言检测系统
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ModelConfig:
    """模型配置"""
    # BERT配置
    bert_model_name: str = "hfl/chinese-roberta-wwm-ext"
    bert_hidden_size: int = 768
    max_seq_length: int = 256
    
    # 知识图谱嵌入配置
    kg_embedding_dim: int = 256
    num_kg_types: int = 4  # 医学、常识、事件、地理
    
    # 图注意力网络配置
    gat_hidden_dim: int = 256
    gat_num_heads: int = 8
    gat_num_layers: int = 2
    gat_dropout: float = 0.1
    
    # 融合配置
    fusion_hidden_dim: int = 512
    num_consistency_layers: int = 4  # 实体属性、关系结构、时序逻辑、常识符合度
    
    # 分类器配置
    classifier_hidden_dim: int = 256
    num_classes: int = 2
    dropout: float = 0.3

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # 对比学习配置
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 0.1
    
    # 损失权重
    kg_loss_weight: float = 0.3
    align_loss_weight: float = 0.2
    type_loss_weight: float = 0.1

@dataclass
class SearchConfig:
    """动态搜索配置"""
    search_threshold: float = 0.4  # 搜索触发阈值
    auth_threshold: float = 0.5   # 来源可信度阈值
    top_k_results: int = 10       # 每个查询返回结果数
    max_queries: int = 4          # 最大查询数
    
    # 时效性权重
    time_weight: float = 0.3
    event_weight: float = 0.2
    
    # 可信度评估权重
    authority_weight: float = 0.35
    expertise_weight: float = 0.25
    technical_weight: float = 0.20
    consistency_weight: float = 0.20

@dataclass
class KnowledgeGraphConfig:
    """知识图谱配置"""
    # 各知识图谱路径
    medical_kg_path: str = "data/kg/medical_kg.json"
    commonsense_kg_path: str = "data/kg/commonsense_kg.json"
    event_kg_path: str = "data/kg/event_kg.json"
    geo_kg_path: str = "data/kg/geo_kg.json"
    
    # 实体对齐配置
    align_threshold: float = 0.7
    name_view_weight: float = 0.4
    attr_view_weight: float = 0.3
    struct_view_weight: float = 0.3

@dataclass
class SystemConfig:
    """系统配置"""
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    max_content_length: int = 10 * 1024 * 1024  # 10MB

# 知识图谱类型枚举
KG_TYPES = {
    "medical": 0,
    "commonsense": 1,
    "event": 2,
    "geo": 3
}

# 实体类型到知识图谱的映射
ENTITY_TO_KG = {
    "disease": "medical",
    "symptom": "medical",
    "drug": "medical",
    "treatment": "medical",
    "person": "commonsense",
    "organization": "commonsense",
    "concept": "commonsense",
    "event": "event",
    "time": "event",
    "location": "geo",
    "place": "geo"
}

# 时间词汇表
TIME_KEYWORDS = [
    "今天", "昨天", "明天", "刚刚", "最新", "突发", "紧急",
    "今日", "昨日", "近日", "日前", "最近", "即将", "马上"
]

# 权威域名列表
AUTHORITY_DOMAINS = {
    "gov.cn": 1.0,
    "xinhuanet.com": 0.9,
    "people.com.cn": 0.9,
    "cctv.com": 0.9,
    "chinanews.com": 0.85,
    "thepaper.cn": 0.8,
    "weibo.com": 0.6
}
