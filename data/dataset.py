"""
数据集模块
微博谣言检测数据集定义
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import json
import pandas as pd
from transformers import BertTokenizer

from .preprocessor import TextPreprocessor, EntityRecognizer


class WeiboRumorDataset(Dataset):
    """
    微博谣言检测数据集
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        max_length: int = 256,
        preprocess: bool = True,
        extract_entities: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径（支持json, csv）
            tokenizer: BERT分词器
            max_length: 最大序列长度
            preprocess: 是否进行文本预处理
            extract_entities: 是否提取实体
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.extract_entities = extract_entities
        
        # 预处理器
        self.text_preprocessor = TextPreprocessor()
        self.entity_recognizer = EntityRecognizer()
        
        # 加载数据
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据文件"""
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            样本字典
        """
        item = self.data[idx]
        
        # 获取文本和标签
        text = item.get('content', item.get('text', ''))
        label = item.get('label', 0)
        
        # 文本预处理
        if self.preprocess:
            preprocess_result = self.text_preprocessor.preprocess(text)
            cleaned_text = preprocess_result['cleaned']
        else:
            cleaned_text = text
        
        # BERT编码
        encoding = self.tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'text': cleaned_text
        }
        
        # 实体提取
        if self.extract_entities:
            entities = self.entity_recognizer.recognize(cleaned_text)
            grouped_entities = self.entity_recognizer.group_by_kg(entities)
            
            result['entities'] = [e.text for e in entities]
            result['entity_types'] = [e.type for e in entities]
            result['entity_kg_types'] = grouped_entities
        
        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批次整理函数
    
    Args:
        batch: 样本列表
        
    Returns:
        整理后的批次字典
    """
    # 基本字段
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    texts = [item['text'] for item in batch]
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'texts': texts
    }
    
    # 实体信息（如果有）
    if 'entities' in batch[0]:
        result['entities'] = [item['entities'] for item in batch]
        result['entity_types'] = [item['entity_types'] for item in batch]
        result['entity_kg_types'] = [item['entity_kg_types'] for item in batch]
    
    return result


class StreamingDataset(Dataset):
    """
    流式数据集
    用于在线检测场景
    """
    
    def __init__(
        self,
        tokenizer: BertTokenizer,
        max_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_preprocessor = TextPreprocessor()
        self.entity_recognizer = EntityRecognizer()
        self.data = []
    
    def add_sample(self, text: str, label: Optional[int] = None):
        """添加样本"""
        self.data.append({'text': text, 'label': label})
    
    def clear(self):
        """清空数据"""
        self.data = []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item['text']
        
        # 预处理
        preprocess_result = self.text_preprocessor.preprocess(text)
        cleaned_text = preprocess_result['cleaned']
        
        # 编码
        encoding = self.tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': cleaned_text
        }
        
        if item['label'] is not None:
            result['label'] = torch.tensor(item['label'], dtype=torch.long)
        
        # 实体
        entities = self.entity_recognizer.recognize(cleaned_text)
        result['entities'] = [e.text for e in entities]
        
        return result


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer: BertTokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        train_path: 训练集路径
        val_path: 验证集路径
        test_path: 测试集路径
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大长度
        num_workers: 工作进程数
        
    Returns:
        训练、验证、测试数据加载器
    """
    train_dataset = WeiboRumorDataset(train_path, tokenizer, max_length)
    val_dataset = WeiboRumorDataset(val_path, tokenizer, max_length)
    test_dataset = WeiboRumorDataset(test_path, tokenizer, max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
