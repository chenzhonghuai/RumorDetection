"""
谣言检测器
封装模型推理逻辑
"""

import torch
from typing import Dict, List, Optional
from transformers import BertTokenizer
import time

from models import DSMK_RD
from data.preprocessor import TextPreprocessor, EntityRecognizer


class RumorDetector:
    """
    谣言检测器
    提供单条和批量检测接口
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        bert_model_name: str = "hfl/chinese-roberta-wwm-ext"
    ):
        """
        初始化检测器
        
        Args:
            model_path: 模型权重路径
            device: 运行设备
            bert_model_name: BERT模型名称
        """
        self.device = device
        self.bert_model_name = bert_model_name
        
        # 初始化分词器
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # 初始化预处理器
        self.preprocessor = TextPreprocessor()
        self.entity_recognizer = EntityRecognizer()
        
        # 初始化模型
        self.model = DSMK_RD(
            bert_model_name=bert_model_name,
            bert_hidden_size=768,
            kg_embedding_dim=256
        )
        
        # 加载权重（如果提供）
        if model_path:
            self.load_model(model_path)
        
        self.model.to(device)
        self.model.eval()
    
    def load_model(self, model_path: str):
        """加载模型权重"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def preprocess(self, text: str) -> Dict:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理结果
        """
        # 文本清洗
        cleaned = self.preprocessor.preprocess(text)
        
        # 实体识别
        entities = self.entity_recognizer.recognize(cleaned['cleaned'])
        entity_texts = [e.text for e in entities]
        grouped_entities = self.entity_recognizer.group_by_kg(entities)
        
        # BERT编码
        encoding = self.tokenizer(
            cleaned['cleaned'],
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        return {
            'original': text,
            'cleaned': cleaned['cleaned'],
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'entities': entity_texts,
            'grouped_entities': grouped_entities
        }
    
    def detect_single(self, text: str) -> Dict:
        """
        检测单条文本
        
        Args:
            text: 待检测文本
            
        Returns:
            检测结果
        """
        start_time = time.time()
        
        # 预处理
        preprocessed = self.preprocess(text)
        
        # 移动到设备
        input_ids = preprocessed['input_ids'].to(self.device)
        attention_mask = preprocessed['attention_mask'].to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                text=preprocessed['cleaned'],
                entities=preprocessed['entities']
            )
        
        # 解析结果
        prediction = outputs['predictions'][0].item()
        probability = outputs['probabilities'][0].cpu().numpy()
        confidence = outputs['confidence'][0].item()
        contradiction_type = outputs['contradiction_type'][0].item()
        
        # 矛盾类型映射
        contradiction_types = [
            "无矛盾",
            "实体属性矛盾",
            "关系结构矛盾",
            "时序逻辑矛盾",
            "常识违背"
        ]
        
        processing_time = (time.time() - start_time) * 1000  # 毫秒
        
        result = {
            'result': 'rumor' if prediction == 1 else 'real',
            'confidence': float(confidence),
            'probability': {
                'real': float(probability[0]),
                'rumor': float(probability[1])
            },
            'contradiction_type': contradiction_types[contradiction_type],
            'search_triggered': outputs.get('search_triggered', False),
            'kg_coverage': float(outputs['kg_coverage'][0]) if outputs['kg_coverage'].numel() > 0 else 0.0,
            'entities_found': preprocessed['entities'],
            'processing_time_ms': round(processing_time, 2)
        }
        
        # 添加跨源一致性（如果有）
        if 'cross_source_consistency' in outputs:
            result['cross_source_consistency'] = float(
                outputs['cross_source_consistency']['cross_source_consistency']
            )
        
        return result
    
    def detect_batch(self, texts: List[str]) -> List[Dict]:
        """
        批量检测
        
        Args:
            texts: 文本列表
            
        Returns:
            检测结果列表
        """
        results = []
        
        for text in texts:
            result = self.detect_single(text)
            results.append(result)
        
        return results
    
    def get_explanation(self, result: Dict) -> str:
        """
        生成检测结果的解释
        
        Args:
            result: 检测结果
            
        Returns:
            解释文本
        """
        explanation_parts = []
        
        # 判定结果
        if result['result'] == 'rumor':
            explanation_parts.append(f"该内容被判定为【谣言】，置信度为{result['confidence']*100:.1f}%。")
        else:
            explanation_parts.append(f"该内容被判定为【真实信息】，置信度为{result['confidence']*100:.1f}%。")
        
        # 矛盾类型
        if result['contradiction_type'] != "无矛盾":
            explanation_parts.append(f"检测到的主要问题类型：{result['contradiction_type']}。")
        
        # 知识图谱覆盖
        coverage = result['kg_coverage']
        if coverage < 0.3:
            explanation_parts.append("知识图谱覆盖率较低，部分实体无法验证。")
        elif coverage > 0.7:
            explanation_parts.append("知识图谱覆盖率较高，验证结果可靠性较强。")
        
        # 搜索触发
        if result['search_triggered']:
            explanation_parts.append("已触发动态搜索增强验证。")
        
        return " ".join(explanation_parts)


# 单例模式
_detector_instance = None


def get_detector(model_path: Optional[str] = None) -> RumorDetector:
    """
    获取检测器实例（单例）
    
    Args:
        model_path: 模型路径
        
    Returns:
        检测器实例
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = RumorDetector(model_path=model_path)
    
    return _detector_instance
