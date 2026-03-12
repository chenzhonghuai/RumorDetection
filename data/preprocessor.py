"""
数据预处理模块
文本清洗、实体识别、特征提取
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Entity:
    """实体数据类"""
    text: str
    type: str
    start: int
    end: int
    kg_type: Optional[str] = None


class TextPreprocessor:
    """
    文本预处理器
    执行Unicode规范化、HTML标签移除、URL过滤、表情符号清理等操作
    """
    
    def __init__(self):
        # URL正则
        self.url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+|'
            r'http://t\.cn/[a-zA-Z0-9]+|'
            r'www\.[^\s<>"{}|\\^`\[\]]+'
        )
        
        # HTML标签正则
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # 微博话题正则
        self.topic_pattern = re.compile(r'#([^#]+)#')
        
        # @用户正则
        self.mention_pattern = re.compile(r'@[\w\u4e00-\u9fff]+')
        
        # 表情符号正则（包括emoji和微博表情）
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F]|'  # emoticons
            r'[\U0001F300-\U0001F5FF]|'  # symbols & pictographs
            r'[\U0001F680-\U0001F6FF]|'  # transport & map symbols
            r'[\U0001F1E0-\U0001F1FF]|'  # flags
            r'[\U00002702-\U000027B0]|'
            r'[\U0001f926-\U0001f937]|'
            r'[\U00010000-\U0010ffff]|'
            r'\[[\u4e00-\u9fff]+\]'  # 微博表情 [哈哈]
        )
        
        # 多余空白正则
        self.whitespace_pattern = re.compile(r'\s+')
    
    def normalize_unicode(self, text: str) -> str:
        """
        Unicode标准化
        将全角字符转换为半角，规范化标点符号
        """
        # NFKC标准化
        text = unicodedata.normalize('NFKC', text)
        
        # 全角转半角映射
        full_to_half = {
            '，': ',', '。': '.', '！': '!', '？': '?',
            '：': ':', '；': ';', '"': '"', '"': '"',
            ''': "'", ''': "'", '（': '(', '）': ')',
            '【': '[', '】': ']', '｛': '{', '｝': '}',
            '～': '~', '＠': '@', '＃': '#', '＄': '$',
            '％': '%', '＆': '&', '＊': '*', '＋': '+',
            '－': '-', '／': '/', '＝': '=', '＜': '<',
            '＞': '>', '｜': '|', '＼': '\\', '＾': '^',
            '＿': '_', '｀': '`'
        }
        
        for full, half in full_to_half.items():
            text = text.replace(full, half)
        
        return text
    
    def remove_html_tags(self, text: str) -> str:
        """移除HTML标签"""
        return self.html_pattern.sub('', text)
    
    def remove_urls(self, text: str) -> str:
        """移除URL"""
        return self.url_pattern.sub('', text)
    
    def extract_topics(self, text: str) -> Tuple[str, List[str]]:
        """
        提取并处理话题标签
        返回处理后的文本和话题列表
        """
        topics = self.topic_pattern.findall(text)
        # 保留话题内容，移除#符号
        processed = self.topic_pattern.sub(r'\1', text)
        return processed, topics
    
    def remove_mentions(self, text: str) -> str:
        """移除@用户"""
        return self.mention_pattern.sub('', text)
    
    def remove_emojis(self, text: str) -> str:
        """移除表情符号"""
        return self.emoji_pattern.sub('', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        return self.whitespace_pattern.sub(' ', text).strip()
    
    def preprocess(self, text: str, keep_topics: bool = True) -> Dict[str, any]:
        """
        完整预处理流程
        
        Args:
            text: 原始文本
            keep_topics: 是否保留话题内容
            
        Returns:
            预处理结果字典
        """
        original = text
        
        # 1. Unicode标准化
        text = self.normalize_unicode(text)
        
        # 2. 移除HTML标签
        text = self.remove_html_tags(text)
        
        # 3. 移除URL
        text = self.remove_urls(text)
        
        # 4. 处理话题
        text, topics = self.extract_topics(text)
        
        # 5. 移除@用户
        text = self.remove_mentions(text)
        
        # 6. 移除表情
        text = self.remove_emojis(text)
        
        # 7. 规范化空白
        text = self.normalize_whitespace(text)
        
        return {
            "original": original,
            "cleaned": text,
            "topics": topics,
            "length": len(text)
        }


class EntityRecognizer:
    """
    命名实体识别器
    识别医学实体、通用实体、事件实体和地理实体
    """
    
    # 实体类型到知识图谱的映射
    ENTITY_TO_KG = {
        "disease": "medical",
        "symptom": "medical",
        "drug": "medical",
        "treatment": "medical",
        "body_part": "medical",
        "person": "commonsense",
        "organization": "commonsense",
        "product": "commonsense",
        "concept": "commonsense",
        "event": "event",
        "time": "event",
        "number": "event",
        "location": "geo",
        "facility": "geo"
    }
    
    def __init__(self):
        # 医学实体词典（简化版）
        self.medical_dict = {
            "diseases": ["感冒", "发烧", "咳嗽", "新冠", "肺炎", "糖尿病", "高血压", "癌症"],
            "symptoms": ["头痛", "恶心", "呕吐", "腹泻", "乏力", "发热"],
            "drugs": ["阿司匹林", "布洛芬", "板蓝根", "连花清瘟", "维生素"],
            "treatments": ["手术", "化疗", "放疗", "针灸", "按摩"]
        }
        
        # 时间词典
        self.time_keywords = [
            "今天", "昨天", "明天", "今日", "昨日", "近日",
            "上午", "下午", "晚上", "凌晨",
            "周一", "周二", "周三", "周四", "周五", "周六", "周日",
            "一月", "二月", "三月", "四月", "五月", "六月",
            "七月", "八月", "九月", "十月", "十一月", "十二月"
        ]
        
        # 地点词典（简化版）
        self.location_keywords = [
            "北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "成都",
            "中国", "美国", "日本", "韩国", "英国", "法国", "德国",
            "医院", "学校", "公司", "政府", "机场", "车站"
        ]
    
    def recognize(self, text: str) -> List[Entity]:
        """
        识别文本中的实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities = []
        
        # 识别医学实体
        for entity_type, keywords in self.medical_dict.items():
            for keyword in keywords:
                for match in re.finditer(re.escape(keyword), text):
                    entity = Entity(
                        text=keyword,
                        type=entity_type.rstrip('s'),  # 去掉复数s
                        start=match.start(),
                        end=match.end(),
                        kg_type="medical"
                    )
                    entities.append(entity)
        
        # 识别时间实体
        for keyword in self.time_keywords:
            for match in re.finditer(re.escape(keyword), text):
                entity = Entity(
                    text=keyword,
                    type="time",
                    start=match.start(),
                    end=match.end(),
                    kg_type="event"
                )
                entities.append(entity)
        
        # 识别地点实体
        for keyword in self.location_keywords:
            for match in re.finditer(re.escape(keyword), text):
                entity = Entity(
                    text=keyword,
                    type="location",
                    start=match.start(),
                    end=match.end(),
                    kg_type="geo"
                )
                entities.append(entity)
        
        # 识别数字（可能是数值声明）
        number_pattern = re.compile(r'\d+(?:\.\d+)?(?:%|倍|次|个|人|元|万|亿)?')
        for match in number_pattern.finditer(text):
            entity = Entity(
                text=match.group(),
                type="number",
                start=match.start(),
                end=match.end(),
                kg_type="event"
            )
            entities.append(entity)
        
        # 去重（按位置）
        seen_positions = set()
        unique_entities = []
        for entity in entities:
            pos = (entity.start, entity.end)
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_entities.append(entity)
        
        # 按位置排序
        unique_entities.sort(key=lambda e: e.start)
        
        return unique_entities
    
    def group_by_kg(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """
        按知识图谱类型分组实体
        
        Args:
            entities: 实体列表
            
        Returns:
            分组后的实体字典
        """
        grouped = {
            "medical": [],
            "commonsense": [],
            "event": [],
            "geo": []
        }
        
        for entity in entities:
            if entity.kg_type in grouped:
                grouped[entity.kg_type].append(entity)
        
        return grouped


class FeatureExtractor:
    """
    特征提取器
    提取文本的各种统计特征
    """
    
    def __init__(self):
        # 情感词典（简化版）
        self.positive_words = ["好", "棒", "赞", "优秀", "成功", "安全"]
        self.negative_words = ["坏", "差", "糟", "失败", "危险", "假"]
        
        # 夸张词
        self.exaggeration_words = [
            "震惊", "惊人", "绝对", "一定", "必须", "紧急",
            "重大", "突发", "爆炸", "疯狂", "史上最"
        ]
        
        # 催促词
        self.urgency_words = [
            "快转", "速看", "赶紧", "马上", "立即", "紧急扩散"
        ]
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        提取文本特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征字典
        """
        features = {}
        
        # 基本统计
        features["length"] = len(text)
        features["word_count"] = len(text.split())
        
        # 标点符号统计
        features["exclamation_count"] = text.count('!') + text.count('！')
        features["question_count"] = text.count('?') + text.count('？')
        
        # 情感特征
        positive_count = sum(1 for w in self.positive_words if w in text)
        negative_count = sum(1 for w in self.negative_words if w in text)
        features["positive_ratio"] = positive_count / max(len(text), 1)
        features["negative_ratio"] = negative_count / max(len(text), 1)
        
        # 夸张特征
        exaggeration_count = sum(1 for w in self.exaggeration_words if w in text)
        features["exaggeration_ratio"] = exaggeration_count / max(len(text.split()), 1)
        
        # 催促特征
        urgency_count = sum(1 for w in self.urgency_words if w in text)
        features["urgency_ratio"] = urgency_count / max(len(text.split()), 1)
        
        # 数字密度
        digit_count = sum(1 for c in text if c.isdigit())
        features["digit_ratio"] = digit_count / max(len(text), 1)
        
        return features
