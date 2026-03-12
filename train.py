"""
训练脚本
MKGF-RD 和 DSMK-RD 模型训练
"""

import os
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

from config import ModelConfig, TrainingConfig, SystemConfig
from models import MKGF_RD, DSMK_RD
from data import WeiboRumorDataset, collate_fn, create_data_loaders


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    模型训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: TrainingConfig,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 最佳模型追踪
        self.best_val_f1 = 0.0
        self.best_model_state = None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['total_loss']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            predictions = outputs['predictions']
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{total_correct/total_samples:.4f}"
            })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': total_correct / total_samples
        }
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs['total_loss'].item()
                all_predictions.extend(outputs['predictions'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # 计算精确率、召回率、F1
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, num_epochs: int, save_dir: str):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}")
            
            # 评估
            val_metrics = self.evaluate()
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"P: {val_metrics['precision']:.4f}, "
                       f"R: {val_metrics['recall']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_model_state = self.model.state_dict().copy()
                
                save_path = os.path.join(save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': self.best_val_f1,
                    'val_metrics': val_metrics
                }, save_path)
                logger.info(f"Saved best model with F1: {self.best_val_f1:.4f}")
        
        logger.info(f"\nTraining completed. Best F1: {self.best_val_f1:.4f}")
        return self.best_val_f1


def main():
    parser = argparse.ArgumentParser(description="Train rumor detection model")
    parser.add_argument("--model", type=str, default="dsmk_rd",
                       choices=["mkgf_rd", "dsmk_rd"],
                       help="Model type")
    parser.add_argument("--train_data", type=str, required=True,
                       help="Training data path")
    parser.add_argument("--val_data", type=str, required=True,
                       help="Validation data path")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # 配置
    model_config = ModelConfig()
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(model_config.bert_model_name)
    
    # 数据集
    train_dataset = WeiboRumorDataset(args.train_data, tokenizer)
    val_dataset = WeiboRumorDataset(args.val_data, tokenizer)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 模型
    if args.model == "mkgf_rd":
        model = MKGF_RD(
            bert_model_name=model_config.bert_model_name,
            bert_hidden_size=model_config.bert_hidden_size,
            kg_embedding_dim=model_config.kg_embedding_dim
        )
    else:
        model = DSMK_RD(
            bert_model_name=model_config.bert_model_name,
            bert_hidden_size=model_config.bert_hidden_size,
            kg_embedding_dim=model_config.kg_embedding_dim
        )
    
    # 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=args.device
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"{args.model}_{timestamp}")
    
    trainer.train(train_config.num_epochs, save_dir)


if __name__ == "__main__":
    main()
