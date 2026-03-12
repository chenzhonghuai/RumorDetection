# 微博谣言检测系统

基于多源知识图谱融合的微博谣言检测方法实现。

## 项目结构

```
src/
├── config.py                 # 配置文件
├── train.py                  # 训练脚本
├── requirements.txt          # 依赖包
├── models/                   # 模型模块
│   ├── __init__.py
│   ├── text_encoder.py       # 文本编码器 (BERT)
│   ├── knowledge_graph.py    # 知识图谱模块
│   ├── entity_alignment.py   # 跨图谱实体对齐
│   ├── consistency_verification.py  # 多维一致性验证
│   ├── dynamic_search.py     # 动态搜索增强
│   ├── adaptive_fusion.py    # 自适应知识融合
│   ├── classifier.py         # 谣言分类器
│   ├── mkgf_rd.py           # MKGF-RD模型
│   └── dsmk_rd.py           # DSMK-RD模型
├── data/                     # 数据处理模块
│   ├── __init__.py
│   ├── preprocessor.py       # 文本预处理
│   └── dataset.py            # 数据集定义
└── api/                      # Web API模块
    ├── __init__.py
    ├── detector.py           # 检测器封装
    └── app.py                # Flask应用
```

## 核心方法

### MKGF-RD (基于多源知识图谱融合的谣言检测)

1. **统一知识表示空间 (UKRS)**: 将医学、常识、事件、地理四类异构知识图谱映射到共享向量空间
2. **跨图谱实体对齐**: 基于名称、属性、结构三视图的实体匹配
3. **层次化知识融合**: 领域内融合 → 跨领域融合 → 残差增强
4. **多维一致性验证**: 实体属性、关系结构、时序逻辑、常识符合度四层验证

### DSMK-RD (融合动态搜索的多源知识增强谣言检测)

在MKGF-RD基础上增加:
1. **智能搜索触发决策**: 综合评估知识图谱覆盖率、时效性、热点匹配度
2. **多策略查询生成**: 声明提取、实体扩展、上下文、反向验证四类查询
3. **来源可信度评估**: 权威性、专业性、技术性、一致性四维评估
4. **自适应知识协同融合**: 门控机制动态平衡静态知识和动态证据

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python train.py \
    --model dsmk_rd \
    --train_data data/train.json \
    --val_data data/val.json \
    --output_dir checkpoints \
    --epochs 10 \
    --batch_size 32
```

### 启动Web服务

```bash
python -m api.app --host 0.0.0.0 --port 5000 --model checkpoints/best_model.pt
```

### API使用

**单条检测:**
```bash
curl -X POST http://localhost:5000/api/detect \
    -H "Content-Type: application/json" \
    -d '{"content": "板蓝根可以预防新冠病毒"}'
```

**批量检测:**
```bash
curl -X POST http://localhost:5000/api/detect/batch \
    -H "Content-Type: application/json" \
    -d '{"data": [{"id": 1, "content": "文本1"}, {"id": 2, "content": "文本2"}]}'
```

## 模型性能

| 指标 | MKGF-RD | DSMK-RD |
|------|---------|---------|
| 准确率 | 91.8% | 94.2% |
| 精确率 | 90.5% | 93.1% |
| 召回率 | 89.2% | 91.8% |
| F1值 | 89.8% | 92.4% |

## 引用

如果您使用了本项目的代码，请引用相关论文。
