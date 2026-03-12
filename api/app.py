"""
Flask Web API
微博谣言检测在线平台后端
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd

from .detector import RumorDetector, get_detector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(model_path: Optional[str] = None) -> Flask:
    """
    创建Flask应用
    
    Args:
        model_path: 模型权重路径
        
    Returns:
        Flask应用实例
    """
    app = Flask(__name__)
    CORS(app)  # 允许跨域
    
    # 配置
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
    app.config['JSON_AS_ASCII'] = False
    
    # 初始化检测器
    detector = get_detector(model_path)
    
    # ==================== API路由 ====================
    
    @app.route('/')
    def index():
        """首页"""
        return render_template_string(INDEX_HTML)
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """健康检查"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/api/detect', methods=['POST'])
    def detect_single():
        """
        单条检测API
        
        Request Body:
            {
                "content": "待检测的微博文本"
            }
            
        Response:
            {
                "result": "rumor" | "real",
                "confidence": 0.95,
                "probability": {"real": 0.05, "rumor": 0.95},
                "contradiction_type": "医学事实矛盾",
                "search_triggered": true,
                "kg_coverage": 0.75,
                "entities_found": ["板蓝根", "新冠"],
                "processing_time_ms": 287
            }
        """
        try:
            data = request.get_json()
            
            if not data or 'content' not in data:
                return jsonify({
                    'error': '请提供待检测的文本内容',
                    'code': 400
                }), 400
            
            content = data['content'].strip()
            
            if not content:
                return jsonify({
                    'error': '文本内容不能为空',
                    'code': 400
                }), 400
            
            if len(content) > 500:
                return jsonify({
                    'error': '文本长度不能超过500字符',
                    'code': 400
                }), 400
            
            # 检测
            result = detector.detect_single(content)
            
            # 添加解释
            result['explanation'] = detector.get_explanation(result)
            
            logger.info(f"Detected: {content[:50]}... -> {result['result']}")
            
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return jsonify({
                'error': f'检测失败: {str(e)}',
                'code': 500
            }), 500
    
    @app.route('/api/detect/batch', methods=['POST'])
    def detect_batch():
        """
        批量检测API
        
        Request Body:
            {
                "data": [
                    {"id": 1, "content": "文本1"},
                    {"id": 2, "content": "文本2"}
                ]
            }
            
        Response:
            {
                "results": [...],
                "summary": {
                    "total": 100,
                    "rumor_count": 42,
                    "real_count": 58,
                    "avg_confidence": 0.87
                }
            }
        """
        try:
            data = request.get_json()
            
            if not data or 'data' not in data:
                return jsonify({
                    'error': '请提供待检测的数据',
                    'code': 400
                }), 400
            
            items = data['data']
            
            if not items:
                return jsonify({
                    'error': '数据列表不能为空',
                    'code': 400
                }), 400
            
            if len(items) > 10000:
                return jsonify({
                    'error': '单次最多检测10000条数据',
                    'code': 400
                }), 400
            
            # 批量检测
            results = []
            rumor_count = 0
            total_confidence = 0.0
            
            for item in items:
                content = item.get('content', '').strip()
                item_id = item.get('id', len(results) + 1)
                
                if not content:
                    results.append({
                        'id': item_id,
                        'error': '内容为空'
                    })
                    continue
                
                result = detector.detect_single(content)
                result['id'] = item_id
                results.append(result)
                
                if result['result'] == 'rumor':
                    rumor_count += 1
                total_confidence += result['confidence']
            
            # 统计摘要
            valid_count = len([r for r in results if 'error' not in r])
            summary = {
                'total': len(items),
                'valid_count': valid_count,
                'rumor_count': rumor_count,
                'real_count': valid_count - rumor_count,
                'avg_confidence': total_confidence / valid_count if valid_count > 0 else 0
            }
            
            logger.info(f"Batch detection: {len(items)} items, {rumor_count} rumors")
            
            return jsonify({
                'results': results,
                'summary': summary
            })
        
        except Exception as e:
            logger.error(f"Batch detection error: {str(e)}")
            return jsonify({
                'error': f'批量检测失败: {str(e)}',
                'code': 500
            }), 500
    
    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """
        文件上传API
        支持CSV和Excel格式
        """
        try:
            if 'file' not in request.files:
                return jsonify({
                    'error': '请上传文件',
                    'code': 400
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'error': '文件名不能为空',
                    'code': 400
                }), 400
            
            # 检查文件格式
            filename = file.filename.lower()
            if filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                return jsonify({
                    'error': '不支持的文件格式，请上传CSV或Excel文件',
                    'code': 400
                }), 400
            
            # 检查必要列
            if 'content' not in df.columns:
                return jsonify({
                    'error': '文件缺少content列',
                    'code': 400
                }), 400
            
            # 转换为检测格式
            data = []
            for idx, row in df.iterrows():
                item = {
                    'id': row.get('id', idx + 1),
                    'content': str(row['content'])
                }
                data.append(item)
            
            return jsonify({
                'message': '文件解析成功',
                'data': data,
                'total': len(data)
            })
        
        except Exception as e:
            logger.error(f"File upload error: {str(e)}")
            return jsonify({
                'error': f'文件解析失败: {str(e)}',
                'code': 500
            }), 500
    
    @app.route('/api/template', methods=['GET'])
    def download_template():
        """下载数据模板"""
        template_data = {
            'id': [1, 2, 3],
            'content': [
                '示例文本1：某地发生重大事件',
                '示例文本2：专家称某食物有害健康',
                '示例文本3：官方发布最新通知'
            ]
        }
        df = pd.DataFrame(template_data)
        
        # 返回CSV
        csv_data = df.to_csv(index=False)
        
        return csv_data, 200, {
            'Content-Type': 'text/csv; charset=utf-8',
            'Content-Disposition': 'attachment; filename=template.csv'
        }
    
    return app


# 简单的前端页面
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>微博谣言在线检测平台</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .header p { opacity: 0.9; }
        .card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .card h2 {
            color: #333;
            margin-bottom: 16px;
            font-size: 1.2em;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 32px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 16px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 20px;
            padding: 16px;
            border-radius: 8px;
            display: none;
        }
        .result.rumor {
            background: #ffebee;
            border: 1px solid #ef5350;
        }
        .result.real {
            background: #e8f5e9;
            border: 1px solid #66bb6a;
        }
        .result-header {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }
        .result-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        .result.rumor .result-badge {
            background: #ef5350;
            color: white;
        }
        .result.real .result-badge {
            background: #66bb6a;
            color: white;
        }
        .result-confidence {
            margin-left: auto;
            font-size: 14px;
            color: #666;
        }
        .result-details {
            font-size: 14px;
            color: #555;
            line-height: 1.6;
        }
        .result-details p { margin-bottom: 8px; }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading::after {
            content: '';
            display: inline-block;
            width: 24px;
            height: 24px;
            border: 3px solid #667eea;
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .examples {
            margin-top: 16px;
        }
        .examples p {
            font-size: 13px;
            color: #666;
            margin-bottom: 8px;
        }
        .example-btn {
            background: #f5f5f5;
            border: 1px solid #ddd;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            margin-right: 8px;
            margin-bottom: 8px;
            transition: background 0.2s;
        }
        .example-btn:hover {
            background: #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 微博谣言在线检测平台</h1>
            <p>基于多源知识图谱融合的智能谣言检测系统</p>
        </div>
        
        <div class="card">
            <h2>📝 输入待检测内容</h2>
            <textarea id="content" placeholder="请输入微博文本内容..."></textarea>
            
            <div class="examples">
                <p>示例文本：</p>
                <button class="example-btn" onclick="setExample(1)">医疗谣言示例</button>
                <button class="example-btn" onclick="setExample(2)">社会事件示例</button>
                <button class="example-btn" onclick="setExample(3)">科技谣言示例</button>
            </div>
            
            <button class="btn" id="detectBtn" onclick="detect()">开始检测</button>
            
            <div class="loading" id="loading">检测中...</div>
            
            <div class="result" id="result">
                <div class="result-header">
                    <span class="result-badge" id="resultBadge"></span>
                    <span class="result-confidence" id="resultConfidence"></span>
                </div>
                <div class="result-details" id="resultDetails"></div>
            </div>
        </div>
        
        <div class="card">
            <h2>📊 系统说明</h2>
            <div style="font-size: 14px; color: #555; line-height: 1.8;">
                <p>本系统采用 <strong>DSMK-RD</strong> 方法，融合以下技术：</p>
                <ul style="margin-left: 20px; margin-top: 8px;">
                    <li>多源知识图谱融合（医学、常识、事件、地理）</li>
                    <li>动态搜索增强验证</li>
                    <li>多维一致性验证框架</li>
                    <li>自适应知识协同融合</li>
                </ul>
            </div>
        </div>
    </div>
    
    <script>
        const examples = {
            1: "板蓝根可以预防新冠病毒，专家建议每天服用",
            2: "紧急通知：某市自来水氯含量严重超标，请大家暂停使用",
            3: "新款iPhone将配备石墨烯电池，充电5分钟可用一周"
        };
        
        function setExample(id) {
            document.getElementById('content').value = examples[id];
        }
        
        async function detect() {
            const content = document.getElementById('content').value.trim();
            if (!content) {
                alert('请输入待检测的文本内容');
                return;
            }
            
            const btn = document.getElementById('detectBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            btn.disabled = true;
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                // 显示结果
                result.className = 'result ' + data.result;
                result.style.display = 'block';
                
                document.getElementById('resultBadge').textContent = 
                    data.result === 'rumor' ? '⚠️ 谣言' : '✅ 真实';
                document.getElementById('resultConfidence').textContent = 
                    '置信度: ' + (data.confidence * 100).toFixed(1) + '%';
                
                let details = `
                    <p><strong>判定依据：</strong>${data.explanation}</p>
                    <p><strong>矛盾类型：</strong>${data.contradiction_type}</p>
                    <p><strong>知识图谱覆盖率：</strong>${(data.kg_coverage * 100).toFixed(1)}%</p>
                    <p><strong>识别实体：</strong>${data.entities_found.join('、') || '无'}</p>
                    <p><strong>处理时间：</strong>${data.processing_time_ms}ms</p>
                `;
                document.getElementById('resultDetails').innerHTML = details;
                
            } catch (error) {
                alert('检测失败: ' + error.message);
            } finally {
                btn.disabled = false;
                loading.style.display = 'none';
            }
        }
    </script>
</body>
</html>
'''


def run_server(host: str = "0.0.0.0", port: int = 5000, model_path: str = None):
    """
    运行服务器
    
    Args:
        host: 主机地址
        port: 端口号
        model_path: 模型路径
    """
    app = create_app(model_path)
    logger.info(f"Starting server at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run rumor detection API server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    
    args = parser.parse_args()
    run_server(args.host, args.port, args.model)
