<div align="center">
  <img src="images/logo.png" alt="电力设备监控智能问答系统Logo" width="400">
  <h1>电力设备监控智能问答系统</h1>
</div>

## 项目简介

基于RAG架构的智能助手，专为电力设备监控领域设计，集成国产大模型、混合检索和多轮对话能力。

## 核心特色
- 混合检索引擎（BM25 + FAISS + BGE重排）
- 支持千问、DeepSeek等国产大模型
- 使用千问模型作为embedding模型
- 电力设备专用工具集
- 多轮对话能力
- 专业领域知识覆盖

## 千问模型Embedding参考代码
```python
import os
from openai import OpenAI

input_text = "衣服的质量杠杠的"

client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    # 各地域的API Key不同。获取API Key：`https://help.aliyun.com/zh/model-studio/get-api-key`
    api_key=os.getenv("DASHSCOPE_API_KEY"),  
    # 以下是北京地域base-url，如果使用新加坡地域的模型，需要将base_url替换为：`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
    base_url="`https://dashscope.aliyuncs.com/compatible-mode/v1`"
)

completion = client.embeddings.create(
    model="text-embedding-v4",
    input=input_text
)

print(completion.model_dump_json())
```

## 技术路线图

### 当前实现
- **架构**：基于RAG架构的智能问答系统
- **检索**：BM25关键词检索 + FAISS向量相似度检索 + BGE重排
- **模型**：支持千问、DeepSeek等国产大模型，使用千问模型作为embedding
- **界面**：前后端分离，FastAPI后端 + HTML/CSS/JavaScript前端
- **部署**：提供启动脚本，支持Linux/Mac/Windows

### 短期目标（1-3个月）
- 模型优化：评估和选择最佳embedding模型，优化大模型调用参数
- 功能增强：增加电力设备专用工具集的功能，实现更智能的多轮对话管理
- 性能优化：优化向量数据库检索速度，减少API调用延迟

### 中期目标（3-6个月）
- 多模态支持：集成图像识别功能，支持设备图片分析
- 知识图谱集成：构建电力设备领域知识图谱，实现基于知识图谱的推理能力
- 部署优化：支持Docker容器化部署，实现Kubernetes集群部署

### 长期目标（6个月以上）
- 自主学习能力：实现系统自动从对话中学习新知识
- 行业扩展：扩展到其他工业设备领域，支持多语言能力
- 智能运维平台：集成实时监控数据，实现预测性维护建议

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥
1. 复制 `.env.example` 文件并重命名为 `.env`
2. 编辑 `.env` 文件，填入API密钥：
```
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_API_BASE=https://api.deepseek.com/v1
DASHSCOPE_API_KEY=your_qianwen_api_key
```

### 3. 运行系统
- **推荐**：使用启动脚本
  - Linux/Mac：`./start.sh`
  - Windows：`start.bat`
- **手动运行**：
  1. 首次运行：`python main.py`（构建向量数据库）
  2. 后端服务：`python app.py`
  3. 前端服务：`python -m http.server 3000`

## 访问地址
- 后端服务：http://localhost:8000
- API文档：http://localhost:8000/docs
- 前端界面：http://localhost:3000

## 故障排除
- 端口占用：停止占用端口的进程
- API密钥：确保在 `.env` 文件中正确配置
- 向量数据库：确保PDF文档存在且格式正确
- 前端连接：检查后端服务是否正常运行


