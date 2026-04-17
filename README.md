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


