<div align="center">
  <img src="images/logo.png" alt="电力设备监控智能问答系统Logo" width="400">
  <h1>电力设备监控智能问答系统</h1>
</div>

## 🎖️ 项目简介

![电力设备监控系统](images/power_monitoring.jpg)

**电力设备监控智能问答系统**是一个基于先进RAG（检索增强生成）架构的智能助手，专为电力设备监控领域设计。系统集成了**国产大模型**、**混合检索**、**智能工具调用**和**多轮对话**能力，为电力设备运维人员提供专业、实时的智能问答服务。

## 🌟 核心特色

### 🚀 先进技术架构
- **混合检索引擎**：BM25关键词检索 + FAISS向量相似度检索 + BGE重排
- **国产模型集成**：支持千问、DeepSeek等国产大模型
- **智能工具链**：电力设备专用工具集，包括状态检查、告警处理、维护管理
- **多轮对话**：上下文理解和记忆能力

### 🎯 专业领域能力
- **设备状态监控**：实时检查变压器、断路器等设备状态
- **告警管理**：查询、确认和处理设备告警
- **维护计划**：创建和管理设备维护计划
- **应急处理**：火灾、设备故障等紧急情况的专业指导

### 💡 技术创新
- **召回-重排流程**：先召回Top K → 再Rerank → 再取Top N
- **智能降级**：多级容错机制，确保系统稳定性
- **高效向量化**：千问embedding模型，支持中文专业术语
- **模块化设计**：清晰的代码结构，易于扩展和维护

## 🛠️ 系统架构

<div align="center">
  <img src="images/stream.png" alt="系统架构流程图" width="800">
</div>

```
用户查询
    ↓
混合检索（BM25 + 向量）→ 召回 Top K
    ↓
重排（BGE模型）→ 计算相似度排序
    ↓
取 Top N → 选择最相关的文档
    ↓
Agent工具调用 → 设备状态检查、告警处理
    ↓
智能回答生成 → 专业、详细的中文回答
```

## 📁 项目结构

```
├── main.py                # 系统入口，初始化和测试
├── vector_store.py        # 向量数据库管理，混合检索和重排
├── qa_agent.py           # Agent实现，工具调用和对话管理
├── power_equipment_tools.py # 电力设备专用工具集
├── document_processor.py  # PDF文档处理和切分
├── app.py                # FastAPI后端服务
├── index.html            # 前端聊天界面
├── start.sh              # 启动脚本（Linux/Mac）
├── start.bat             # 启动脚本（Windows）
├── .env                  # API配置文件
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明
```

## 🚀 快速开始

### 环境配置

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置API密钥**
编辑 `.env` 文件：
```
# OpenAI API配置（DeepSeek）
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_API_BASE=https://api.deepseek.com/v1

# 千问API配置（阿里云百炼）
DASHSCOPE_API_KEY=your_qianwen_api_key
```

### 运行系统

#### 方法1：使用启动脚本（推荐）

**Linux/Mac**：
```bash
./start.sh
```

**Windows**：
```bash
start.bat
```

#### 方法2：手动运行

1. **首次运行**（构建向量数据库）
```bash
python main.py
```

2. **运行后端服务**
```bash
python app.py
```

3. **运行前端服务**
```bash
python -m http.server 3000
```

## 🌐 访问地址

- **后端服务**：http://localhost:8000
- **API文档**：http://localhost:8000/docs
- **前端界面**：http://localhost:3000

## 🎯 功能演示

### 🔍 知识问答
- **问题**：变压器有载重瓦斯出口的常见原因是啥
- **回答**：详细的专业解释，包括原因分析、后果和处置方法

### 🛠️ 工具调用
- **设备状态检查**：`check_equipment_status('变压器#2')`
- **告警查询**：`query_alerts(status='未处理')`
- **维护计划管理**：`get_maintenance_plans()`
- **告警确认**：`acknowledge_alert(alert_id=1)`

### 🚨 应急处理
- **火灾预警**：提供专业的火灾应急处理流程
- **设备故障**：详细的故障分析和处理建议

## 🔧 技术参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| chunk_size | 文档切分大小 | 500字符 |
| chunk_overlap | 文档重叠大小 | 100字符 |
| top_k | 初始召回数量 | 8 |
| top_n | 最终返回数量 | 3 |
| embedding_model | Embedding模型 | 千问 |
| LLM | 大语言模型 | DeepSeek |

## 🚀 FastAPI服务

### API接口

#### 1. 智能问答
- **接口**：`POST /ask`
- **请求体**：
  ```json
  {
    "question": "变压器温度过高怎么办？",
    "conversation_id": "default"
  }
  ```
- **响应**：
  ```json
  {
    "answer": "专业的回答内容",
    "source_documents": [
      {
        "content": "相关文档片段内容...",
        "metadata": {}
      }
    ],
    "conversation_id": "default",
    "status": "success"
  }
  ```

#### 2. 健康检查
- **接口**：`GET /health`
- **响应**：
  ```json
  {
    "status": "healthy",
    "service": "power-equipment-agent",
    "system_initialized": true
  }
  ```

#### 3. 系统信息
- **接口**：`GET /info`
- **响应**：
  ```json
  {
    "name": "电力设备监控智能问答系统",
    "version": "1.0.0",
    "features": [
      "基于RAG架构的智能问答",
      "混合检索（BM25 + 向量）",
      "智能工具调用",
      "多轮对话能力",
      "应急处理指导",
      "国产大模型集成"
    ],
    "system_initialized": true
  }
  ```

#### 4. 清空对话历史
- **接口**：`POST /clear_history`
- **响应**：
  ```json
  {
    "status": "success",
    "message": "对话历史已清空"
  }
  ```

### API文档

访问 `http://localhost:8000/docs` 可以查看交互式API文档，测试所有接口。

## 📡 客户端调用示例

### Python示例

```python
import requests

# 智能问答
response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "变压器有载重瓦斯出口的常见原因是啥",
        "conversation_id": "user123"
    }
)
print(response.json())

# 健康检查
response = requests.get("http://localhost:8000/health")
print(response.json())

# 清空对话历史
response = requests.post("http://localhost:8000/clear_history")
print(response.json())
```

### cURL示例

```bash
# 智能问答
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "变压器温度过高怎么办？", "conversation_id": "default"}'

# 健康检查
curl http://localhost:8000/health

# 系统信息
curl http://localhost:8000/info

# 清空对话历史
curl -X POST http://localhost:8000/clear_history
```

## 📱 前端界面使用

1. 打开浏览器，访问 `http://localhost:3000`
2. 在输入框中输入您的问题，例如："变压器温度过高怎么办？"
3. 点击"发送"按钮或按Enter键提交问题
4. 系统会返回专业的回答，显示在聊天界面中

## 🔧 故障排除

### 1. 端口占用
- **问题**：端口8000或3000被占用
- **解决方案**：停止占用端口的进程，或修改服务端口

### 2. API密钥配置
- **问题**：系统提示API密钥未配置
- **解决方案**：确保在 `.env` 文件中正确配置了API密钥

### 3. 向量数据库构建
- **问题**：首次运行时向量数据库构建失败
- **解决方案**：确保PDF文档存在且格式正确

### 4. 前端连接问题
- **问题**：前端无法连接到后端API
- **解决方案**：检查后端服务是否正常运行，确保网络连接正常

## 📄 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
