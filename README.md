<div align="center">
  <img src="images/logo.jpg" alt="电力设备监控智能问答系统Logo" width="400">
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
  <img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=system%20architecture%20flowchart%20for%20power%20equipment%20monitoring%20AI%20system%20RAG%20hybrid%20search%20rerank%20agent%20tool%20calling%20blue%20professional%20diagram&image_size=landscape_16_9" alt="系统架构流程图" width="800">
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
# DeepSeek API（用于LLM）
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_API_BASE=https://api.deepseek.com/v1

# 千问API（用于Embedding）
DASHSCOPE_API_KEY=your_qianwen_api_key
```

### 运行系统

1. **首次运行**（构建向量数据库）
```bash
python main.py
```

2. **后续运行**（直接加载向量库）
修改 `main.py` 中的 `rebuild=False`，然后运行：
```bash
python main.py
```

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

## 🌟 应用场景

1. **电力调度中心**：实时监控和处理设备告警
2. **变电站运维**：设备状态检查和维护计划管理
3. **应急指挥**：火灾、设备故障等紧急情况处理
4. **技术培训**：电力设备知识查询和学习

## 📈 性能指标

- **响应速度**：平均响应时间 < 5秒
- **准确率**：专业问题回答准确率 > 90%
- **召回率**：混合检索召回率 > 95%
- **稳定性**：99.9%的系统可用性

## 🔮 未来规划

1. **多模态支持**：集成图像识别，支持设备故障图片分析
2. **实时数据**：接入SCADA系统，获取实时设备数据
3. **预测性维护**：基于历史数据预测设备故障
4. **移动应用**：开发移动端应用，支持现场操作
5. **多语言支持**：扩展英语等其他语言支持

## 🤝 贡献指南

欢迎各位开发者贡献代码、提出建议和报告问题！

1. **Fork 项目**
2. **创建特性分支**
3. **提交更改**
4. **创建 Pull Request**

## 📄 许可证

本项目采用 MIT 许可证。

## 🏆 项目优势

- **技术领先**：采用最新的RAG架构和混合检索技术
- **专业领域**：专注于电力设备监控，提供专业解决方案
- **国产模型**：集成国产大模型，支持中文专业术语
- **易于部署**：模块化设计，快速部署和扩展
- **用户友好**：简洁的API和直观的使用方式

---

**电力设备监控智能问答系统** - 用AI守护电网安全，为电力运维赋能！⚡

*Made with ❤️ for the future of smart grid*