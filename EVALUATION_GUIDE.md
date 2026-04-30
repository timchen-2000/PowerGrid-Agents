# RAG评估使用指南

## 📋 概述

本项目包含完整的RAG系统评估工具，可以测试检索质量和生成质量。

## 🚀 快速开始

### 1️⃣ 配置API密钥

复制`.env.example`为`.env`文件，填入您的API密钥：

```bash
cp .env.example .env
```

编辑`.env`文件：

```env
# OpenAI API配置（DeepSeek）
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_API_BASE=https://api.deepseek.com/v1

# 千问API配置（阿里云百炼）
DASHSCOPE_API_KEY=your_qianwen_api_key
```

### 2️⃣ 运行评估

#### 测试检索功能（推荐先运行）
```bash
python test_retrieval.py
```

#### 完整RAG评估
```bash
python rag_evaluator_v2.py
```

#### RAGAS专业评估（可选）
```bash
python ragas_evaluator.py
```

## 📊 评估脚本说明

### 1️⃣ test_retrieval.py - 检索功能测试
- 仅测试向量检索和混合检索功能
- 不依赖LLM生成，快速验证检索质量
- 输出找到的文档内容

### 2️⃣ rag_evaluator_v2.py - 基础RAG评估
- 测试完整的检索+生成流程
- 计算基础指标：
  - 关键词重叠度
  - 上下文覆盖率
  - 答案长度
  - 检索文档数量
- 生成JSON和CSV格式的结果文件

### 3️⃣ ragas_evaluator.py - RAGAS专业评估
使用RAGAS框架进行专业评估，包含指标：

**检索相关**：
- 上下文精确度 (Context Precision)
- 上下文召回率 (Context Recall)
- 上下文实体召回率 (Context Entity Recall)

**生成相关**：
- 忠实度 (Faithfulness) - 答案是否基于检索上下文
- 答案相关性 (Answer Relevance)
- 答案正确性 (Answer Correctness)

## 📝 测试用例

默认包含5个电力设备监控相关的测试问题：

1. 开关油压低重合闸闭锁是咋回事？
2. 变压器油温过高应该如何处理？
3. GIS设备发生SF6气体泄漏应该如何处理？
4. 母线电压异常时应该检查哪些内容？
5. 断路器拒绝合闸的原因有哪些？

## 🔧 自定义评估

### 添加自己的测试用例

修改评估脚本中的`test_cases`列表：

```python
test_cases = [
    {
        "question": "您的问题",
        "ground_truth": "标准答案"
    },
    # 添加更多测试用例
]
```

### 调整检索参数

修改`top_k`和`top_n`参数：
- `top_k`: 初始召回文档数量
- `top_n`: 重排后最终返回文档数量

## 📁 输出文件

评估完成后会生成以下文件：

| 文件名 | 格式 | 内容 |
|--------|------|------|
| rag_evaluation_results.json | JSON | 完整的评估结果 |
| rag_evaluation_results.csv | CSV | 表格格式的结果 |
| ragas_evaluation_results.csv | CSV | RAGAS专业评估结果 |

## 🎯 评估结果分析

### 检索质量
- 关键词重叠度越高，检索相关性越好
- 上下文覆盖率越高，检索到的内容越全面

### 生成质量
- 答案与标准答案的匹配度
- 答案的完整性和专业性

## 💡 提示

1. **先测试检索**：确保检索功能正常后再测试生成
2. **检查API密钥**：确保.env文件中配置正确
3. **查看输出日志**：评估过程中会显示详细的中间结果
4. **调整参数**：根据结果调整top_k和top_n优化性能

## 📞 需要帮助？

如果遇到问题：
1. 检查向量数据库是否已构建（运行main.py）
2. 确认API密钥是否正确配置
3. 查看控制台错误信息

---
祝您评估顺利！🚀
