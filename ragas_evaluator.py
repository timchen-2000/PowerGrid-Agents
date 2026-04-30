"""
使用RAGAS评估RAG系统的检索和生成效果
RAGAS: RAG Assessment - Automated Evaluation Framework for RAG Systems
"""

import os
import sys
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

sys.path.append('/workspace')

# 安装并导入RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
        answer_similarity,
        answer_correctness
    )
    from ragas.metrics.critique import harmfulness
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    print("⚠️  RAGAS未安装，正在安装...")
    os.system("pip install ragas")
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
        answer_similarity,
        answer_correctness
    )
    from ragas.metrics.critique import harmfulness
    from datasets import Dataset
    RAGAS_AVAILABLE = True

from vector_store import VectorStoreManager
from app import get_prompt
from langchain_openai import ChatOpenAI


class RAGASEvaluator:
    """使用RAGAS进行RAG系统评估"""
    
    def __init__(self):
        print("🔄 正在初始化RAGAS评估器...")
        
        # 初始化向量管理器
        self.vector_manager = VectorStoreManager(embedding_model="qianwen")
        
        # 加载向量数据库
        try:
            self.vector_manager.load_local()
            print("✅ 向量数据库加载成功")
        except FileNotFoundError:
            print("❌ 向量数据库不存在，请先运行main.py构建向量数据库")
            raise
        
        # 初始化LLM (用于评估)
        self.llm = ChatOpenAI(model_name="deepseek-chat", temperature=0.1)
        # 初始化embeddings模型 (用于评估)
        self.embeddings = self.vector_manager.embeddings
        
        print("✅ LLM初始化成功")
        print("✅ Embeddings模型初始化成功")
        
        # 测试数据
        self.test_samples = []
    
    def retrieve_context(self, question: str, top_k: int = 8, top_n: int = 3) -> List[str]:
        """检索相关上下文"""
        try:
            docs = self.vector_manager.hybrid_search_with_rerank(
                question, 
                top_k=top_k, 
                top_n=top_n
            )
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"检索失败: {e}")
            return []
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """生成答案"""
        if not contexts:
            return "抱歉，没有找到相关的知识库内容来回答您的问题。"
        
        # 构建上下文
        context_str = "以下是相关的知识库信息：\n"
        for i, ctx in enumerate(contexts, 1):
            context_str += f"\n【文档片段{i}】\n{ctx}\n"
        
        # 构建输入
        input_text = question + ("\n\n" + context_str if context_str else "")
        
        # 获取提示模板
        prompt = get_prompt()
        messages = prompt.format_messages(
            input=input_text,
            chat_history=[]
        )
        
        # 调用LLM
        response = self.llm.invoke(messages)
        return response.content
    
    def add_sample(self, user_input: str, ground_truth: str, ground_truth_contexts: List[str] = None):
        """添加评估样本"""
        # 生成答案和检索上下文
        contexts = self.retrieve_context(user_input)
        answer = self.generate_answer(user_input, contexts)
        
        sample = {
            "user_input": user_input,
            "ground_truth": ground_truth,
            "ground_truth_contexts": ground_truth_contexts or contexts,
            "retrieved_contexts": contexts,
            "response": answer
        }
        
        self.test_samples.append(sample)
        return sample
    
    def prepare_dataset(self) -> Dataset:
        """准备RAGAS数据集"""
        if not self.test_samples:
            raise ValueError("没有添加任何测试样本")
        
        # 转换为RAGAS格式
        data = {
            "user_input": [s["user_input"] for s in self.test_samples],
            "ground_truth": [s["ground_truth"] for s in self.test_samples],
            "ground_truth_contexts": [s["ground_truth_contexts"] for s in self.test_samples],
            "retrieved_contexts": [s["retrieved_contexts"] for s in self.test_samples],
            "response": [s["response"] for s in self.test_samples]
        }
        
        # 创建Dataset
        dataset = Dataset.from_dict(data)
        print(f"✅ 已创建包含 {len(self.test_samples)} 个样本的数据集")
        return dataset
    
    def run_evaluation(self, dataset: Dataset = None):
        """运行RAGAS评估"""
        if dataset is None:
            dataset = self.prepare_dataset()
        
        print("\n" + "="*60)
        print("🔍 开始RAGAS评估...")
        print("="*60)
        
        # 选择评估指标
        # 检索相关指标
        retrieval_metrics = [
            context_precision,      # 上下文精确度
            context_recall,          # 上下文召回率
            context_entity_recall,   # 上下文实体召回率
        ]
        
        # 生成相关指标
        generation_metrics = [
            faithfulness,            # 忠诚度 (答案是否基于上下文)
            answer_relevancy,        # 答案相关性
            answer_correctness,       # 答案正确性
        ]
        
        all_metrics = retrieval_metrics + generation_metrics
        
        # 运行评估
        try:
            result = evaluate(
                dataset,
                metrics=all_metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            print("\n✅ 评估完成！")
            return result
            
        except Exception as e:
            print(f"\n❌ 评估失败: {e}")
            print("\n请确保：")
            print("1. LLM API密钥配置正确")
            print("2. Embeddings模型可用")
            print("3. 测试样本数量充足")
            raise
    
    def print_results(self, result):
        """打印评估结果"""
        print("\n" + "="*60)
        print("📊 RAGAS评估结果")
        print("="*60)
        
        # 获取结果DataFrame
        result_df = result.to_pandas()
        
        # 打印各项指标
        metrics = [
            ("context_precision", "上下文精确度"),
            ("context_recall", "上下文召回率"),
            ("context_entity_recall", "上下文实体召回率"),
            ("faithfulness", "忠诚度"),
            ("answer_relevancy", "答案相关性"),
            ("answer_correctness", "答案正确性")
        ]
        
        print("\n📈 各指标得分:")
        for metric_key, metric_name in metrics:
            if metric_key in result_df.columns:
                scores = result_df[metric_key].dropna()
                if len(scores) > 0:
                    mean_score = scores.mean()
                    std_score = scores.std()
                    print(f"  {metric_name}: {mean_score:.4f} (±{std_score:.4f})")
        
        # 打印每个样本的详细结果
        print("\n📝 各样本详细结果:")
        for idx, row in result_df.iterrows():
            print(f"\n  样本 {idx+1}:")
            print(f"    问题: {row['user_input'][:50]}...")
            for metric_key, metric_name in metrics:
                if metric_key in row and pd.notna(row[metric_key]):
                    print(f"    {metric_name}: {row[metric_key]:.4f}")
        
        # 计算综合得分
        print("\n" + "="*60)
        print("📊 综合评估")
        print("="*60)
        
        retrieval_score = 0
        generation_score = 0
        retrieval_count = 0
        generation_count = 0
        
        for metric_key, _ in metrics:
            if metric_key in result_df.columns:
                scores = result_df[metric_key].dropna()
                if len(scores) > 0:
                    mean_score = scores.mean()
                    if metric_key.startswith("context_"):
                        retrieval_score += mean_score
                        retrieval_count += 1
                    else:
                        generation_score += mean_score
                        generation_count += 1
        
        if retrieval_count > 0:
            print(f"  检索能力得分: {retrieval_score/retrieval_count:.4f}")
        if generation_count > 0:
            print(f"  生成能力得分: {generation_score/generation_count:.4f}")
        
        overall_score = (retrieval_score + generation_score) / (retrieval_count + generation_count) if (retrieval_count + generation_count) > 0 else 0
        print(f"  综合得分: {overall_score:.4f}")
        
        return result_df


def main():
    """主函数"""
    print("="*60)
    print("🔬 RAGAS RAG系统评估工具")
    print("="*60)
    
    # 创建评估器
    evaluator = RAGASEvaluator()
    
    # 添加测试样本
    # 这些样本覆盖了电力设备监控的不同场景
    test_samples = [
        {
            "question": "开关油压低重合闸闭锁是咋回事？",
            "ground_truth": "开关油压低时，液压机构压力不足以驱动断路器合闸操作，此时重合闸功能会被闭锁，防止在压力不足的情况下进行操作，避免造成设备损坏或安全事故。"
        },
        {
            "question": "变压器油温过高应该如何处理？",
            "ground_truth": "变压器油温过高时，应立即检查负载情况、冷却系统运行状态，检查油位是否正常，必要时降低负载或启动备用冷却设备，并加强对变压器的监测。"
        },
        {
            "question": "GIS设备发生SF6气体泄漏应该如何处理？",
            "ground_truth": "GIS设备SF6气体泄漏时，应立即隔离故障区域，疏散人员，佩戴防护设备进行查漏，处理后补充SF6气体，并检测微水含量是否符合标准。"
        },
        {
            "question": "母线电压异常时应该检查哪些内容？",
            "ground_truth": "母线电压异常时应检查：电压互感器是否正常、母线负荷分配是否合理、无功补偿装置运行状态、系统中是否存在接地故障等。"
        },
        {
            "question": "断路器拒绝合闸的原因有哪些？",
            "ground_truth": "断路器拒绝合闸的原因包括：控制回路故障、操作机构问题、继电保护闭锁、液压/气压不足、弹簧未储能、合闸电源故障等。"
        }
    ]
    
    print(f"\n📝 准备添加 {len(test_samples)} 个测试样本...")
    for sample in test_samples:
        evaluator.add_sample(
            user_input=sample["question"],
            ground_truth=sample["ground_truth"]
        )
        print(f"  ✅ 已添加: {sample['question'][:30]}...")
    
    # 准备数据集
    dataset = evaluator.prepare_dataset()
    
    # 运行评估
    result = evaluator.run_evaluation(dataset)
    
    # 打印结果
    result_df = evaluator.print_results(result)
    
    # 保存结果
    output_file = "ragas_evaluation_results.csv"
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 详细结果已保存到: {output_file}")
    
    print("\n🎉 RAGAS评估完成！")
    
    return result


if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        print(f"\n❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()
