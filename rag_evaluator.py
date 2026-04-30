import os
import sys
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.append('/workspace')

from vector_store import VectorStoreManager
from app import get_prompt
from langchain_openai import ChatOpenAI

class RAGEvaluator:
    def __init__(self):
        print("正在初始化RAG评估器...")
        
        # 初始化向量管理器
        self.vector_manager = VectorStoreManager(embedding_model="qianwen")
        
        # 加载向量数据库
        try:
            self.vector_manager.load_local()
            print("✅ 向量数据库加载成功")
        except FileNotFoundError:
            print("❌ 向量数据库不存在，请先运行main.py构建向量数据库")
            raise
        
        # 初始化LLM
        self.llm = ChatOpenAI(model_name="deepseek-chat", temperature=0.1)
        print("✅ LLM初始化成功")
        
        # 评估数据集
        self.test_data = []
        
    def add_test_case(self, question: str, ground_truth: str):
        """添加测试用例"""
        self.test_data.append({
            "question": question,
            "ground_truth": ground_truth
        })
    
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
    
    def generate_answer(self, question: str, context: List[str]) -> str:
        """生成答案"""
        if not context:
            return "抱歉，没有找到相关的知识库内容来回答您的问题。"
        
        # 构建上下文
        context_str = "以下是相关的知识库信息：\n"
        for i, ctx in enumerate(context, 1):
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
    
    def run_rag_pipeline(self, question: str) -> Dict:
        """运行完整的RAG流程"""
        # 1. 检索上下文
        context = self.retrieve_context(question)
        
        # 2. 生成答案
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "context": context,
            "answer": answer
        }
    
    def evaluate_single(self, question: str, ground_truth: str) -> Dict:
        """评估单个问题"""
        print(f"\n{'='*60}")
        print(f"评估问题: {question}")
        print(f"{'='*60}")
        
        # 运行RAG流程
        result = self.run_rag_pipeline(question)
        
        # 打印结果
        print(f"\n📄 检索到的上下文 (共{len(result['context'])}个):")
        for i, ctx in enumerate(result['context'], 1):
            print(f"\n  [{i}] {ctx[:200]}...")
        
        print(f"\n🤖 生成的答案:\n{result['answer']}")
        print(f"\n📝 标准答案:\n{ground_truth}")
        
        # 计算简单指标
        answer_length = len(result['answer"])
        context_count = len(result['context"])
        
        print(f"\n📊 简单统计:")
        print(f"  - 答案长度: {answer_length} 字符")
        print(f"  - 检索文档数: {context_count}")
        
        return {
            "question": question,
            "ground_truth": ground_truth,
            "answer": result['answer'],
            "context": result['context'],
            "context_count": context_count,
            "answer_length": answer_length
        }
    
    def run_evaluation(self) -> pd.DataFrame:
        """运行完整评估"""
        print(f"\n{'#'*60}")
        print("# RAG系统评估报告")
        print(f"{'#'*60}")
        
        results = []
        for test_case in self.test_data:
            result = self.evaluate_single(
                test_case["question"],
                test_case["ground_truth"]
            )
            results.append(result)
        
        # 创建结果DataFrame
        df = pd.DataFrame(results)
        
        # 打印汇总统计
        print(f"\n{'#'*60}")
        print("# 评估汇总")
        print(f"{'#'*60}")
        print(f"总测试用例数: {len(results)}")
        print(f"平均检索文档数: {df['context_count'].mean():.2f}")
        print(f"平均答案长度: {df['answer_length'].mean():.2f} 字符")
        
        return df


def main():
    print("=" * 60)
    print("RAG系统评估工具")
    print("=" * 60)
    
    # 创建评估器
    evaluator = RAGEvaluator()
    
    # 添加测试用例
    # 这些测试用例应该覆盖电力设备监控的不同方面
    test_cases = [
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
    
    for case in test_cases:
        evaluator.add_test_case(case["question"], case["ground_truth"])
    
    # 运行评估
    results_df = evaluator.run_evaluation()
    
    # 保存结果
    output_file = "rag_evaluation_results.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 评估结果已保存到: {output_file}")
    
    # 返回详细结果供RAGAS进一步分析
    return results_df


if __name__ == "__main__":
    try:
        results = main()
        print("\n评估完成！")
    except Exception as e:
        print(f"\n评估失败: {e}")
        import traceback
        traceback.print_exc()
