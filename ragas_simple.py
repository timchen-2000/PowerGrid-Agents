#!/usr/bin/env python3
"""
RAGAS评估脚本 - 精简版
评估指标：Faithfulness、Answer Relevance、Context Relevance
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append('/workspace')

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
    from datasets import Dataset
except ImportError:
    print("正在安装 RAGAS...")
    os.system("pip install ragas datasets -q")
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
    from datasets import Dataset

from vector_store import VectorStoreManager
from app import get_prompt
from langchain_openai import ChatOpenAI


class SimpleRAGASEvaluator:
    def __init__(self):
        print("=" * 60)
        print("🔄 初始化RAGAS评估器...")
        print("=" * 60)
        
        self.vector_manager = VectorStoreManager(embedding_model="qianwen")
        
        try:
            self.vector_manager.load_local()
            print("✅ 向量数据库加载成功")
        except FileNotFoundError:
            print("正在构建向量数据库...")
            from document_processor import DocumentProcessor
            doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
            documents = doc_processor.process_pdf("变电设备监控信息释义及处置原则.pdf")
            self.vector_manager.create_from_documents(documents)
            self.vector_manager.save_local()
            print("✅ 向量数据库构建完成")
        
        self.llm = ChatOpenAI(model_name="deepseek-chat", temperature=0.1)
        self.embeddings = self.vector_manager.embeddings
        print("✅ LLM和Embedding模型初始化成功")
        
        self.test_samples = []
    
    def retrieve_and_generate(self, question: str) -> dict:
        """执行检索和生成"""
        contexts = []
        try:
            docs = self.vector_manager.hybrid_search_with_rerank(
                question, top_k=8, top_n=3
            )
            contexts = [doc.page_content for doc in docs]
        except Exception as e:
            print(f"检索失败: {e}")
        
        if not contexts:
            return {
                "user_input": question,
                "response": "抱歉，无法找到相关信息",
                "retrieved_contexts": [],
                "ground_truth": ""
            }
        
        context_str = "以下是相关的知识库信息：\n"
        for i, ctx in enumerate(contexts, 1):
            context_str += f"\n【文档片段{i}】\n{ctx}\n"
        
        input_text = question + "\n\n" + context_str
        prompt = get_prompt()
        messages = prompt.format_messages(input=input_text, chat_history=[])
        response = self.llm.invoke(messages)
        
        return {
            "user_input": question,
            "response": response.content,
            "retrieved_contexts": contexts,
            "ground_truth": ""  # 可手动添加标准答案
        }
    
    def evaluate(self, questions: list):
        """执行RAGAS评估"""
        print("\n" + "=" * 60)
        print("🔍 开始RAGAS评估")
        print("=" * 60)
        print("评估指标：")
        print("  1. Faithfulness（忠实度）")
        print("  2. Answer Relevance（答案相关性）")
        print("  3. Context Relevance（上下文相关性）")
        print("=" * 60)
        
        for idx, question in enumerate(questions, 1):
            print(f"\n[{idx}/{len(questions)}] 处理: {question}")
            sample = self.retrieve_and_generate(question)
            self.test_samples.append(sample)
        
        print("\n✅ 数据准备完成，开始评估...")
        
        dataset = Dataset.from_dict({
            "user_input": [s["user_input"] for s in self.test_samples],
            "response": [s["response"] for s in self.test_samples],
            "retrieved_contexts": [s["retrieved_contexts"] for s in self.test_samples],
            "ground_truth": [s["ground_truth"] for s in self.test_samples]
        })
        
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_relevancy],
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        return result
    
    def print_results(self, result):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("📊 RAGAS评估结果")
        print("=" * 60)
        
        result_df = result.to_pandas()
        
        metrics_info = [
            ("faithfulness", "Faithfulness（忠实度）"),
            ("answer_relevancy", "Answer Relevance（答案相关性）"),
            ("context_relevancy", "Context Relevance（上下文相关性）")
        ]
        
        print("\n📈 各指标平均得分:")
        for col, name in metrics_info:
            if col in result_df.columns:
                mean_val = result_df[col].mean()
                std_val = result_df[col].std()
                print(f"  {name}: {mean_val:.4f} (±{std_val:.4f})")
        
        print("\n📝 各样本详细结果:")
        for idx, row in result_df.iterrows():
            print(f"\n  样本 {idx+1}: {row['user_input'][:40]}...")
            for col, name in metrics_info:
                if col in row and row[col] is not None:
                    print(f"    {name}: {row[col]:.4f}")
        
        result_df.to_csv("ragas_results.csv", index=False)
        print("\n✅ 结果已保存至 ragas_results.csv")
        
        return result_df


def main():
    evaluator = SimpleRAGASEvaluator()
    
    test_questions = [
        "开关油压低重合闸闭锁是咋回事？",
        "变压器油温过高应该如何处理？",
        "GIS设备发生SF6气体泄漏应该如何处理？",
        "母线电压异常时应该检查哪些内容？",
        "断路器拒绝合闸的原因有哪些？"
    ]
    
    result = evaluator.evaluate(test_questions)
    evaluator.print_results(result)
    
    print("\n" + "=" * 60)
    print("🎉 评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()
