#!/usr/bin/env python3
"""
RAG系统基础评估脚本
提供RAG流程测试、检索质量、生成质量的基础评估功能
"""

import os
import sys
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import pandas as pd

# 加载环境变量
load_dotenv()

sys.path.append('/workspace')

from vector_store import VectorStoreManager
from app import get_prompt
from langchain_openai import ChatOpenAI


class RAGEvaluator:
    def __init__(self):
        print("=" * 60)
        print("🔄 正在初始化RAG评估器...")
        print("=" * 60)
        
        # 初始化向量管理器
        self.vector_manager = VectorStoreManager(embedding_model="qianwen")
        
        # 加载向量数据库
        try:
            self.vector_manager.load_local()
            print("✅ 向量数据库加载成功")
        except FileNotFoundError:
            print("❌ 向量数据库不存在，正在构建...")
            from document_processor import DocumentProcessor
            doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
            documents = doc_processor.process_pdf("变电设备监控信息释义及处置原则.pdf")
            self.vector_manager.create_from_documents(documents)
            self.vector_manager.save_local()
            print("✅ 向量数据库构建完成")
        
        # 初始化LLM
        self.llm = ChatOpenAI(model_name="deepseek-chat", temperature=0.1)
        print("✅ LLM初始化成功")
        
        # 评估数据存储
        self.test_data = []
        self.results_df = None
    
    def add_test_case(self, question: str, ground_truth: str):
        """添加测试用例"""
        self.test_data.append({
            "question": question,
            "ground_truth": ground_truth
        })
    
    def retrieve_context(self, question: str, top_k: int = 8, top_n: int = 3) -> List[str]:
        """检索相关上下文"""
        try:
            # 使用混合检索+重排
            docs = self.vector_manager.hybrid_search_with_rerank(
                question, 
                top_k=top_k, 
                top_n=top_n
            )
            context_list = [doc.page_content for doc in docs]
            return context_list
        except Exception as e:
            print(f"检索失败: {e}")
            return []
    
    def generate_answer(self, question: str, context: List[str]) -> str:
        """生成答案"""
        if not context:
            return "抱歉，没有找到相关的知识库内容来回答您的问题。"
        
        # 构建上下文字符串
        context_str = "以下是相关的知识库内容：\n"
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
    
    def evaluate_single(self, question: str, ground_truth: str) -> Dict:
        """评估单个问题"""
        print(f"\n{'='*60}")
        print(f"📝 评估问题: {question}")
        print(f"{'='*60}")
        
        # 执行RAG流程
        context = self.retrieve_context(question)
        answer = self.generate_answer(question, context)
        
        # 打印结果
        print(f"\n📚 检索到的上下文 (共{len(context)}个):")
        for i, ctx in enumerate(context, 1):
            preview = ctx[:150] + "..." if len(ctx) > 150 else ctx
            print(f"\n  [{i}] {preview}")
        
        print(f"\n🤖 生成的答案:\n{answer}")
        print(f"\n📋 标准答案:\n{ground_truth}")
        
        # 计算基础指标
        metrics = self._calculate_basic_metrics(answer, ground_truth, context)
        
        result = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "context": context,
            "context_count": len(context),
            "answer_length": len(answer),
            **metrics
        }
        
        return result
    
    def _calculate_basic_metrics(self, answer: str, ground_truth: str, context: List[str]) -> Dict:
        """计算基础评估指标"""
        metrics = {}
        
        # 答案长度比较
        metrics["answer_length_ratio"] = len(answer) / max(len(ground_truth), 1)
        
        # 关键词重叠度
        from collections import Counter
        import re
        
        # 提取答案和标准答案中的关键词
        def extract_keywords(text):
            words = re.findall(r'[\u4e00-\u9fa5]+', text)  # 提取中文
            return set(words)
        
        answer_keywords = extract_keywords(answer)
        gt_keywords = extract_keywords(ground_truth)
        
        if gt_keywords:
            overlap = len(answer_keywords & gt_keywords) / len(gt_keywords)
            metrics["keyword_overlap_ratio"] = overlap
        else:
            metrics["keyword_overlap_ratio"] = 0
        
        # 上下文包含度：检查答案是否包含检索到的上下文
        context_str = " ".join(context)
        context_mentions = sum(1 for word in extract_keywords(answer) if word in context_str)
        metrics["context_coverage_score"] = min(context_mentions / max(len(answer_keywords), 1), 1.0)
        
        return metrics
    
    def run_evaluation(self) -> pd.DataFrame:
        """运行完整评估"""
        print(f"\n{'#'*60}")
        print("# 🧪 开始完整评估")
        print(f"{'#'*60}")
        
        all_results = []
        
        for idx, test_case in enumerate(self.test_data, 1):
            print(f"\n[进度 {idx}/{len(self.test_data)}]")
            result = self.evaluate_single(test_case["question"], test_case["ground_truth"])
            all_results.append(result)
        
        # 创建结果DataFrame
        self.results_df = pd.DataFrame(all_results)
        
        # 打印汇总统计
        self._print_summary_stats()
        
        return self.results_df
    
    def _print_summary_stats(self):
        """打印汇总统计信息"""
        print(f"\n{'#'*60}")
        print("# 📊 评估汇总报告")
        print(f"{'#'*60}")
        
        df = self.results_df
        
        print(f"\n📈 总体统计:")
        print(f"  - 总测试用例数: {len(df)}")
        print(f"  - 平均检索文档数: {df['context_count'].mean():.2f}")
        print(f"  - 平均答案长度: {df['answer_length'].mean():.0f} 字符")
        print(f"  - 平均关键词重叠度: {df['keyword_overlap_ratio'].mean():.2%}")
        print(f"  - 平均上下文覆盖度: {df['context_coverage_score'].mean():.2%}")
        
        # 最高分/最低分
        if "keyword_overlap_ratio" in df.columns:
            best_idx = df["keyword_overlap_ratio"].idxmax()
            worst_idx = df["keyword_overlap_ratio"].idxmin()
            
            print(f"\n🏆 最佳案例 (关键词重叠度 {df.loc[best_idx, 'keyword_overlap_ratio']:.2%}):")
            print(f"  问题: {df.loc[best_idx, 'question']}")
            
            print(f"\n⚠️ 需要改进 (关键词重叠度 {df.loc[worst_idx, 'keyword_overlap_ratio']:.2%}):")
            print(f"  问题: {df.loc[worst_idx, 'question']}")
    
    def save_results(self, filename: str = "rag_evaluation_results.json"):
        """保存评估结果"""
        if self.results_df is not None:
            # 保存为JSON（保留完整数据）
            results_dict = self.results_df.to_dict(orient="records")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 评估结果已保存为JSON: {filename}")
            
            # 同时保存为CSV
            csv_filename = filename.replace(".json", ".csv")
            self.results_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
            print(f"✅ 评估结果已保存为CSV: {csv_filename}")
            

def main():
    """主函数"""
    evaluator = RAGEvaluator()
    
    # 添加测试用例
    print("\n📝 准备测试用例...")
    
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
        print(f"  ✅ 已添加: {case['question'][:40]}...")
    
    # 运行评估
    results_df = evaluator.run_evaluation()
    
    # 保存结果
    evaluator.save_results("rag_evaluation_results.json")
    
    print(f"\n{'='*60}")
    print("🎉 RAG评估完成！")
    print(f"{'='*60}")
    
    return results_df


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
