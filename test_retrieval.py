#!/usr/bin/env python3
"""
简化的RAG检索测试脚本
只测试检索功能，不依赖LLM
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append('/workspace')

from vector_store import VectorStoreManager


def test_retrieval():
    print("="*60)
    print("🧪 RAG检索功能测试")
    print("="*60)
    
    # 初始化向量管理器
    print("\n🔄 正在初始化向量数据库...")
    vector_manager = VectorStoreManager(embedding_model="qianwen")
    
    try:
        vector_manager.load_local()
        print("✅ 向量数据库加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("请先运行 main.py 构建向量数据库")
        return
    
    # 测试问题
    test_questions = [
        "开关油压低重合闸闭锁是咋回事？",
        "变压器油温过高应该如何处理？",
        "GIS设备发生SF6气体泄漏应该如何处理？",
        "母线电压异常时应该检查哪些内容？",
        "断路器拒绝合闸的原因有哪些？"
    ]
    
    print(f"\n📝 准备测试 {len(test_questions)} 个问题...")
    
    # 逐个测试
    for idx, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(test_questions)}] 问题: {question}")
        print(f"{'='*60}")
        
        try:
            # 测试混合检索+重排
            docs = vector_manager.hybrid_search_with_rerank(
                question,
                top_k=8,
                top_n=3
            )
            
            print(f"\n✅ 成功检索到 {len(docs)} 个相关文档:")
            
            for doc_idx, doc in enumerate(docs, 1):
                print(f"\n📄 文档 {doc_idx}:")
                content = doc.page_content
                preview = content[:300] + "..." if len(content) > 300 else content
                print(f"   内容: {preview}")
                if doc.metadata:
                    print(f"   元数据: {doc.metadata}")
        
        except Exception as e:
            print(f"❌ 检索失败: {e}")
    
    print(f"\n{'='*60}")
    print("🎉 检索测试完成！")
    print("="*60)


if __name__ == "__main__":
    test_retrieval()
