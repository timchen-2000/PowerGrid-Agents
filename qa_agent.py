from typing import Optional, Dict, List
from langchain_core.documents import Document
from vector_store import VectorStoreManager


class PowerEquipmentQAAgent:
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        k: int = 4
    ):
        self.vector_store_manager = vector_store_manager
        self.k = k
    
    def ask(self, question: str) -> dict:
        # 简单的回答逻辑，避免使用LLM
        # 实际应用中，这里会使用更强大的语言模型来生成专业的回答
        
        # 检索相关文档
        source_documents = self.vector_store_manager.similarity_search(question, k=self.k)
        
        # 构建回答
        answer = "这是一个基于本地模型的回答。在实际应用中，这里会使用更强大的语言模型来生成专业的电力设备监控领域的回答。"
        
        return {
            "question": question,
            "answer": answer,
            "source_documents": source_documents
        }
    
    def get_simple_answer(self, question: str) -> str:
        result = self.ask(question)
        return result["answer"]
