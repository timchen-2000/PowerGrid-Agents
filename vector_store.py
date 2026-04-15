import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


class VectorStoreManager:
    def __init__(self, persist_directory: str = "./faiss_db"):
        self.persist_directory = persist_directory
        # 注意：这里使用FakeEmbeddings用于演示
        # 在实际使用时，请改为真实的embedding模型，例如：
        # - OpenAIEmbeddings() (需要OpenAI API密钥)
        # - HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese") (需要网络连接)
        self.embeddings = FakeEmbeddings(size=128)
        self.vector_store: Optional[FAISS] = None

    def create_from_documents(self, documents: List[Document]) -> FAISS:
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return self.vector_store

    def save_local(self):
        if self.vector_store is None:
            raise ValueError("向量数据库未初始化，请先创建或加载向量库")
        
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store.save_local(self.persist_directory)

    def load_local(self) -> FAISS:
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"向量数据库目录不存在: {self.persist_directory}")
        
        self.vector_store = FAISS.load_local(
            self.persist_directory, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vector_store

    def get_retriever(self, k: int = 4) -> VectorStoreRetriever:
        if self.vector_store is None:
            raise ValueError("向量数据库未初始化，请先创建或加载向量库")
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.vector_store is None:
            raise ValueError("向量数据库未初始化，请先创建或加载向量库")
        
        return self.vector_store.similarity_search(query, k=k)
