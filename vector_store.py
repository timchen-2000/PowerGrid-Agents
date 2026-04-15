import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from openai import OpenAI


class QwenEmbeddings(Embeddings):
    """直接使用千问API的embedding类"""
    
    def __init__(self, model: str = "text-embedding-v4", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对文档列表进行向量化"""
        embeddings = []
        for text in texts:
            # 确保文本是字符串
            clean_text = str(text)
            # 单个文本调用，避免批处理格式问题
            response = self.client.embeddings.create(
                model=self.model,
                input=clean_text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """对查询文本进行向量化"""
        response = self.client.embeddings.create(
            model=self.model,
            input=str(text)
        )
        return response.data[0].embedding


class VectorStoreManager:
    def __init__(self, persist_directory: str = "./faiss_db", embedding_model: str = "fake"):
        self.persist_directory = persist_directory
        
        # 初始化不同的embedding模型
        if embedding_model == "openai":
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings()
        elif embedding_model == "huggingface":
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
            except Exception as e:
                print(f"无法加载HuggingFace模型: {e}")
                print("切换到FakeEmbeddings进行演示")
                self.embeddings = FakeEmbeddings(size=128)
        elif embedding_model == "qianwen":
            try:
                # 使用千问的embedding模型（阿里云百炼API）
                api_key = os.getenv("DASHSCOPE_API_KEY")
                if api_key:
                    self.embeddings = QwenEmbeddings(
                        model="text-embedding-v4",
                        api_key=api_key,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                    )
                    print("成功初始化千问embedding模型（阿里云百炼API）")
                else:
                    raise ValueError("千问API密钥未配置，请设置DASHSCOPE_API_KEY环境变量")
            except Exception as e:
                print(f"无法加载千问模型: {e}")
                print("切换到FakeEmbeddings进行演示")
                self.embeddings = FakeEmbeddings(size=128)
        else:
            # 默认使用FakeEmbeddings用于演示
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
