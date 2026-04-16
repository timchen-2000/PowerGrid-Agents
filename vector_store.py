import os
from typing import List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


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
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.documents: List[Document] = []

    def create_from_documents(self, documents: List[Document]) -> FAISS:
        # 保存文档列表
        self.documents = documents
        
        # 创建FAISS向量数据库
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # 创建BM25检索器
        try:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            print(f"成功创建BM25检索器，文档数: {len(documents)}")
        except Exception as e:
            print(f"创建BM25检索器失败: {e}")
            self.bm25_retriever = None
        
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
    
    def hybrid_search(self, query: str, k: int = 4, vector_weight: float = 0.5) -> List[Document]:
        """混合检索：结合BM25关键词检索和向量相似度检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            vector_weight: 向量检索权重（0-1），BM25权重为1-vector_weight
            
        Returns:
            混合排序后的文档列表
        """
        if self.vector_store is None:
            raise ValueError("向量数据库未初始化，请先创建或加载向量库")
        
        # 1. 向量相似度检索
        vector_results = self.vector_store.similarity_search_with_score(query, k=k)
        vector_documents = [doc for doc, score in vector_results]
        vector_scores = [score for doc, score in vector_results]
        
        # 2. BM25关键词检索
        bm25_documents = []
        bm25_scores = []
        
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.invoke(query)
            bm25_documents = bm25_results
            # BM25返回的是排序后的结果，我们赋予递减的分数
            bm25_scores = [1.0 / (i + 1) for i in range(len(bm25_results))]
        else:
            # 如果BM25不可用，只使用向量检索
            print("BM25检索器不可用，使用纯向量检索")
            return vector_documents
        
        # 3. 合并结果并去重
        all_documents = []
        seen_content = set()
        
        # 先添加向量检索结果
        for doc in vector_documents:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                all_documents.append(doc)
        
        # 再添加BM25检索结果（排除已存在的）
        for doc in bm25_documents:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                all_documents.append(doc)
        
        # 4. 限制返回数量
        return all_documents[:k]
    
    def rerank(self, query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
        """对检索结果进行重排
        
        Args:
            query: 查询文本
            documents: 待重排的文档列表
            top_n: 重排后返回的文档数量
            
        Returns:
            重排后的文档列表
        """
        if not documents:
            return []
        
        try:
            # 使用更精细的embedding模型进行重排
            # 这里使用BGE模型，效果较好
            rerank_embeddings = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # 计算查询向量
            query_embedding = rerank_embeddings.embed_query(query)
            
            # 计算每个文档与查询的相似度
            doc_embeddings = rerank_embeddings.embed_documents([doc.page_content for doc in documents])
            
            # 计算余弦相似度
            from numpy import dot
            from numpy.linalg import norm
            
            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                similarity = dot(query_embedding, doc_embedding) / (norm(query_embedding) * norm(doc_embedding))
                similarities.append((documents[i], similarity))
            
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前top_n个
            reranked_documents = [doc for doc, _ in similarities[:top_n]]
            print(f"重排完成，从 {len(documents)} 个文档中选择前 {top_n} 个")
            
            return reranked_documents
        except Exception as e:
            print(f"重排失败，使用原始排序: {e}")
            # 失败时返回原始文档的前top_n个
            return documents[:top_n]
    
    def hybrid_search_with_rerank(self, query: str, top_k: int = 8, top_n: int = 3) -> List[Document]:
        """混合检索 + 重排：完整的召回-重排流程
        
        Args:
            query: 查询文本
            top_k: 初始召回数量
            top_n: 重排后返回数量
            
        Returns:
            重排后的文档列表
        """
        # 1. 召回 Top K
        retrieved_docs = self.hybrid_search(query, k=top_k)
        print(f"初始召回 {len(retrieved_docs)} 个文档")
        
        # 2. 重排
        reranked_docs = self.rerank(query, retrieved_docs, top_n=top_n)
        
        # 3. 返回重排后的 Top N
        return reranked_docs
