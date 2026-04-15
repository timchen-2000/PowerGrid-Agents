from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_store import VectorStoreManager


class PowerEquipmentQAAgent:
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = "deepseek-chat",
        temperature: float = 0.1,
        k: int = 4
    ):
        self.vector_store_manager = vector_store_manager
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.k = k
        self.qa_chain = self._build_qa_chain()

    def _build_qa_chain(self) -> RetrievalQA:
        prompt_template = """你是一位电力设备监控领域的专业专家。请根据提供的上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：{question}

回答要求：
1. 基于提供的上下文信息回答，不要编造内容
2. 如果上下文中没有相关信息，请明确告知
3. 回答要专业、准确、详细
4. 使用中文回答

专业回答："""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}

        retriever = self.vector_store_manager.get_retriever(k=self.k)

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

        return qa_chain

    def ask(self, question: str) -> dict:
        result = self.qa_chain.invoke({"query": question})
        return {
            "question": question,
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }

    def get_simple_answer(self, question: str) -> str:
        result = self.ask(question)
        return result["answer"]
