from typing import Optional, Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from vector_store import VectorStoreManager
from power_equipment_tools import get_tools


class PowerEquipmentQAAgent:
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = "deepseek-chat",
        temperature: float = 0.1,
        k: int = 4
    ):
        self.vector_store_manager = vector_store_manager
        self.k = k
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model_name=model_name, 
            temperature=temperature
        )
        
        # 获取电力设备监控专用工具
        self.tools = get_tools()
        
        # 初始化对话历史
        self.chat_history: List[Any] = []
        
        # 创建Agent
        self.agent = self._build_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _build_agent(self):
        """构建Agent，包含RAG和工具调用能力"""
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位电力设备监控领域的专业专家助手。你的职责是帮助用户解决电力设备监控相关的问题。

你可以使用以下能力：
1. **知识检索**：基于电力设备监控知识库回答专业问题
2. **设备状态检查**：查询电力设备的实时运行状态
3. **告警查询**：查看和筛选设备告警信息
4. **维护计划管理**：查看、创建和管理设备维护计划

当用户提问时，请遵循以下步骤：
1. 首先判断是否需要使用工具，还是可以直接回答
2. 如果需要检查设备状态、查询告警或管理维护计划，请使用相应的工具
3. 如果是专业知识问题，可以基于知识库回答
4. 可以结合多种能力，提供全面的解决方案

回答要求：
1. 基于提供的信息回答，不要编造内容
2. 回答要专业、准确、详细
3. 使用中文回答
4. 保持友好、专业的态度"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建OpenAI Tools Agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return agent
    
    def ask(self, question: str) -> dict:
        """
        处理用户问题，支持多轮对话和工具调用
        
        Args:
            question: 用户问题
            
        Returns:
            包含回答和相关信息的字典
        """
        try:
            # 调用Agent执行器
            result = self.agent_executor.invoke({
                "input": question,
                "chat_history": self.chat_history
            })
            
            # 更新对话历史
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=result["output"]))
            
            # 尝试获取相关文档（如果Agent没有使用知识库工具）
            source_documents = []
            try:
                source_documents = self.vector_store_manager.similarity_search(question, k=self.k)
            except:
                pass
            
            return {
                "question": question,
                "answer": result["output"],
                "source_documents": source_documents
            }
        except Exception as e:
            # 错误处理
            error_msg = f"处理问题时出错: {str(e)}"
            return {
                "question": question,
                "answer": error_msg,
                "source_documents": []
            }
    
    def clear_chat_history(self):
        """清空对话历史"""
        self.chat_history = []
    
    def get_simple_answer(self, question: str) -> str:
        """获取简单的回答文本"""
        result = self.ask(question)
        return result["answer"]
