from typing import Optional, Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool
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
        
        # 创建Agent
        system_prompt = "你是一位电力设备监控领域的专业专家助手。你的职责是帮助用户解决电力设备监控相关的问题。\n\n你可以使用以下能力：\n1. **知识检索**：基于电力设备监控知识库回答专业问题\n2. **设备状态检查**：查询电力设备的实时运行状态\n3. **告警查询**：查看和筛选设备告警信息\n4. **维护计划管理**：查看、创建和管理设备维护计划\n\n当用户提问时，请遵循以下步骤：\n1. 首先判断是否需要使用工具，还是可以直接回答\n2. 如果需要检查设备状态、查询告警或管理维护计划，请使用相应的工具\n3. 如果是专业知识问题，可以基于知识库回答\n4. 可以结合多种能力，提供全面的解决方案\n\n回答要求：\n1. 基于提供的信息回答，不要编造内容\n2. 回答要专业、准确、详细\n3. 使用中文回答\n4. 保持友好、专业的态度"
        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
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
            # 使用完整的召回-重排流程：先召回Top K → 再Rerank → 再取重排后的Top N
            source_documents = []
            try:
                # 先召回更多文档，再重排
                top_k = 8  # 初始召回数量
                top_n = self.k  # 最终返回数量
                source_documents = self.vector_store_manager.hybrid_search_with_rerank(question, top_k=top_k, top_n=top_n)
                print(f"召回-重排流程完成，最终选择 {len(source_documents)} 个相关文档")
            except Exception as e:
                print(f"召回-重排流程失败，使用混合检索: {e}")
                try:
                    source_documents = self.vector_store_manager.hybrid_search(question, k=self.k)
                    print(f"混合检索完成，找到 {len(source_documents)} 个相关文档")
                except Exception as e2:
                    print(f"混合检索失败，使用纯向量检索: {e2}")
                    try:
                        source_documents = self.vector_store_manager.similarity_search(question, k=self.k)
                    except:
                        pass
            
            # 如果有相关文档，将其内容加入到上下文中
            context = ""
            if source_documents:
                context = "以下是相关的知识库信息：\n"
                for i, doc in enumerate(source_documents, 1):
                    context += f"\n【文档片段{i}】\n{doc.page_content}\n"
            
            # 构建输入消息
            messages = []
            for msg in self.chat_history:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
            
            # 添加当前问题
            messages.append({"role": "user", "content": question + ("\n\n" + context if context else "")})
            
            # 调用Agent
            inputs = {"messages": messages}
            result = self.agent.invoke(inputs)
            
            # 获取回答
            answer = ""
            if result and "messages" in result:
                for msg in result["messages"]:
                    if isinstance(msg, AIMessage) or (isinstance(msg, dict) and msg.get("role") == "assistant"):
                        answer = msg.content if isinstance(msg, AIMessage) else msg.get("content", "")
                        break
            
            # 更新对话历史
            self.chat_history.append(HumanMessage(content=question))
            if answer:
                self.chat_history.append(AIMessage(content=answer))
            
            return {
                "question": question,
                "answer": answer,
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
