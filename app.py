from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

sys.path.append('/workspace')
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from power_equipment_tools import get_tools
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional, Dict, List, Any

# 初始化FastAPI应用
app = FastAPI(
    title="电力设备监控智能问答系统",
    description="基于RAG架构的电力设备监控专业AI助手，集成混合检索、智能工具调用和多轮对话能力",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局组件实例
vector_manager = None
chat_history = []
tools = []
llm = None

# 启动时初始化
@app.on_event("startup")
async def startup_event():
    global vector_manager, tools, llm
    try:
        print("正在初始化系统...")
        
        # 初始化文档处理器和向量管理器
        doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        vector_manager = VectorStoreManager(embedding_model="qianwen")
        
        # 加载或创建向量数据库
        try:
            vector_manager.load_local()
            print("向量数据库加载完成")
        except FileNotFoundError:
            print("向量数据库不存在，正在构建...")
            documents = doc_processor.process_pdf("变电设备监控信息释义及处置原则.pdf")
            vector_manager.create_from_documents(documents)
            vector_manager.save_local()
            print("向量数据库构建完成")
        
        # 初始化工具和LLM
        tools = get_tools()
        llm = ChatOpenAI(model_name="deepseek-chat", temperature=0.1)
        
        print("系统初始化完成，服务已就绪")
    except Exception as e:
        print(f"初始化失败: {e}")

# 请求模型
class QuestionRequest(BaseModel):
    question: str
    conversation_id: str = "default"

# 响应模型
class AnswerResponse(BaseModel):
    answer: str
    source_documents: list
    conversation_id: str
    status: str

# 构建提示模板
def get_prompt():
    return ChatPromptTemplate.from_messages([
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
    ])

# 主要问答接口
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    global vector_manager, tools, llm, chat_history
    
    if not vector_manager or not llm:
        raise HTTPException(status_code=503, detail="系统未初始化")
    
    try:
        # 执行混合检索和重排
        source_documents = []
        try:
            # 先召回更多文档，再重排
            top_k = 8  # 初始召回数量
            top_n = 3  # 最终返回数量
            source_documents = vector_manager.hybrid_search_with_rerank(request.question, top_k=top_k, top_n=top_n)
            print(f"召回-重排流程完成，最终选择 {len(source_documents)} 个相关文档")
        except Exception as e:
            print(f"召回-重排流程失败，使用混合检索: {e}")
            try:
                source_documents = vector_manager.hybrid_search(request.question, k=3)
                print(f"混合检索完成，找到 {len(source_documents)} 个相关文档")
            except Exception as e2:
                print(f"混合检索失败，使用纯向量检索: {e2}")
                try:
                    source_documents = vector_manager.similarity_search(request.question, k=3)
                except:
                    pass
        
        # 构建上下文
        context = ""
        if source_documents:
            context = "以下是相关的知识库信息：\n"
            for i, doc in enumerate(source_documents, 1):
                context += f"\n【文档片段{i}】\n{doc.page_content}\n"
        
        # 构建输入
        input_text = request.question + ("\n\n" + context if context else "")
        
        # 生成回答
        prompt = get_prompt()
        messages = prompt.format_messages(
            input=input_text,
            chat_history=chat_history
        )
        
        # 调用LLM
        response = llm.invoke(messages)
        answer = response.content
        
        # 更新对话历史
        chat_history.append(HumanMessage(content=request.question))
        chat_history.append(AIMessage(content=answer))
        
        # 限制对话历史长度
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        
        return AnswerResponse(
            answer=answer,
            source_documents=[
                {
                    "content": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                    "metadata": doc.metadata
                }
                for doc in source_documents
            ],
            conversation_id=request.conversation_id,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

# 健康检查接口
@app.get("/health")
async def health_check():
    status = "healthy" if vector_manager else "unhealthy"
    return {
        "status": status,
        "service": "power-equipment-agent",
        "system_initialized": vector_manager is not None
    }

# 系统信息接口
@app.get("/info")
async def system_info():
    return {
        "name": "电力设备监控智能问答系统",
        "version": "1.0.0",
        "features": [
            "基于RAG架构的智能问答",
            "混合检索（BM25 + 向量）",
            "智能工具调用",
            "多轮对话能力",
            "应急处理指导",
            "国产大模型集成"
        ],
        "system_initialized": vector_manager is not None
    }

# 清空对话历史接口
@app.post("/clear_history")
async def clear_history():
    global chat_history
    chat_history = []
    return {"status": "success", "message": "对话历史已清空"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
