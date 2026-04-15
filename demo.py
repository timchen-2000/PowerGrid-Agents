import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from qa_agent import PowerEquipmentQAAgent


def initialize_system(pdf_path: str, rebuild: bool = False, embedding_model: str = "qianwen"):
    load_dotenv()
    
    doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    vector_manager = VectorStoreManager(embedding_model=embedding_model)
    
    db_exists = os.path.exists("./faiss_db")
    
    if rebuild or not db_exists:
        print(f"正在处理PDF文档: {pdf_path}")
        documents = doc_processor.process_pdf(pdf_path)
        print(f"文档分割完成，共 {len(documents)} 个片段")
        
        print(f"正在使用 {embedding_model} 模型构建向量数据库...")
        vector_manager.create_from_documents(documents)
        vector_manager.save_local()
        print("向量数据库构建完成并已保存")
    else:
        print("正在加载已有向量数据库...")
        vector_manager.load_local()
        print("向量数据库加载完成")
    
    agent = PowerEquipmentQAAgent(vector_manager, k=3)
    return agent


def demo_knowledge_qa(agent):
    print("\n" + "="*60)
    print("📚 知识问答演示")
    print("="*60)
    
    questions = [
        "变压器有载重瓦斯出口的常见原因是啥",
        "断路器跳闸的常见原因有哪些",
        "电容器组异常的处理方法"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}] 问题: {question}")
        print("正在思考中...")
        result = agent.ask(question)
        
        print("\n回答:")
        print("-"*60)
        print(result["answer"])
        print("-"*60)
        print(f"参考来源: {len(result['source_documents'])} 个文档片段")


def demo_tool_calls(agent):
    print("\n" + "="*60)
    print("🛠️ 工具调用演示")
    print("="*60)
    
    tool_demos = [
        "检查所有设备的状态",
        "查询未处理的告警",
        "获取维护计划",
        "确认告警ID为1的告警"
    ]
    
    for i, demo in enumerate(tool_demos, 1):
        print(f"\n[{i}] {demo}")
        print("正在执行...")
        result = agent.ask(demo)
        
        print("\n结果:")
        print("-"*60)
        print(result["answer"])
        print("-"*60)


def demo_emergency_response(agent):
    print("\n" + "="*60)
    print("🚨 应急处理演示")
    print("="*60)
    
    emergency_questions = [
        "完了完了，妈的六号报警器火灾预警了，一直亮着红灯，咋办咋办",
        "变压器温度过高，达到了85度，怎么办",
        "断路器突然跳闸，是什么原因，怎么处理"
    ]
    
    for i, question in enumerate(emergency_questions, 1):
        print(f"\n[{i}] 紧急情况: {question}")
        print("正在思考中...")
        result = agent.ask(question)
        
        print("\n应急处理方案:")
        print("-"*60)
        print(result["answer"])
        print("-"*60)


def demo_multi_turn_dialogue(agent):
    print("\n" + "="*60)
    print("💬 多轮对话演示")
    print("="*60)
    
    dialogue = [
        "检查变压器#2的状态",
        "它的温度是多少",
        "这样的温度正常吗",
        "如果温度过高，应该怎么处理"
    ]
    
    for i, question in enumerate(dialogue, 1):
        print(f"\n[{i}] 用户: {question}")
        print("正在思考中...")
        result = agent.ask(question)
        
        print("\n系统:")
        print("-"*60)
        print(result["answer"])
        print("-"*60)


def main():
    pdf_path = "变电设备监控信息释义及处置原则.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"错误: PDF文件不存在: {pdf_path}")
        return
    
    print("="*60)
    print("⚡ 电力设备监控智能问答系统 - 功能演示")
    print("="*60)
    
    # 初始化系统
    agent = initialize_system(pdf_path, rebuild=False)
    
    # 运行各个演示模块
    demo_knowledge_qa(agent)
    demo_tool_calls(agent)
    demo_emergency_response(agent)
    demo_multi_turn_dialogue(agent)
    
    print("\n" + "="*60)
    print("🎉 演示完成！")
    print("="*60)
    print("\n项目特色:")
    print("• 基于先进RAG架构的智能问答")
    print("• 混合检索（BM25 + 向量）提高召回率")
    print("• 智能工具调用，实现设备状态检查和告警处理")
    print("• 多轮对话能力，理解上下文")
    print("• 应急处理指导，提供专业解决方案")
    print("• 国产大模型集成，支持中文专业术语")


if __name__ == "__main__":
    main()
