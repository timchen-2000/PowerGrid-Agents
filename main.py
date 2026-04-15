import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from qa_agent import PowerEquipmentQAAgent


def initialize_system(pdf_path: str, rebuild: bool = False, embedding_model: str = "fake", chunk_size: int = 1000, chunk_overlap: int = 200):
    load_dotenv()
    
    doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
    
    agent = PowerEquipmentQAAgent(vector_manager)
    return agent


def interactive_qa(agent):
    print("\n" + "="*50)
    print("电力设备监控智能问答系统")
    print("="*50)
    print("输入 'quit' 或 'exit' 退出")
    print("-"*50)
    
    while True:
        question = input("\n请输入您的问题: ").strip()
        
        if question.lower() in ['quit', 'exit', '退出']:
            print("感谢使用，再见！")
            break
        
        if not question:
            continue
        
        print("\n正在思考中...")
        result = agent.ask(question)
        
        print("\n" + "="*50)
        print("回答:")
        print("-"*50)
        print(result["answer"])
        print("="*50)
        
        print(f"\n参考来源: {len(result['source_documents'])} 个文档片段")


def main():
    pdf_path = "变电设备监控信息释义及处置原则.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"错误: PDF文件不存在: {pdf_path}")
        return
    
    # 配置选项
    embedding_model = "qianwen"  # 选择embedding模型: fake, openai, huggingface, qianwen
    chunk_size = 1000  # 文本块大小（字符数）
    chunk_overlap = 200  # 文本块重叠大小（字符数）
    
    print(f"="*50)
    print("系统配置:")
    print(f"  - Embedding模型: {embedding_model}")
    print(f"  - Chunk大小: {chunk_size}字符")
    print(f"  - Chunk重叠: {chunk_overlap}字符")
    print(f"="*50)
    
    agent = initialize_system(
        pdf_path, 
        rebuild=True, 
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # 自动测试模式，模拟用户交互
    print("\n" + "="*50)
    print("电力设备监控智能问答系统 - Agent工具链演示")
    print("="*50)
    print("正在演示几个典型问题的回答...")
    print("-"*50)
    
    # 测试问题列表 - 包含知识问答和工具调用
    test_questions = [
        "什么是电力设备监控？",
        "检查一下变压器#2的状态怎么样",
        "有哪些未处理的告警？",
        "最近有什么维护计划？",
        "帮我确认一下ID为1的告警"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("正在思考中...")
        result = agent.ask(question)
        
        print("\n" + "="*50)
        print("回答:")
        print("-"*50)
        print(result["answer"])
        print("="*50)
        
        print(f"\n参考来源: {len(result['source_documents'])} 个文档片段")
        print("-"*50)
    
    print("\n" + "="*50)
    print("多轮对话演示")
    print("="*50)
    
    # 多轮对话演示
    print("\n问题1: 变压器#2的温度是多少？")
    result1 = agent.ask("变压器#2的温度是多少？")
    print("\n回答: " + result1["answer"])
    
    print("\n问题2: 那它有告警吗？")
    result2 = agent.ask("那它有告警吗？")
    print("\n回答: " + result2["answer"])
    
    print("\n多轮对话演示完成！")
    print("\nAgent工具链演示完成！")


if __name__ == "__main__":
    main()
