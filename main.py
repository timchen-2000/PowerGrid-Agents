import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from qa_agent import PowerEquipmentQAAgent


def initialize_system(pdf_path: str, rebuild: bool = False, embedding_model: str = "fake", chunk_size: int = 1000, chunk_overlap: int = 200, search_k: int = 4):
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
    
    agent = PowerEquipmentQAAgent(vector_manager, k=search_k)
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
    
    # 配置选项 - 优化检索效果
    embedding_model = "qianwen"  # 选择embedding模型: fake, openai, huggingface, qianwen
    chunk_size = 500  # 减小chunk大小，提高检索精度
    chunk_overlap = 100  # 调整重叠大小
    search_k = 8  # 增加检索结果数量
    
    print(f"="*50)
    print("系统配置:")
    print(f"  - Embedding模型: {embedding_model}")
    print(f"  - Chunk大小: {chunk_size}字符")
    print(f"  - Chunk重叠: {chunk_overlap}字符")
    print(f"  - 检索结果数: {search_k}")
    print(f"="*50)
    
    agent = initialize_system(
        pdf_path, 
        rebuild=True,  # 重新构建向量库，使用新的chunk配置
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        search_k=search_k
    )
    
    # 测试特定问题
    print("\n" + "="*50)
    print("电力设备监控智能问答系统 - 知识问答")
    print("="*50)
    
    question = "双鉴电源坏了咋办"
    print(f"\n问题: {question}")
    print("正在思考中...")
    result = agent.ask(question)
    
    print("\n" + "="*50)
    print("回答:")
    print("-"*50)
    print(result["answer"])
    print("="*50)
    
    print(f"\n参考来源: {len(result['source_documents'])} 个文档片段")
    if result["source_documents"]:
        print("-"*50)
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"\n参考片段 {i}:")
            print(doc.page_content)
    
    print("\n" + "="*50)
    print("知识问答完成！")


if __name__ == "__main__":
    main()
