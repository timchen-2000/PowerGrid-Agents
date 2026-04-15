import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from qa_agent import PowerEquipmentQAAgent


def initialize_system(pdf_path: str, rebuild: bool = False):
    load_dotenv()
    
    doc_processor = DocumentProcessor()
    vector_manager = VectorStoreManager()
    
    db_exists = os.path.exists("./faiss_db")
    
    if rebuild or not db_exists:
        print(f"正在处理PDF文档: {pdf_path}")
        documents = doc_processor.process_pdf(pdf_path)
        print(f"文档分割完成，共 {len(documents)} 个片段")
        
        print("正在构建向量数据库...")
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
    
    agent = initialize_system(pdf_path, rebuild=False)
    interactive_qa(agent)


if __name__ == "__main__":
    main()
