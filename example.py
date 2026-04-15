import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from qa_agent import PowerEquipmentQAAgent


def example_usage():
    load_dotenv()
    
    pdf_path = "变电设备监控信息释义及处置原则.pdf"
    
    print("步骤1: 处理文档并构建向量数据库")
    doc_processor = DocumentProcessor()
    documents = doc_processor.process_pdf(pdf_path)
    print(f"文档分割完成，共 {len(documents)} 个片段\n")
    
    print("步骤2: 创建FAISS向量数据库")
    vector_manager = VectorStoreManager()
    vector_manager.create_from_documents(documents)
    vector_manager.save_local()
    print("向量数据库已保存\n")
    
    print("步骤3: 初始化问答Agent")
    agent = PowerEquipmentQAAgent(vector_manager)
    print("Agent初始化完成\n")
    
    print("步骤4: 测试问答功能")
    test_questions = [
        "变压器的常见故障有哪些？",
        "断路器异常时应该如何处置？",
        "请介绍一下变电设备监控的基本原则"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        answer = agent.get_simple_answer(question)
        print(f"回答: {answer}")


if __name__ == "__main__":
    example_usage()
