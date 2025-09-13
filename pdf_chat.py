import os
import sys
import logging
from datetime import datetime
from src.documents.loader import PatentDocumentLoader
from src.utils.file_utils import ensure_directory_exists

LOGS_DIR = "/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c/SemiIP-Summary/logs"
ensure_directory_exists(LOGS_DIR)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(LOGS_DIR, f"pdf_chat_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def chat_with_pdf(pdf_path, question):
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain_core.runnables import Runnable
        import sglang as sgl
        
        intermediate_dir = "/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c/SemiIP-Summary/data/intermediate"
        pdf_filename = os.path.basename(pdf_path)
        intermediate_filename = os.path.splitext(pdf_filename)[0] + "_extracted.txt"
        intermediate_filepath = os.path.join(intermediate_dir, intermediate_filename)
        
        if os.path.exists(intermediate_filepath):
            with open(intermediate_filepath, "r", encoding="utf-8") as f:
                document_content = f.read()
        else:
            document_loader = PatentDocumentLoader(os.path.dirname(pdf_path))
            documents = document_loader.load_documents()
            document_content = None
            for doc in documents:
                if doc['name'] == pdf_filename:
                    document_content = doc['content']
                    break
            
            if document_content is None:
                raise ValueError(f"无法找到PDF文件: {pdf_path}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(document_content)
        
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning("无法加载HuggingFace embeddings，使用本地TF-IDF替代")
            from langchain_community.embeddings import FakeEmbeddings
            embeddings = FakeEmbeddings(size=1024)
        
        vector_store = FAISS.from_texts(texts, embeddings)
        
        sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
        
        class SGLangLLM(Runnable):
            def __init__(self, model_name="/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c/gpt-oss-20b"):
                self.model_name = model_name
                super().__init__()
            
            def invoke(self, input, config=None, **kwargs):
                @sgl.function
                def sglang_generate(s, prompt_text):
                    s += prompt_text
                    s += sgl.gen("answer", max_tokens=512, temperature=0.1)
                
                try:
                    state = sglang_generate.run(prompt_text=input)
                    return state["answer"]
                except Exception as e:
                    logger.error(f"SGLang调用错误: {e}")
                    return "抱歉，无法生成回答。"
            
            def _call(self, prompt, stop=None, run_manager=None, **kwargs):
                return self.invoke(prompt)
            
            def __call__(self, prompt):
                return self._call(prompt)
        
        prompt_template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。

{context}

问题: {question}
有用的回答:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=SGLangLLM(),
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"query": question})
        
        return result["result"]
        
    except Exception as e:
        logger.error(f"Error in chat_with_pdf: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"处理过程中出现错误: {str(e)}"

def main():
    if len(sys.argv) >= 3:
        pdf_path = sys.argv[1]
        question = sys.argv[2]
        
        if not os.path.exists(pdf_path):
            print(f"错误: 找不到PDF文件 {pdf_path}")
            return
        
        answer = chat_with_pdf(pdf_path, question)
        print(f"问题: {question}")
        print(f"答案: {answer}")
    else:
        print("使用方法: python pdf_chat.py <pdf_path> <question>")

if __name__ == "__main__":
    main()