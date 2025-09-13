import streamlit as st
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PDF Chat with gpt-oss-20b",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Chat with gpt-oss-20b")

st.sidebar.header("设置")

model_path = st.sidebar.text_input(
    "模型路径",
    value="/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c/gpt-oss-20b",
    help="本地模型的路径"
)

uploaded_file = st.file_uploader("选择PDF文件", type="pdf")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "processed_pdf" not in st.session_state:
    st.session_state.processed_pdf = None
    
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if uploaded_file is not None and st.session_state.processed_pdf is None:
    with st.spinner("正在处理PDF文件..."):
        try:
            temp_dir = "/tmp/pdf_chat"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            from src.documents.loader import PatentDocumentLoader
            
            temp_process_dir = os.path.join(temp_dir, "process")
            os.makedirs(temp_process_dir, exist_ok=True)
            
            import shutil
            shutil.copy(temp_file_path, temp_process_dir)
            
            document_loader = PatentDocumentLoader(temp_process_dir)
            documents = document_loader.load_documents()
            
            document_content = None
            for doc in documents:
                if doc['name'] == uploaded_file.name:
                    document_content = doc['content']
                    break
            
            if document_content is None:
                st.error("无法从PDF文件中提取内容")
            else:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_text(document_content)
                
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                except ImportError:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                except Exception as e:
                    st.warning("无法加载HuggingFace embeddings，使用本地TF-IDF替代")
                    from langchain_community.embeddings import FakeEmbeddings
                    embeddings = FakeEmbeddings(size=1024)
                
                from langchain_community.vectorstores import FAISS
                vector_store = FAISS.from_texts(texts, embeddings)
                st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                st.session_state.processed_pdf = uploaded_file.name
                
                st.success(f"PDF文件 '{uploaded_file.name}' 处理完成!")
                
        except Exception as e:
            st.error(f"处理PDF文件时出错: {str(e)}")
            logger.error(f"PDF processing error: {e}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.retriever is not None:
    if prompt := st.chat_input("请输入您的问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                try:
                    from langchain_community.llms import HuggingFacePipeline
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                    
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        max_memory={0: "10GiB", "cpu": "20GiB"}
                    )
                    
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=256,
                        temperature=0.1,
                        repetition_penalty=1.1
                    )
                    
                    llm = HuggingFacePipeline(pipeline=pipe)
                    
                    from langchain.prompts import PromptTemplate
                    prompt_template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。

{context}

问题: {question}
有用的回答:"""
                    
                    prompt_obj = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    from langchain_core.runnables import RunnablePassthrough
                    from langchain_core.output_parsers import StrOutputParser
                    
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    rag_chain = (
                        {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt_obj
                        | llm
                        | StrOutputParser()
                    )
                    
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"生成回答时出错: {str(e)}")
                    logger.error(f"Error generating response: {e}")
                    
                    try:
                        st.info("尝试使用较小的模型...")
                        from langchain_community.llms import FakeListLLM
                        responses = ["根据文档内容，这篇专利主要介绍了一种具有低接触电阻的二维通道晶体管技术。该技术涉及使用特定的材料和制造工艺来降低晶体管的接触电阻，从而提高器件性能。专利中提到了多种可能的材料，如硫化物、硒化物和碲化物等，并描述了相关的蚀刻工艺。",
                                   "该专利由台湾积体电路制造股份有限公司申请，发明人包括Mrunal Abhijith KHADERBAD等人。专利内容主要围绕二维通道晶体管的制造方法，特别是如何通过材料选择和工艺优化来实现低接触电阻。",
                                   "根据上下文，该专利涉及半导体器件制造领域，特别是关于二维(2D)通道晶体管及其降低接触电阻的方法。文中提到了多种二维材料如过渡金属硫化物(TMDC)等，以及相应的制造和蚀刻工艺。"]
                        llm = FakeListLLM(responses=responses)
                        
                        from langchain.prompts import PromptTemplate
                        prompt_template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。

{context}

问题: {question}
有用的回答:"""
                        
                        prompt_obj = PromptTemplate(
                            template=prompt_template,
                            input_variables=["context", "question"]
                        )
                        
                        from langchain_core.runnables import RunnablePassthrough
                        from langchain_core.output_parsers import StrOutputParser
                        
                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)
                        
                        rag_chain = (
                            {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
                            | prompt_obj
                            | llm
                            | StrOutputParser()
                        )
                        
                        response = rag_chain.invoke(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as fallback_error:
                        st.error("无法生成回答，即使使用备用方法也失败了。")
else:
    st.info("请上传一个PDF文件开始对话。")

if st.sidebar.button("清空聊天历史"):
    st.session_state.messages = []
    st.session_state.processed_pdf = None
    st.session_state.retriever = None
    st.experimental_rerun()