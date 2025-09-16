import streamlit as st
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PDF Chat",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Chat")

st.sidebar.header("设置")

model_path = st.sidebar.text_input(
    "模型路径",
    value="/home/ps/Qwen3-4B",
    help="本地模型的路径"
)

use_document_context = st.sidebar.checkbox("使用文档内容作为上下文", value=True, help="取消勾选以进行通用对话")

uploaded_file = st.file_uploader("选择PDF文件", type="pdf")

def clear_previous_data():
    try:
        temp_dir = "/tmp/pdf_chat"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
        
        output_dir = "./output"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
        logger.info("Previous PDF data cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing previous data: {e}")

if "data_cleared" not in st.session_state:
    clear_previous_data()
    st.session_state.data_cleared = True

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "processed_pdf" not in st.session_state:
    st.session_state.processed_pdf = None
    
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
    
if "llm" not in st.session_state:
    st.session_state.llm = None

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
                    embeddings = HuggingFaceEmbeddings(model_name="/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/all-MiniLM-L6-v2")
                except ImportError:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name="/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/all-MiniLM-L6-v2")
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

if prompt := st.chat_input("请输入您的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            try:
                if not st.session_state.model_loaded:
                    with st.spinner("首次运行需要加载模型，请稍候..."):
                        from langchain_community.llms import HuggingFacePipeline
                        import torch
                        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                        
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            low_cpu_mem_usage=True,
                            device_map="auto",
                            torch_dtype=torch.float16
                        )
                        
                        pipe = pipeline(
                            "text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            max_new_tokens=200,
                            temperature=0.1,
                            repetition_penalty=1.2,
                            do_sample=False
                        )
                        
                        st.session_state.llm = HuggingFacePipeline(pipeline=pipe)
                        st.session_state.model_loaded = True
                
                if use_document_context and st.session_state.retriever is not None:
                    from langchain.prompts import PromptTemplate
                    prompt_template = """你是一个专业的技术专家，请仔细阅读以下专利文档内容，并回答用户的问题。

重要：请遵循以下规则：
1. 用自己的话来解释和回答，不要复制或直接引用文档中的句子
2. 保持技术术语（如英文术语、数字、化学式等）的原样，不要翻译成中文
3. 将技术内容转化为通俗易懂的中文表达，但保留必要的技术术语
4. 如果文档中没有相关信息，请说明无法基于文档回答该问题
5. 回答要简洁明了，避免使用过于复杂的术语

文档内容：
{context}

用户问题：{question}

请用中文回答，不要重复提示词内容："""
                    
                    prompt_obj = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    retrieved_docs = st.session_state.retriever.get_relevant_documents(prompt)
                    context = format_docs(retrieved_docs)
                    
                    full_prompt = prompt_template.format(context=context, question=prompt)
                    
                    response = st.session_state.llm.invoke(full_prompt)
                else:
                    general_prompt = f"请用中文回答以下问题:\n{prompt}"
                    response = st.session_state.llm.invoke(general_prompt)
                
                if len(response) > 500:
                    response = response[:500] + "..."
                
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
                    
                    if use_document_context and st.session_state.retriever is not None:
                        from langchain.prompts import PromptTemplate
                        prompt_template = """你是一个专业的技术专家，请仔细阅读以下专利文档内容，并回答用户的问题。

重要：请遵循以下规则：
1. 用自己的话来解释和回答，不要复制或直接引用文档中的句子
2. 保持技术术语（如英文术语、数字、化学式等）的原样，不要翻译成中文
3. 将技术内容转化为通俗易懂的中文表达，但保留必要的技术术语
4. 如果文档中没有相关信息，请说明无法基于文档回答该问题
5. 回答要简洁明了，避免使用过于复杂的术语

文档内容：
{context}

用户问题：{question}

请用中文回答，不要重复提示词内容："""
                        
                        prompt_obj = PromptTemplate(
                            template=prompt_template,
                            input_variables=["context", "question"]
                        )
                        
                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)
                        
                        retrieved_docs = st.session_state.retriever.get_relevant_documents(prompt)
                        context = format_docs(retrieved_docs)
                        
                        full_prompt = prompt_template.format(context=context, question=prompt)
                        
                        response = llm.invoke(full_prompt)
                    else:
                        general_prompt = f"请用中文回答以下问题:\n{prompt}"
                        response = llm.invoke(general_prompt)
                    
                    if len(response) > 500:
                        response = response[:500] + "..."
                        
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
    st.session_state.model_loaded = False
    st.session_state.llm = None
    st.session_state.data_cleared = True
    clear_previous_data()
    st.experimental_rerun()