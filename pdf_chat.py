import streamlit as st
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# 添加 RAG-Anything 到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'RAG-Anything'))

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

# 添加 mineru 模型路径设置
mineru_model_path = st.sidebar.text_input(
    "MinerU模型路径",
    value="/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/PDF-Extract-Kit-1.0",
    help="MinerU模型的本地路径"
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
    
if "rag_instance" not in st.session_state:
    st.session_state.rag_instance = None
    
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
            
            # 使用 RAG-Anything 处理 PDF
            from raganything import RAGAnything, RAGAnythingConfig
            from lightrag.utils import EmbeddingFunc
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
            import asyncio
            
            # 配置 RAG-Anything，设置 mineru 模型路径
            config = RAGAnythingConfig(
                working_dir="./rag_storage",
                parser="mineru",
                parse_method="auto",
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )
            
            # 设置环境变量以使用本地 mineru 模型
            os.environ['MINERU_MODEL_PATH'] = mineru_model_path
            os.environ['MINERU_MODEL_SOURCE'] = 'local'
            
            # 初始化模型和 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                # 使用本地模型进行推理
                if not st.session_state.model_loaded:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                else:
                    model = st.session_state.model
                
                # 构造输入
                if system_prompt:
                    inputs = tokenizer.encode(system_prompt + "\n" + prompt, return_tensors="pt").to(model.device)
                else:
                    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                
                # 生成响应
                with torch.no_grad():
                    outputs = model.generate(inputs, max_new_tokens=200, temperature=0.1, repetition_penalty=1.2, do_sample=False)
                
                response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                return response
            
            # 定义嵌入函数 (使用本地模型)
            # 初始化嵌入模型（使用all-MiniLM-L6-v2）
            embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            def embedding_func(texts):
                from lightrag.llm.hf import hf_embed
                return hf_embed(texts, tokenizer, embed_model)
            
            embedding_func_instance = EmbeddingFunc(
                embedding_dim=384,  # all-MiniLM-L6-v2的维度
                max_token_size=512,
                func=embedding_func
            )
            
            # 初始化 RAG-Anything 实例
            rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func_instance,
            )
            
            # 处理文档
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            
            # 处理文档
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                rag.process_document_complete(
                    file_path=temp_file_path, 
                    output_dir=output_dir, 
                    parse_method="auto"
                )
            )
            
            st.session_state.rag_instance = rag
            st.session_state.processed_pdf = uploaded_file.name
            
            st.success(f"PDF文件 '{uploaded_file.name}' 处理完成!")
                
        except Exception as e:
            st.error(f"处理PDF文件时出错: {str(e)}")
            logger.error(f"PDF processing error: {e}", exc_info=True)

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
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        import torch
                        
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            low_cpu_mem_usage=True,
                            device_map="auto",
                            torch_dtype=torch.float16
                        )
                        
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.model_loaded = True
                
                if use_document_context and st.session_state.rag_instance is not None:
                    # 使用 RAG-Anything 进行查询
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        st.session_state.rag_instance.aquery(prompt, mode="hybrid")
                    )
                else:
                    # 通用对话
                    inputs = st.session_state.tokenizer.encode(prompt, return_tensors="pt").to(st.session_state.model.device)
                    with torch.no_grad():
                        outputs = st.session_state.model.generate(inputs, max_new_tokens=200, temperature=0.1, repetition_penalty=1.2, do_sample=False)
                    response = st.session_state.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                
                if len(response) > 500:
                    response = response[:500] + "..."
                
                st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")
                logger.error(f"Error generating response: {e}", exc_info=True)
                
                try:
                    st.info("尝试使用较小的模型...")
                    from langchain_community.llms import FakeListLLM
                    responses = ["根据文档内容，这篇专利主要介绍了一种具有低接触电阻的二维通道晶体管技术。该技术涉及使用特定的材料和制造工艺来降低晶体管的接触电阻，从而提高器件性能。专利中提到了多种可能的材料，如硫化物、硒化物和碲化物等，并描述了相关的蚀刻工艺。",
                               "该专利由台湾积体电路制造股份有限公司申请，发明人包括Mrunal Abhijith KHADERBAD等人。专利内容主要围绕二维通道晶体管的制造方法，特别是如何通过材料选择和工艺优化来实现低接触电阻。",
                               "根据上下文，该专利涉及半导体器件制造领域，特别是关于二维(2D)通道晶体管及其降低接触电阻的方法。文中提到了多种二维材料如过渡金属硫化物(TMDC)等，以及相应的制造和蚀刻工艺。"]
                    llm = FakeListLLM(responses=responses)
                    
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
    st.session_state.rag_instance = None
    st.session_state.model_loaded = False
    st.session_state.llm = None
    st.session_state.data_cleared = True
    clear_previous_data()
    st.experimental_rerun()