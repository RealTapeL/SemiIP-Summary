import streamlit as st
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
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

mineru_model_path = st.sidebar.text_input(
    "MinerU模型路径",
    value="/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/PDF-Extract-Kit-1.0",
    help="MinerU模型的本地路径"
)

use_document_context = st.sidebar.checkbox("使用文档内容作为上下文", value=True, help="取消勾选以进行通用对话")

query_timeout = st.sidebar.slider("查询超时时间(秒)", min_value=30, max_value=600, value=180, step=10)

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
            
        rag_storage_dir = "./rag_storage"
        if os.path.exists(rag_storage_dir):
            shutil.rmtree(rag_storage_dir)
            os.makedirs(rag_storage_dir, exist_ok=True)
            
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

@st.cache_resource
def load_models(_model_path, _mineru_model_path):
    st.info("正在预加载模型，请稍候...")
    try:
        os.environ['MINERU_MODEL_PATH'] = _mineru_model_path
        os.environ['MINERU_MODEL_SOURCE'] = 'local'
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
        import torch
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("正在加载分词器...")
        progress_bar.progress(10)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(_model_path)
        status_text.text("正在加载主模型...")
        progress_bar.progress(30)
        
        model = AutoModelForCausalLM.from_pretrained(
            _model_path,
            low_cpu_mem_usage=True,
            device_map="cuda",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
        status_text.text("正在加载嵌入模型...")
        progress_bar.progress(70)
        
        embed_model = AutoModel.from_pretrained("/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/all-MiniLM-L6-v2")
        embed_tokenizer = AutoTokenizer.from_pretrained("/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/all-MiniLM-L6-v2")
        
        progress_bar.progress(100)
        status_text.text("模型加载完成!")
        
        st.success("模型预加载完成!")
        return tokenizer, model, embed_tokenizer, embed_model
    except Exception as e:
        st.error(f"模型预加载失败: {str(e)}")
        logger.error(f"Model loading error: {e}", exc_info=True)
        return None, None, None, None

if not st.session_state.model_loaded and model_path and mineru_model_path:
    with st.spinner("正在预加载模型，请稍候...这可能需要几分钟时间"):
        tokenizer, model, embed_tokenizer, embed_model = load_models(model_path, mineru_model_path)
        if model is not None:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.embed_tokenizer = embed_tokenizer
            st.session_state.embed_model = embed_model
            st.session_state.model_loaded = True

if uploaded_file is not None and st.session_state.processed_pdf is None:
    with st.spinner("正在处理PDF文件..."):
        try:
            temp_dir = "/tmp/pdf_chat"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            from raganything import RAGAnything, RAGAnythingConfig
            from lightrag.utils import EmbeddingFunc
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
            import asyncio
            
            config = RAGAnythingConfig(
                working_dir="./rag_storage",
                parser="mineru",
                parse_method="auto",
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
                context_window=3,
                max_context_tokens=3000,
            )
            
            if not st.session_state.model_loaded:
                st.info("正在加载模型，请稍候...")
                os.environ['MINERU_MODEL_PATH'] = mineru_model_path
                os.environ['MINERU_MODEL_SOURCE'] = 'local'
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("正在加载分词器...")
                progress_bar.progress(10)
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                status_text.text("正在加载主模型...")
                progress_bar.progress(30)
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    device_map="cuda",
                    torch_dtype=torch.float16,
                    quantization_config=bnb_config,
                )
                status_text.text("正在加载嵌入模型...")
                progress_bar.progress(70)
                
                embed_model = AutoModel.from_pretrained("/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/all-MiniLM-L6-v2")
                embed_tokenizer = AutoTokenizer.from_pretrained("/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/all-MiniLM-L6-v2")
                
                progress_bar.progress(100)
                status_text.text("模型加载完成!")
                
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_loaded = True
            else:
                tokenizer = st.session_state.tokenizer
                model = st.session_state.model
            
            async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                if system_prompt:
                    full_prompt = system_prompt + "\n" + prompt
                else:
                    full_prompt = prompt
                    
                inputs = tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(inputs, max_new_tokens=500, temperature=0.1, repetition_penalty=1.2, do_sample=False)
                
                response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                return response
            
            if "embed_model" not in st.session_state:
                embed_model = AutoModel.from_pretrained("/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/all-MiniLM-L6-v2")
                embed_tokenizer = AutoTokenizer.from_pretrained("/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/all-MiniLM-L6-v2")
                st.session_state.embed_model = embed_model
                st.session_state.embed_tokenizer = embed_tokenizer
            else:
                embed_model = st.session_state.embed_model
                embed_tokenizer = st.session_state.embed_tokenizer
            
            def embedding_func(texts):
                from lightrag.llm.hf import hf_embed
                return hf_embed(texts, embed_tokenizer, embed_model)
            
            embedding_func_instance = EmbeddingFunc(
                embedding_dim=384,
                max_token_size=512,
                func=embedding_func
            )
            
            rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func_instance,
            )
            
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            
            st.info("开始解析PDF文档，请稍候...")
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
                        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                        import torch
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("正在加载分词器...")
                        progress_bar.progress(10)
                        
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )
                        
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        status_text.text("正在加载主模型...")
                        progress_bar.progress(30)
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            low_cpu_mem_usage=True,
                            device_map="cuda",
                            torch_dtype=torch.float16,
                            quantization_config=bnb_config,
                        )
                        status_text.text("正在加载嵌入模型...")
                        progress_bar.progress(70)
                        
                        embed_model = AutoModel.from_pretrained("/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/all-MiniLM-L6-v2")
                        embed_tokenizer = AutoTokenizer.from_pretrained("/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c1/SemiIP-Summary/all-MiniLM-L6-v2")
                        
                        progress_bar.progress(100)
                        status_text.text("模型加载完成!")
                        
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.model_loaded = True
                
                # 检查是否使用文档上下文并且RAG实例存在
                if use_document_context and st.session_state.rag_instance is not None:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # 首先尝试使用naive模式直接查询文档内容
                        response = loop.run_until_complete(
                            asyncio.wait_for(
                                st.session_state.rag_instance.aquery(prompt, mode="naive"),
                                timeout=float(query_timeout)
                            )
                        )
                        # 如果naive模式没有返回结果或结果太短
                        if not response or len(response.strip()) < 10:
                            st.info("直接文档查询未找到相关内容，尝试知识图谱查询...")
                            try:
                                response = loop.run_until_complete(
                                    asyncio.wait_for(
                                        st.session_state.rag_instance.aquery(prompt, mode="hybrid"),
                                        timeout=float(query_timeout)
                                    )
                                )
                            except:
                                # 如果知识图谱查询失败，尝试获取原始文档内容
                                pass
                        
                        # 如果仍然没有结果，直接使用文档内容进行问答
                        if not response or len(response.strip()) < 10:
                            st.info("使用文档内容直接生成回答...")
                            try:
                                # 获取文档内容
                                full_docs_storage = st.session_state.rag_instance.lightrag.full_docs
                                doc_keys = loop.run_until_complete(full_docs_storage.get_keys())
                                context_content = ""
                                for key in doc_keys[:1]:  # 只取第一个文档
                                    doc_content = loop.run_until_complete(full_docs_storage.get_by_id(key))
                                    if doc_content and 'content' in doc_content:
                                        context_content = doc_content['content']
                                        break
                                
                                if context_content:
                                    # 构造包含文档内容的提示
                                    full_prompt = f"基于以下文档内容回答问题：\n\n{context_content[:4000]}\n\n问题：{prompt}\n\n请用中文回答。"
                                    inputs = st.session_state.tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(st.session_state.model.device)
                                    with torch.no_grad():
                                        outputs = st.session_state.model.generate(inputs, max_new_tokens=500, temperature=0.1, repetition_penalty=1.2, do_sample=False)
                                    response = st.session_state.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                                else:
                                    # 如果没有文档内容，直接使用LLM
                                    st.info("未找到相关文档内容，直接使用大语言模型生成回答...")
                                    inputs = st.session_state.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048).to(st.session_state.model.device)
                                    with torch.no_grad():
                                        outputs = st.session_state.model.generate(inputs, max_new_tokens=500, temperature=0.1, repetition_penalty=1.2, do_sample=False)
                                    response = st.session_state.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                            except Exception as context_error:
                                logger.error(f"使用文档内容生成回答时出错: {context_error}")
                                # 最后的备选方案
                                st.info("直接使用大语言模型生成回答...")
                                inputs = st.session_state.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048).to(st.session_state.model.device)
                                with torch.no_grad():
                                    outputs = st.session_state.model.generate(inputs, max_new_tokens=500, temperature=0.1, repetition_penalty=1.2, do_sample=False)
                                response = st.session_state.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                    except asyncio.TimeoutError:
                        # 如果混合模式超时，尝试使用更简单的模式
                        try:
                            st.info("混合模式超时，尝试使用简化的检索模式...")
                            response = loop.run_until_complete(
                                asyncio.wait_for(
                                    st.session_state.rag_instance.aquery(prompt, mode="naive"),
                                    timeout=float(query_timeout)
                                )
                            )
                            if not response or len(response.strip()) == 0:
                                st.info("未找到相关文档内容，直接使用大语言模型生成回答...")
                                inputs = st.session_state.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048).to(st.session_state.model.device)
                                with torch.no_grad():
                                    outputs = st.session_state.model.generate(inputs, max_new_tokens=500, temperature=0.1, repetition_penalty=1.2, do_sample=False)
                                response = st.session_state.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                        except asyncio.TimeoutError:
                            # 如果仍然超时，使用直接的LLM生成
                            st.info("检索模式也超时，直接使用大语言模型生成回答...")
                            inputs = st.session_state.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048).to(st.session_state.model.device)
                            with torch.no_grad():
                                outputs = st.session_state.model.generate(inputs, max_new_tokens=500, temperature=0.1, repetition_penalty=1.2, do_sample=False)
                            response = st.session_state.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                else:
                    # 即使不使用文档上下文，也与大模型进行对话
                    inputs = st.session_state.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048).to(st.session_state.model.device)
                    with torch.no_grad():
                        outputs = st.session_state.model.generate(inputs, max_new_tokens=500, temperature=0.1, repetition_penalty=1.2, do_sample=False)
                    response = st.session_state.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                
                if len(response) > 1000:
                    response = response[:1000] + "..."
                
                st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")
                logger.error(f"Error generating response: {e}", exc_info=True)
                
                try:
                    st.info("使用直接对话模式...")
                    # 直接使用加载的模型进行对话，将PDF内容作为上下文
                    if st.session_state.rag_instance is not None and hasattr(st.session_state.rag_instance, 'lightrag'):
                        # 尝试获取文档内容作为上下文
                        try:
                            # 获取文档内容
                            full_docs_storage = st.session_state.rag_instance.lightrag.full_docs
                            doc_keys = loop.run_until_complete(full_docs_storage.get_keys())
                            context_content = ""
                            for key in doc_keys[:3]:  # 限制前3个文档
                                doc_content = loop.run_until_complete(full_docs_storage.get_by_id(key))
                                if doc_content and 'content' in doc_content:
                                    context_content += doc_content['content'] + "\n"
                            
                            if context_content:
                                full_prompt = f"基于以下文档内容回答问题：\n\n{context_content[:4000]}\n\n问题：{prompt}"
                            else:
                                full_prompt = prompt
                        except Exception as context_error:
                            logger.error(f"获取文档上下文时出错: {context_error}")
                            full_prompt = prompt
                    else:
                        full_prompt = prompt
                    
                    inputs = st.session_state.tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(st.session_state.model.device)
                    with torch.no_grad():
                        outputs = st.session_state.model.generate(inputs, max_new_tokens=500, temperature=0.1, repetition_penalty=1.2, do_sample=False)
                    response = st.session_state.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                    
                    if len(response) > 1000:
                        response = response[:1000] + "..."
                        
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