import os
import logging
from pathlib import Path
from typing import List
import shutil
import json

logger = logging.getLogger(__name__)

class PatentDocumentLoader:
    def __init__(self, documents_path: str):
        self.documents_path = Path(documents_path)
        self.output_dir = "./output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.configure_models()
    
    def configure_models(self):
        home_dir = Path.home()
        config_file = home_dir / "mineru.json"
        
        local_models_path = "/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c/SemiIP-Summary/PDF-Extract-Kit-1.0"
        
        config = {
            "models-dir": {
                "pipeline": local_models_path,
                "vlm": local_models_path
            },
            "config_version": "1.3.0"
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        
        os.environ["MINERU_MODEL_SOURCE"] = "local"
        os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "4"
    
    def load_documents(self) -> List[dict]:
        documents = []
            
        for file_path in self.documents_path.glob("*.pdf"):
            try:
                document = self.load_document(file_path)
                if document and "content" in document and document["content"] and not document["content"].startswith("ERROR:"):
                    documents.append(document)
                else:
                    if document:
                        logger.warning(f"Skipping document with error: {document.get('content', 'Unknown error')}")
                    else:
                        logger.warning(f"Document processing failed for {file_path.name}")
            except Exception as e:
                logger.warning(f"Exception while processing document {file_path.name}: {e}")
        return documents
        
    def load_document(self, file_path: Path) -> dict:
        try:
            try:
                text = self.extract_text(file_path)
                if text and len(text.strip()) > 0:
                    logger.info(f"Successfully processed {file_path.name} with MinerU")
                    return {
                        "name": file_path.name,
                        "path": str(file_path),
                        "content": text
                    }
                else:
                    error_msg = f"ERROR: MinerU returned empty text for file: {file_path.name}"
                    logger.error(error_msg)
                    return {
                        "name": file_path.name,
                        "path": str(file_path),
                        "content": error_msg
                    }
            except Exception as mineru_error:
                error_msg = f"ERROR: MinerU extraction failed for {file_path.name}: {str(mineru_error)}"
                logger.error(error_msg)
                return {
                    "name": file_path.name,
                    "path": str(file_path),
                    "content": error_msg
                }
                
        except Exception as e:
            error_msg = f"ERROR: Exception during processing of {file_path.name}: {str(e)}"
            logger.error(error_msg)
            return {
                "name": file_path.name,
                "path": str(file_path),
                "content": error_msg
            }
    
    def extract_text(self, file_path: Path) -> str:
        try:
            from mineru.cli.common import do_parse, read_fn
            from mineru.utils.enum_class import MakeMode
            
            pdf_file_name = file_path.stem
            
            pdf_output_dir = os.path.join(self.output_dir, pdf_file_name)
            os.makedirs(pdf_output_dir, exist_ok=True)
            
            pdf_bytes = read_fn(file_path)
            
            do_parse(
                output_dir=self.output_dir,
                pdf_file_names=[pdf_file_name],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=["ch"],
                backend="pipeline",
                parse_method="auto",
                formula_enable=False,
                table_enable=True,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                f_make_md_mode=MakeMode.NLP_MD,
            )
            
            output_file_path = os.path.join(self.output_dir, pdf_file_name, "auto", f"{pdf_file_name}.md")
            
            if os.path.exists(output_file_path):
                with open(output_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return content
            else:
                raise FileNotFoundError(f"Output file not found: {output_file_path}")
                
        except Exception as e:
            logger.warning(f"MinerU extraction failed for {file_path.name}: {e}")
            raise