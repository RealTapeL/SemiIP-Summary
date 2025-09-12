import os
import sys
import logging
from datetime import datetime
from src.config.model_config import MODEL_PATH
from src.config.app_config import DOCUMENTS_PATH, SUMMARIES_PATH
from src.documents.loader import PatentDocumentLoader
from src.utils.file_utils import ensure_directory_exists

LOGS_DIR = "/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c/SemiIP-Summary/logs"
ensure_directory_exists(LOGS_DIR)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(LOGS_DIR, f"{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting patent processing in two stages")
        
        # Stage 1: Extract text from PDFs using MinerU
        logger.info("Stage 1: Extracting text from PDFs using MinerU")
        logger.info("Initializing document loader...")
        document_loader = PatentDocumentLoader(DOCUMENTS_PATH)
        documents = document_loader.load_documents()
        logger.info(f"Extracted text from {len(documents)} documents.")
        
        if not documents:
            logger.warning("No documents found in the specified directory.")
            return
            
        # Save extracted documents to intermediate files
        intermediate_dir = "/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c/SemiIP-Summary/data/intermediate"
        ensure_directory_exists(intermediate_dir)
        
        for i, document in enumerate(documents):
            # Save extracted content to intermediate file
            intermediate_filename = os.path.splitext(document['name'])[0] + "_extracted.txt"
            intermediate_filepath = os.path.join(intermediate_dir, intermediate_filename)
            
            with open(intermediate_filepath, "w", encoding="utf-8") as f:
                f.write(document['content'])
                
        logger.info(f"Saved extracted documents to {intermediate_dir}")
        
        # Stage 2: Load model and generate summaries
        logger.info("Stage 2: Loading model and generating summaries")
        logger.info(f"Initializing model adapter...")
        from src.models.gpt_oss_adapter import GPTOSS20BAdapter
        model_adapter = GPTOSS20BAdapter(MODEL_PATH)
        logger.info("Model adapter initialized successfully.")
        
        from src.summarizer.full_summarizer import FullDocumentSummarizer
        summarizer = FullDocumentSummarizer(model_adapter)
        
        # Ensure summaries directory exists
        ensure_directory_exists(SUMMARIES_PATH)
        
        # Statistics
        successful_summaries = 0
        failed_documents = 0
        
        for i, document in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {document['name']}")
            logger.info(f"Document length: {len(document['content'])} characters")
            
            # Check if it's an error message
            if document['content'].startswith("ERROR:"):
                logger.warning(f"Skipping document with error: {document['content']}")
                failed_documents += 1
                continue
                
            if len(document['content']) < 100:
                logger.warning(f"Document content seems too short. Content preview: {document['content'][:100]}...")
            
            try:
                # Generate English summary
                english_summary = summarizer.summarize(document)
                logger.info(f"English Summary: {english_summary}")
                
                # Generate Chinese summary
                chinese_summary = model_adapter.translateToChinese(english_summary)
                logger.info(f"Chinese Summary: {chinese_summary}")
                
                # Save summary to file
                # Clean special characters from filename
                clean_name = "".join(c for c in document['name'] if c.isalnum() or c in (' ','.','_','-')).rstrip()
                summary_filename = os.path.splitext(clean_name)[0] + "_summary.txt"
                summary_filepath = os.path.join(SUMMARIES_PATH, summary_filename)
                
                with open(summary_filepath, "w", encoding="utf-8") as f:
                    f.write(f"Document: {document['name']}\n")
                    f.write(f"Original Path: {document['path']}\n")
                    f.write("\n--- English Summary ---\n")
                    f.write(english_summary)
                    f.write("\n\n--- Chinese Summary ---\n")
                    f.write(chinese_summary)
                
                logger.info(f"Summary saved to: {summary_filepath}")
                successful_summaries += 1
                
            except Exception as e:
                logger.error(f"Error processing document {document['name']}: {e}")
                failed_documents += 1
                import traceback
                logger.error(traceback.format_exc())
                
            logger.info("-" * 50)
                
        logger.info(f"Finished patent summarization process")
        logger.info(f"Successfully processed: {successful_summaries} documents")
        logger.info(f"Failed to process: {failed_documents} documents")
                
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()