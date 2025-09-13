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

def count_pdf_files(documents_path):
    pdf_count = 0
    for root, dirs, files in os.walk(documents_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_count += 1
    return pdf_count

def count_extracted_files(output_path):
    if not os.path.exists(output_path):
        return 0
    
    md_count = 0
    for item in os.listdir(output_path):
        item_path = os.path.join(output_path, item)
        if os.path.isdir(item_path):
            auto_path = os.path.join(item_path, 'auto')
            if os.path.exists(auto_path):
                for file in os.listdir(auto_path):
                    if file.endswith('.md'):
                        md_count += 1
                        break
    return md_count

def is_pdf_extraction_completed(documents_path, output_path):
    pdf_count = count_pdf_files(documents_path)
    extracted_count = count_extracted_files(output_path)
    logger.info(f"PDF文件总数: {pdf_count}, 已提取文件数: {extracted_count}")
    return pdf_count > 0 and pdf_count == extracted_count

def save_summary_as_md(document, english_summary, chinese_summary, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {document['name']} Summary\n\n")
        f.write(f"**Document**: {document['name']}\n\n")
        f.write(f"**Original Path**: {document['path']}\n\n")
        
        f.write("## English Summary\n\n")
        f.write(english_summary)
        f.write("\n\n")
        
        f.write("## Chinese Summary\n\n")
        f.write(chinese_summary)
        f.write("\n")

def save_summary_as_pdf(document, english_summary, chinese_summary, filepath):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_LEFT
        
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_LEFT
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            alignment=TA_LEFT
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT
        )
        
        title = Paragraph(f"{document['name']} Summary", title_style)
        story.append(title)
        
        doc_info = Paragraph(f"<b>Document</b>: {document['name']}", normal_style)
        story.append(doc_info)
        
        path_info = Paragraph(f"<b>Original Path</b>: {document['path']}", normal_style)
        story.append(path_info)
        story.append(Spacer(1, 0.2*inch))
        
        eng_title = Paragraph("English Summary", heading_style)
        story.append(eng_title)
        
        eng_summary = Paragraph(english_summary.replace('\n', '<br/>'), normal_style)
        story.append(eng_summary)
        story.append(Spacer(1, 0.2*inch))
        
        chn_title = Paragraph("Chinese Summary", heading_style)
        story.append(chn_title)
        
        chn_summary = Paragraph(chinese_summary.replace('\n', '<br/>'), normal_style)
        story.append(chn_summary)
        
        doc.build(story)
    except ImportError:
        logger.warning("ReportLab library not installed, cannot generate PDF files")
        with open(filepath.replace('.pdf', '_no_pdf_support.txt'), "w", encoding="utf-8") as f:
            f.write("PDF generation requires the ReportLab library.\n")
            f.write("Please run: pip install reportlab\n\n")
            f.write(f"Document: {document['name']}\n")
            f.write(f"Original Path: {document['path']}\n")
            f.write("\n--- English Summary ---\n")
            f.write(english_summary)
            f.write("\n\n--- Chinese Summary ---\n")
            f.write(chinese_summary)

def main():
    try:
        logger.info("Starting patent processing")
        
        output_dir = "/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c/SemiIP-Summary/output"
        
        if is_pdf_extraction_completed(DOCUMENTS_PATH, output_dir):
            logger.info("PDF extraction already completed. Skipping MinerU processing.")
        else:
            logger.info("Stage 1: Extracting text from PDFs using MinerU")
            logger.info("Initializing document loader...")
            document_loader = PatentDocumentLoader(DOCUMENTS_PATH)
            documents = document_loader.load_documents()
            logger.info(f"Extracted text from {len(documents)} documents.")
            
            if not documents:
                logger.warning("No documents found in the specified directory.")
                return
                
            intermediate_dir = "/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c/SemiIP-Summary/data/intermediate"
            ensure_directory_exists(intermediate_dir)
            
            for i, document in enumerate(documents):
                intermediate_filename = os.path.splitext(document['name'])[0] + "_extracted.txt"
                intermediate_filepath = os.path.join(intermediate_dir, intermediate_filename)
                
                with open(intermediate_filepath, "w", encoding="utf-8") as f:
                    f.write(document['content'])
                    
            logger.info(f"Saved extracted documents to {intermediate_dir}")
        
        logger.info("Stage 2: Loading model and generating summaries")
        logger.info(f"Initializing model adapter...")
        from src.models.gpt_oss_adapter import GPTOSS20BAdapter
        model_adapter = GPTOSS20BAdapter(MODEL_PATH)
        logger.info("Model adapter initialized successfully.")
        
        from src.summarizer.full_summarizer import FullDocumentSummarizer
        summarizer = FullDocumentSummarizer(model_adapter)
        
        ensure_directory_exists(SUMMARIES_PATH)
        
        intermediate_dir = "/media/ps/4be23142-02e1-4581-90e2-3316bdb6f49c/SemiIP-Summary/data/intermediate"
        documents = []
        if os.path.exists(intermediate_dir):
            for filename in os.listdir(intermediate_dir):
                if filename.endswith("_extracted.txt"):
                    filepath = os.path.join(intermediate_dir, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        original_name = filename[:-len("_extracted.txt")] + ".pdf"
                        documents.append({
                            'name': original_name,
                            'content': content,
                            'path': filepath
                        })
        
        logger.info(f"Loaded {len(documents)} documents from intermediate files.")
        
        if not documents:
            logger.warning("No documents found in the intermediate directory.")
            return
            
        successful_summaries = 0
        failed_documents = 0
        
        for i, document in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {document['name']}")
            logger.info(f"Document length: {len(document['content'])} characters")
            
            if document['content'].startswith("ERROR:"):
                logger.warning(f"Skipping document with error: {document['content']}")
                failed_documents += 1
                continue
                
            if len(document['content']) < 100:
                logger.warning(f"Document content seems too short. Content preview: {document['content'][:100]}...")
            
            try:
                english_summary = summarizer.summarize(document)
                logger.info(f"English Summary: {english_summary}")
                
                chinese_summary = model_adapter.translateToChinese(english_summary)
                logger.info(f"Chinese Summary: {chinese_summary}")
                
                clean_name = "".join(c for c in document['name'] if c.isalnum() or c in (' ','.','_','-')).rstrip()
                
                # Save as Markdown format
                md_filename = os.path.splitext(clean_name)[0] + "_summary.md"
                md_filepath = os.path.join(SUMMARIES_PATH, md_filename)
                save_summary_as_md(document, english_summary, chinese_summary, md_filepath)
                logger.info(f"Markdown summary saved to: {md_filepath}")
                
                # Save as PDF format
                pdf_filename = os.path.splitext(clean_name)[0] + "_summary.pdf"
                pdf_filepath = os.path.join(SUMMARIES_PATH, pdf_filename)
                save_summary_as_pdf(document, english_summary, chinese_summary, pdf_filepath)
                logger.info(f"PDF summary saved to: {pdf_filepath}")
                
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