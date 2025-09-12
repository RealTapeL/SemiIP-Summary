from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChunkSummarizer:
    def __init__(self, model_adapter):
        self.model_adapter = model_adapter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def split_document(self, document_text: str) -> List[str]:
        return self.text_splitter.split_text(document_text)
        
    def summarize_chunks(self, chunks: List[str]) -> List[str]:
        summaries = []
        for chunk in chunks:
            summary = self.model_adapter.generate_summary(chunk)
            summaries.append(summary)
        return summaries