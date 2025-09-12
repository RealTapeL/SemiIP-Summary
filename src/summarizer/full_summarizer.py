from typing import List

class FullDocumentSummarizer:
    def __init__(self, model_adapter):
        self.model_adapter = model_adapter
        
    def summarize(self, document: dict) -> str:
        return self.model_adapter.generate_detailed_summary(document["content"])
        
    def summarize_with_chunks(self, document: dict) -> str:
        # Implementation for chunk-based summarization
        pass