import json
import os
from typing import List, Dict, Any

try:
    import sglang as sgl
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    sgl = None

class GPTOSS20BAdapter:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config = self.loadConfig()
        self.backend = self.initializeBackend()
        
    def loadConfig(self) -> Dict[str, Any]:
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, "r") as f:
            return json.load(f)
            
    def initializeBackend(self):
        if SGLANG_AVAILABLE:
            # Initialize SGLang backend with reduced memory usage
            backend = sgl.Runtime(
                model_path=self.model_path,
                tp_size=1,
                mem_fraction_static=0.7,
                chunked_prefill_size=4096
            )
            sgl.set_default_backend(backend)
            return backend
        else:
            raise ImportError("SGLang not available. Please install sglang.")
        
    def generateSummary(self, text: str) -> str:
        if not SGLANG_AVAILABLE:
            return "SGLang not available"
            
        # Check for empty or very short text
        if not text or len(text.strip()) < 10:
            return "Document content is too short or empty to summarize."
            
        @sgl.function
        def summarize_patent(s, patent_text):
            s += "Summarize the following semiconductor intellectual property patent document:\n\n"
            s += patent_text + "\n\n"
            s += "Summary:\n"
            s += sgl.gen("summary", max_tokens=512, temperature=0.1)
        
        try:
            result = summarize_patent.run(patent_text=text)
            return result["summary"]
        except Exception as e:
            return f"Error generating summary: {str(e)}"
        
    def generateDetailedSummary(self, text: str) -> str:
        if not SGLANG_AVAILABLE:
            return "SGLang not available"
            
        # Check for empty or very short text
        if not text or len(text.strip()) < 10:
            return "Document content is too short or empty to summarize."
            
        @sgl.function
        def detailed_summary(s, patent_text):
            s += """Provide a detailed summary of the following semiconductor patent document. 
Include technical field, background, invention content, technical effects, and key claims:

"""
            s += patent_text + "\n\n"
            s += "Detailed Summary:\n"
            s += sgl.gen("detailed_summary", max_tokens=1024, temperature=0.1)
        
        try:
            result = detailed_summary.run(patent_text=text)
            return result["detailed_summary"]
        except Exception as e:
            return f"Error generating detailed summary: {str(e)}"
            
    def translateToChinese(self, text: str) -> str:
        if not SGLANG_AVAILABLE:
            return "SGLang not available"
            
        # Check for empty or very short text
        if not text or len(text.strip()) < 10:
            return "Text is too short or empty to translate."
            
        # Avoid translating error messages or Chinese text
        if "too short or empty" in text.lower() or "无法摘要" in text:
            return text
            
        @sgl.function
        def translate_text(s, english_text):
            s += "Translate the following English technical text to Chinese:\n\n"
            s += english_text + "\n\n"
            s += "Chinese translation:\n"
            s += sgl.gen("chinese_translation", max_tokens=2048, temperature=0.1)
        
        try:
            result = translate_text.run(english_text=text)
            return result["chinese_translation"]
        except Exception as e:
            return f"Error translating to Chinese: {str(e)}"