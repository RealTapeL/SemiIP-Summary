import os
import json
from pathlib import Path

class DocumentStore:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def save_document(self, document_id: str, document_data: dict):
        file_path = self.storage_path / f"{document_id}.json"
        with open(file_path, "w") as f:
            json.dump(document_data, f)
            
    def load_document(self, document_id: str) -> dict:
        file_path = self.storage_path / f"{document_id}.json"
        with open(file_path, "r") as f:
            return json.load(f)