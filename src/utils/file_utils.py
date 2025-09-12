import os
from pathlib import Path

def ensure_directory_exists(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    
def get_file_extension(file_path: str) -> str:
    return Path(file_path).suffix