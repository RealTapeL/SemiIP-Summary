def clean_text(text: str) -> str:
    # Remove extra whitespace and normalize text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return ' '.join(chunk for chunk in chunks if chunk)

def truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."