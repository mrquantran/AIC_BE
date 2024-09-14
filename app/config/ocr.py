import os
from typing import Dict

def load_txt(file_path: str) -> Dict[str, str]:
    with open(file_path, "r", encoding='utf-8') as f:
        print(f"Loading data from {file_path}")
        return f.readlines()

content_l02 = load_txt(os.path.join(os.path.dirname(__file__), "../../data/ocr/text.txt"))

# content_l02 = [item.replace('\n', '') for item in content_l02 if item != '\n' and not item.startswith('File Path processing')]

def process_content(content: Dict[str, str]) -> Dict[str, str]:
    return [item.replace('\n', '') for item in content if item != '\n' and not item.startswith('File Path processing')]

# OCR_CONTENT = process_content(content_l02)


# create singleton OCR_CONTENT to store the OCR content
OCR_CONTENT = process_content(content_l02)