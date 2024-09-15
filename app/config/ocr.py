import os
from typing import Dict
import json
import numpy as np

def load_txt(file_path: str) -> Dict[str, str]:
    with open(file_path, "r", encoding='utf-8') as f:
        print(f"Loading data from {file_path}")
        return f.readlines()

content_l02 = load_txt(os.path.join(os.path.dirname(__file__), "../../data/ocr/text_2.txt"))
content_l03 = load_txt(os.path.join(os.path.dirname(__file__), "../../data/ocr/text_3.txt"))
content_l04 = load_txt(os.path.join(os.path.dirname(__file__), "../../data/ocr/text_4.txt"))
content_l05 = load_txt(os.path.join(os.path.dirname(__file__), "../../data/ocr/text_5.txt"))
content_l06 = load_txt(os.path.join(os.path.dirname(__file__), "../../data/ocr/text_6.txt"))
content_l07 = load_txt(os.path.join(os.path.dirname(__file__), "../../data/ocr/text_7.txt"))

content_l02_fix = [
    item.strip()
    for item in content_l02
    if item.strip() and not item.startswith("File Path processing")
]
content_l03_fix = [
    item.strip()
    for item in content_l03
    if item.strip() and not item.startswith("File Path processing")
]
content_l04_fix = [
    item.strip()
    for item in content_l04
    if item.strip() and not item.startswith("File Path processing")
]
content_l05_fix = [
    item.strip()
    for item in content_l05
    if item.strip() and not item.startswith("File Path processing")
]
content_l06_fix = [
    item.strip()
    for item in content_l06
    if item.strip() and not item.startswith("File Path processing")
]
content_l07_fix = [
    item.strip()
    for item in content_l07
    if item.strip() and not item.startswith("File Path processing")
]


# create singleton OCR_CONTENT to store the OCR content
OCR_CONTENT = content_l02_fix + content_l03_fix + content_l04_fix + content_l05_fix + content_l06_fix + content_l07_fix
