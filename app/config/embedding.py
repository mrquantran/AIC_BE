# config.py
import os
from app.services.clip_embedding import CLIPEmbedding

# Initialize the CLIPEmbedding class
embedder = CLIPEmbedding(
    model_name="hf-hub:apple/MobileCLIP-B-LT-OpenCLIP",
    model_nick_name="mobile_clip_B_LT_openCLIP",
)
embedder_ocr = CLIPEmbedding(
    model_name="hf-hub:apple/MobileCLIP-B-LT-OpenCLIP",
    model_nick_name="mobile_clip_B_LT_openCLIP",
)

# Load the indexes (adjust paths as necessary)
faiss_path = os.path.join(
    os.path.dirname(__file__),
    "../../data/embedding/faiss.bin",
)
usearch_path = os.path.join(
    os.path.dirname(__file__),
    "../../data/embedding/usearch.bin",
)  
faiss_ocr_path = os.path.join(
    os.path.dirname(__file__), "../../data/embedding/ocr/faiss_index.bin"
)
usearch_ocr_path = os.path.join(
    os.path.dirname(__file__), "../../data/embedding/ocr/usearch_index.bin"
)
# Assuming you have a similar file for USearch
global2imgpath_path = os.path.join(
    os.path.dirname(__file__), "../../data/embedding/global2imgpath.json"
)
global2imgpath_path_ocr = os.path.join(
    os.path.dirname(__file__), "../../data/embedding/ocr/global_json_path.json"
)


embedder.load_indexes(
    faiss_path=faiss_path,
    usearch_path=usearch_path,
    global2imgpath_path=global2imgpath_path,
)
embedder_ocr.load_indexes(
    faiss_path=faiss_ocr_path,
    usearch_path=usearch_ocr_path,
    global2imgpath_path=global2imgpath_path,
)
