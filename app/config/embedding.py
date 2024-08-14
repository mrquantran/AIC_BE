# config.py
import os
from app.services.clip_embedding import CLIPEmbedding

# Initialize the CLIPEmbedding class
embedder = CLIPEmbedding(
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
)  # Assuming you have a similar file for USearch
global2imgpath_path = os.path.join(
    os.path.dirname(__file__), "../../data/embedding/global2imgpath.json"
)

embedder.load_indexes(
    faiss_path=faiss_path,
    usearch_path=usearch_path,
    global2imgpath_path=global2imgpath_path,
)