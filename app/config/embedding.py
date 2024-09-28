import os
from app.services.clip_embedding import CLIPEmbedding

# Define the model name and nick name
model_name = "hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
model_nick_name = "CLIP_convnext_xxlarge_laion2B_s34B_b82K_augreg_soup"
# model_name = "hf-hub:apple/MobileCLIP-B-LT-OpenCLIP"
# model_nick_name = "mobile_clip_B_LT_openCLIP"

# Initialize the CLIPEmbedding class
embedder = CLIPEmbedding(
    model_name=model_name,
    model_nick_name=model_nick_name,
    device="cuda",
)
# embedder_ocr = CLIPEmbedding(
#     model_name="hf-hub:apple/MobileCLIP-B-LT-OpenCLIP",
#     model_nick_name="mobile_clip_B_LT_openCLIP",
# )

model_faiss = "faiss_merged.bin"

# Load the indexes (adjust paths as necessary)
faiss_path = os.path.join(
    os.path.dirname(__file__),
    f"../../data/embedding/{model_faiss}",
)
usearch_path = os.path.join(
    os.path.dirname(__file__),
    "../../data/embedding/usearch_batch2.bin",
)
audio_usearch_path = os.path.join(
    os.path.dirname(__file__),
    "../../data/embedding/audio_usearch.bin",
)
# faiss_ocr_path = os.path.join(
#     os.path.dirname(__file__), "../../data/embedding/ocr/faiss.bin"
# )
# usearch_ocr_path = os.path.join(
#     os.path.dirname(__file__), "../../data/embedding/ocr/usearch_index.bin"
# )
# Assuming you have a similar file for USearch
global2imgpath_path = os.path.join(
    os.path.dirname(__file__), "../../data/embedding/merged_json.json"
)
# global2imgpath_path_ocr = os.path.join(
#     os.path.dirname(__file__), "../../data/embedding/ocr/global_json_path.json"
# )
embedder.load_indexes(
    faiss_path=faiss_path,
    usearch_path=None,
    global2imgpath_path=global2imgpath_path,
    audio_usearch_path=None,
)
# embedder_ocr.load_indexes(
#     faiss_path=faiss_ocr_path,
#     usearch_path=usearch_ocr_path,
#     global2imgpath_path=global2imgpath_path,
# )
