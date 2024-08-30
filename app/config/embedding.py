import os
from app.services.clip_embedding import CLIPEmbedding

model_name = "hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
model_nick_name = "CLIP_convnext_xxlarge_laion2B_s34B_b82K_augreg_soup"

# Initialize the CLIPEmbedding class
embedder = CLIPEmbedding(
    model_name=model_name,
    model_nick_name=model_nick_name,
)
# embedder_ocr = CLIPEmbedding(
#     model_name="hf-hub:apple/MobileCLIP-B-LT-OpenCLIP",
#     model_nick_name="mobile_clip_B_LT_openCLIP",
# )

# Load the indexes (adjust paths as necessary)
faiss_path = os.path.join(
    os.path.dirname(__file__),
    "../../data/embedding/faiss.pt",
)
usearch_path = os.path.join(
    os.path.dirname(__file__),
    "../../data/embedding/usearch.bin",
)
# faiss_ocr_path = os.path.join(
#     os.path.dirname(__file__), "../../data/embedding/ocr/faiss.bin"
# )
# usearch_ocr_path = os.path.join(
#     os.path.dirname(__file__), "../../data/embedding/ocr/usearch_index.bin"
# )
# Assuming you have a similar file for USearch
global2imgpath_path = os.path.join(
    os.path.dirname(__file__), "../../data/embedding/global2imgpath.json"
)
# global2imgpath_path_ocr = os.path.join(
#     os.path.dirname(__file__), "../../data/embedding/ocr/global_json_path.json"
# )


embedder.load_indexes(
    faiss_path=faiss_path,
    usearch_path=usearch_path,
    global2imgpath_path=global2imgpath_path,
)
# embedder_ocr.load_indexes(
#     faiss_path=faiss_ocr_path,
#     usearch_path=usearch_ocr_path,
#     global2imgpath_path=global2imgpath_path,
# )
