import torch
import faiss
import numpy as np
import os 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

path1 = os.path.join(os.path.dirname(__file__), "CLIP_BATCH_1.pt")
path2 = os.path.join(os.path.dirname(__file__), "CLIP_BATCH_2.pt")
output = os.path.join(os.path.dirname(__file__), "faiss_merged.bin")
tensor1 = torch.load(path1)
tensor2 = torch.load(path2)


stacked_tensor = torch.vstack((tensor1, tensor2))
print(f"Loaded {len(tensor1)} and {len(tensor2)} embeddings")
print(f"Stacked tensors to {len(stacked_tensor)} embeddings")

embeddings_np = stacked_tensor.cpu().detach().numpy().astype("float32")
num_samples, embedding_dim = embeddings_np.shape
print(f"Loaded {num_samples} embeddings of dimension {embedding_dim}")
faiss.normalize_L2(embeddings_np)


nlist = int(np.sqrt(num_samples))
res = faiss.StandardGpuResources()
config = faiss.GpuIndexIVFFlatConfig()
config.device = 0
index2 = faiss.GpuIndexIVFFlat(
    res, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT, config
)
index2.train(embeddings_np)
index2.add(embeddings_np)
print(f"Created and trained GpuIndexIVFFlat with {nlist} clusters")

cpu_index2 = faiss.index_gpu_to_cpu(index2)
faiss.write_index(cpu_index2, output)
