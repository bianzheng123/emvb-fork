import faiss
import numpy as np
from faiss.loader import swig_ptr
from tqdm.notebook import tqdm
import pandas as pd
import os
import time
# from faiss.contrib.inspect_tools import get_invlist
from faiss.contrib import inspect_tools

# from run_inference import *

def get_invlist(invlists, l):
    """ returns the inverted lists content as a pair of (list_ids, list_codes).
    The codes are reshaped to a proper size
    """
    invlists = faiss.downcast_InvertedLists(invlists)
    ls = invlists.list_size(l)
    list_ids = np.zeros(ls, dtype='int64')
    ids = codes = None
    try:
        ids = invlists.get_ids(l)
        if ls > 0:
            faiss.memcpy(faiss.swig_ptr(list_ids), ids, list_ids.nbytes)
        codes = invlists.get_codes(l)
        if invlists.code_size != faiss.InvertedLists.INVALID_CODE_SIZE:
            list_codes = np.zeros((ls, invlists.code_size), dtype='uint8')
        else:
            # it's a BlockInvertedLists
            npb = invlists.n_per_block
            bs = invlists.block_size
            ls_round = (ls + npb - 1) // npb
            list_codes = np.zeros((ls_round, bs // npb, npb), dtype='uint8')
        if ls > 0:
            faiss.memcpy(faiss.swig_ptr(list_codes), codes, list_codes.nbytes)
    finally:
        if ids is not None:
            invlists.release_ids(l, ids)
        if codes is not None:
            invlists.release_codes(l, codes)
    return list_ids, list_codes

username = 'bianzheng'
dataset = 'lotte-500-gnd'

'''build the IVFPQ index'''
d = 128
embedding_path = f'/home/{username}/Dataset/vector-set-similarity-search/Embedding/{dataset}'
item_l_path = os.path.join(embedding_path, 'base_embedding', 'encoding0_float32.npy')
training_set = np.load(item_l_path)
print(training_set.shape)

n_centroid = 2 ** 10  # specify the number of centroids, somthing like 2**18
m = 16  # specify the number of partitions
nbits = 8
quantizer = faiss.IndexFlatL2(d)

index = faiss.IndexIVFPQ(quantizer, d, n_centroid, m, nbits)
index.train(training_set)

save_index_path = f"./index/{dataset}.index"
faiss.write_index(index, save_index_path)

## Path to the faiss index
index_path = save_index_path
# Path to the directory where the generated files will be saved
dest_dir = './index/lotte-500-gnd'
index_jmpq = faiss.read_index(index_path)

residuals = np.zeros([index_jmpq.ntotal, index_jmpq.pq.M], dtype=np.uint8)
all_indices = np.zeros([index_jmpq.ntotal], dtype=np.uint64)
print('index_jmpq.nlist', index_jmpq.nlist)
centroids = index_jmpq.quantizer.reconstruct_n(0, index_jmpq.nlist)
print("centroids", centroids)
centroids_to_pids = [None] * centroids.shape[0]

doclensArray = np.load(os.path.join(embedding_path, 'doclens.npy')).astype(np.uint32)
## total number of embeddings in your collection. Usually it can be obtained from index_jmpq.ntotal
tot_embedding = int(np.sum(np.load(os.path.join(embedding_path, 'doclens.npy'))))

n_docs = len(doclensArray)
emb2pid = np.zeros(tot_embedding, dtype=np.int64)
offset = 0
for i in range(n_docs):
    l = doclensArray[i]
    emb2pid[offset: offset + l] = i
    offset = offset + l
doc_offsets = np.zeros(n_docs, dtype=np.int64)
for i in range(1, n_docs):
    doc_offsets[i] = doc_offsets[i - 1] + doclensArray[i - 1]

print("emb2pid", emb2pid, len(emb2pid))

print(inspect_tools.get_invlist_sizes(index_jmpq.invlists))
for i in tqdm(range(index_jmpq.nlist)):
    ids, codes = get_invlist(index_jmpq.invlists, i)
    residuals[ids] = codes
    all_indices[ids] = i
    centroids_to_pids[i] = emb2pid[ids]

# Write centroids to pids
# print("centroids_to_pids", centroids_to_pids)
with open(os.path.join(dest_dir, "centroids_to_pids.txt"), "w") as file:
    for centroids_list in tqdm(centroids_to_pids):
        for x in centroids_list:
            file.write(f"{x} ")
        file.write("\n")

np.save(os.path.join(dest_dir, "query_embeddings.npy"),
        np.load(f'/home/{username}/Dataset/vector-set-similarity-search/Embedding/{dataset}/query_embedding.npy'))

# Write residuals
np.save(os.path.join(dest_dir, "residuals.npy"), residuals)

# Write centroids
np.save(os.path.join(dest_dir, "centroids.npy"), centroids)

# Write index_assignments
np.save(os.path.join(dest_dir, "index_assignment.npy"), all_indices)

# Write pq_centroids
pq_centroids = faiss.vector_to_array(index_jmpq.pq.centroids)
np.save(os.path.join(dest_dir, "pq_centroids.npy"), pq_centroids)

query_fname = f'/home/{username}/Dataset/vector-set-similarity-search/Embedding/{dataset}/query_embedding.npy'
n_query = np.load(query_fname).shape[0]
qid_l = np.arange(n_query)
np.savetxt(os.path.join('./index', "qid.txt"), qid_l, fmt='%d')