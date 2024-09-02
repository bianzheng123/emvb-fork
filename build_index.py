import faiss
import numpy as np
from faiss.loader import swig_ptr
from tqdm.notebook import tqdm
import pandas as pd
import os
import time
from faiss.contrib.inspect_tools import get_invlist
from faiss.contrib import inspect_tools


# from run_inference import *


def build_index(username: str, dataset: str, n_centroid: int, pq_n_partition: int, pq_n_bit_per_partition: int):
    '''build the IVFPQ index'''
    embedding_path = f'/home/{username}/Dataset/vector-set-similarity-search/Embedding/{dataset}'

    # sample the training set

    item_l_path = os.path.join(embedding_path, 'base_embedding', 'encoding0_float32.npy')
    training_set = np.load(item_l_path)
    vec_dim = training_set.shape[1]
    print(training_set.shape)

    quantizer = faiss.IndexFlatL2(vec_dim)
    index = faiss.IndexIVFPQ(quantizer, vec_dim, n_centroid, pq_n_partition, pq_n_bit_per_partition)
    index.train(training_set)

    n_sample = 100

    index.add(training_set[:n_sample])
    arr = [index.invlists.list_size(i) for i in range(n_centroid)]
    assert np.sum(arr) == n_sample
    index.add(training_set[n_sample:])
    print("list size ", arr)

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

    np.save(os.path.join(dest_dir, "doclens.npy"),
            np.load(f'/home/{username}/Dataset/vector-set-similarity-search/Embedding/{dataset}/doclens.npy').astype(
                np.int32))

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


if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-500-gnd'
    n_centroid = 2 ** 10
    pq_n_partition = 16  # specify the number of partitions of vec_dim
    pq_n_bit_per_partition = 8

    build_index(username=username, dataset=dataset, n_centroid=n_centroid,
                pq_n_partition=pq_n_partition, pq_n_bit_per_partition=pq_n_bit_per_partition)
