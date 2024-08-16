import numpy as np

centroid_l = np.load('index/260k_m32_LOTTE_OPQ/centroids.npy')
print(f'centroid_l.shape {centroid_l.shape}')
n_centroid = centroid_l.shape[0]

with open('index/260k_m32_LOTTE_OPQ/centroids_to_pids.txt', 'r') as f:
    centroids_to_pids = f.readlines()
    centroids_to_pids = [line.strip() for line in centroids_to_pids]
    centroids_to_pids = [[int(x) for x in line.split(' ')] for line in centroids_to_pids]

len_pids = [len(pids) for pids in centroids_to_pids]
print(f"n_centroid in ivf {len(len_pids)}")
# print(f'len_pids {len_pids}')

vq_code_l = np.load('index/260k_m32_LOTTE_OPQ/index_assignment.npy')
print(f'vq_code_l.shape {vq_code_l.shape}')
for code in vq_code_l:
    assert code < n_centroid

pq_centroids = np.load('index/260k_m32_LOTTE_OPQ/pq_centroids.npy')
print(f'pq_centroids.shape {pq_centroids.shape}')

query_embeddings = np.load('index/260k_m32_LOTTE_OPQ/query_embeddings.npy')
print(f'query_embeddings.shape {query_embeddings.shape}')

residuals = np.load('index/260k_m32_LOTTE_OPQ/residuals.npy')
print(f'residuals.shape {residuals.shape}')
