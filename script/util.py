import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re


def compile_file(username: str, is_debug: bool = False):
    build_path = f'/home/{username}/emvb-fork/build'
    build_type = 'Debug' if is_debug else 'Release'
    os.system(f'cd {build_path} && cmake -DCMAKE_BUILD_TYPE={build_type} ..')
    os.system(f'cd {build_path} && make -j')


def item_vecs_in_chunk(vecs_l: np.ndarray, itemlen_l: np.ndarray, itemID: int):
    vecs_start_idx = int(np.sum(itemlen_l[:itemID]))
    n_item_vecs = int(itemlen_l[itemID])
    item_vecs = vecs_l[vecs_start_idx: vecs_start_idx + n_item_vecs]
    return item_vecs


def get_n_chunk(base_dir: str):
    filename_l = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]

    doclen_patten = r'doclens(.*).npy'
    embedding_patten = r'encoding(.*)_float32.npy'

    match_obj_l = [re.match(embedding_patten, filename) for filename in filename_l]
    match_chunkID_l = np.array([int(_.group(1)) if _ else None for _ in match_obj_l])
    match_chunkID_l = match_chunkID_l[match_chunkID_l != np.array(None)]
    assert len(match_chunkID_l) == np.sort(match_chunkID_l)[-1] + 1
    return len(match_chunkID_l)


def get_DEFAULT_SIZE(username: str, dataset: str):
    embedding_dir = f'/home/{username}/Dataset/emvb-fork/Embedding/{dataset}/'
    base_embedding_dir = os.path.join(embedding_dir, 'base_embedding')
    itemlen_l_chunk = np.load(os.path.join(base_embedding_dir, f'doclens{0}.npy'))
    return len(itemlen_l_chunk)
