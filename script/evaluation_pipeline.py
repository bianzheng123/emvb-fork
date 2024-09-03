import copy

import numpy as np
import os
import time
from typing import Dict, Callable
import sys
import tqdm
import json
import pandas as pd

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir)
sys.path.append(ROOT_PATH)
from script import performance_metric
from script import util


def save_retrieval_result(est_dist_l: np.ndarray, est_id_l: np.ndarray,
                          queryID_l: list,
                          retrieval_result_filename: str):
    assert len(est_dist_l) == len(est_id_l)
    n_query = len(est_dist_l)
    with open(retrieval_result_filename, "w") as f:
        for local_queryID, single_dist_l, single_id_l in zip(np.arange(n_query), est_dist_l, est_id_l):
            queryID = queryID_l[local_queryID]
            for rank, (dist, passageID) in enumerate(zip(single_dist_l, single_id_l)):
                f.write(f"{queryID}\t{passageID}\t{rank + 1}\t{dist}\n")


def load_query(username: str, dataset: str):
    embedding_dir = f'/home/{username}/Dataset/vector-set-similarity-search/Embedding/{dataset}/'
    query_l = np.load(os.path.join(embedding_dir, 'query_embedding.npy'))

    rawdata_path = f'/home/{username}/Dataset/vector-set-similarity-search/RawData/{dataset}'

    query_text_filename = os.path.join(rawdata_path, f'document/queries.dev.tsv')
    if os.path.exists(query_text_filename):
        queryID_l = []
        with open(query_text_filename, 'r') as f:
            for line in f:
                query_text_l = line.split('\t')
                queryID_l.append(int(query_text_l[0]))
            assert len(queryID_l) == len(query_l)
    else:
        queryID_l = np.arange(len(query_l))
    return query_l, queryID_l


def approximate_solution_retrieval_outter(username: str, dataset: str,
                                          build_index_config: Dict,
                                          build_index_suffix: str,
                                          topk: int,
                                          retrieval_parameter_l: list,
                                          retrieval_f: Callable,
                                          method_name: str = 'emvb',
                                          ):
    query_l, queryID_l = load_query(username=username, dataset=dataset)

    mrr_gnd, success_gnd, recall_gnd_id_m = performance_metric.load_groundtruth(username=username, dataset=dataset,
                                                                                topk=topk)

    final_result_l = []
    for retrieval_config in retrieval_parameter_l:
        retrieval_suffix, search_time_m, time_ms_l = retrieval_f(
            username=username, dataset=dataset,
            build_index_suffix=build_index_suffix,
            retrieval_config=retrieval_config)

        result_answer_path = f'/home/{username}/Dataset/vector-set-similarity-search/Result/answer/'
        method_ans_name = f'{dataset}-{method_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'
        retrieval_result_filename = os.path.join(result_answer_path, method_ans_name)

        performance_metric.read_method_tsv(base_dir=result_answer_path, dataset=dataset, method_name=method_name,
                                           topk=topk, build_index_suffix=build_index_suffix,
                                           retrieval_suffix=retrieval_suffix)

        recall_l, mrr_l, success_l, search_accuracy_m = performance_metric.count_accuracy(
            username=username, dataset=dataset, topk=topk,
            method_name=method_name, build_index_suffix=build_index_suffix, retrieval_suffix=retrieval_suffix,
            mrr_gnd=mrr_gnd, success_gnd=success_gnd, recall_gnd_id_m=recall_gnd_id_m)

        build_index_config = copy.deepcopy(build_index_config)
        if 'item_n_vec_l' in build_index_config:
            del build_index_config['item_n_vec_l']
        if 'item_n_vecs_l' in build_index_config:
            del build_index_config['item_n_vecs_l']
        retrieval_info_m = {'n_query': len(queryID_l), 'topk': topk,
                            'build_index': build_index_config, 'retrieval': retrieval_config,
                            'search_time': search_time_m, 'search_accuracy': search_accuracy_m}
        del retrieval_info_m['build_index']['n_centroid_f']
        method_performance_name = f'{dataset}-retrieval-{method_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.json'
        result_performance_path = f'/home/{username}/Dataset/vector-set-similarity-search/Result/performance'
        performance_filename = os.path.join(result_performance_path, method_performance_name)
        with open(performance_filename, "w") as f:
            json.dump(retrieval_info_m, f)

        df = pd.DataFrame({'time(ms)': time_ms_l, 'recall': recall_l})
        df.index.name = 'local_queryID'
        if mrr_l:
            df['mrr'] = mrr_l
        if success_l:
            df['success'] = success_l
        single_query_performance_name = f'{dataset}-retrieval-{method_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.csv'
        result_performance_path = f'/home/{username}/Dataset/vector-set-similarity-search/Result/single_query_performance'
        single_query_performance_filename = os.path.join(result_performance_path, single_query_performance_name)
        df.to_csv(single_query_performance_filename, index=True)

        print("#############final result###############")
        print("filename", method_performance_name)
        print("search time", retrieval_info_m['search_time'])
        print("search accuracy", retrieval_info_m['search_accuracy'])
        print("########################################")

        final_result_l.append({'filename': method_performance_name, 'search_time': retrieval_info_m['search_time'],
                               'search_accuracy': retrieval_info_m['search_accuracy']})

    for final_result in final_result_l:
        print("#############final result###############")
        print("filename", final_result['filename'])
        print("search time", final_result['search_time'])
        print("search accuracy", final_result['search_accuracy'])
        print("########################################")
