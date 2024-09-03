#include <iostream>
#include <vector>
#include <set>
#include <chrono>
#include <cfloat>
#include <string>
#include <tuple>
#include <queue>
#include <fstream>
#include <cstdint>
#include <omp.h>
#include <cnpy.h>

#include "include/parser.hpp"

#include "DocumentScorer.hpp"

using namespace std;

void configure(cmd_line_parser::parser &parser) {
    parser.add("topk", "Number of nearest neighbours.", "-topk", false);
    parser.add("nprobe", "Number of cell to look during index search.", "-nprobe", false);
    parser.add("thresh", "Threshold", "-thresh", false);
    parser.add("out_second_stage", "Number of candidate documents selected with bitvectors", "-out-second-stage",
               false);

    parser.add("thresh_query", "Threshold", "-thresh-query", false);
    parser.add("n_doc_to_score", "Number of document to score", "-n-doc-to-score", false);

    parser.add("username", "Username", "-username", false);
    parser.add("dataset", "Dataset", "-dataset", false);
    parser.add("build_index_suffix", "Suffix of the index construction", "-build-index-suffix", false);
    parser.add("retrieval_suffix", "Suffix of the retrieval phase", "-retrieval-suffix", false);
}

int main(int argc, char **argv) {
    omp_set_num_threads(1);

    cmd_line_parser::parser parser(argc, argv);
    configure(parser);
    bool success = parser.parse();
    if (!success)
        return 1;

    int topk = parser.get<int>("topk");
    float thresh = parser.get<float>("thresh");
    float thresh_query = parser.get<float>("thresh_query");

    size_t n_doc_to_score = parser.get<size_t>("n_doc_to_score");
    size_t nprobe = parser.get<size_t>("nprobe");
    size_t out_second_stage = parser.get<size_t>("out_second_stage");

    string username = parser.get<string>("username");
    string dataset = parser.get<string>("dataset");
    string build_index_suffix = parser.get<string>("build_index_suffix");
    string retrieval_suffix = parser.get<string>("retrieval_suffix");

    //    string index_dir_path = parser.get<string>("index_dir_path");
//    string queries_id_file = parser.get<string>("queries_id_file");
//    string alldoclens_path = parser.get<string>("alldoclens_path");
//    string output_path = parser.get<string>("output_path");

    char buffer[256];
    sprintf(buffer, "/home/%s/Dataset/vector-set-similarity-search/Index/%s/emvb", username.c_str(), dataset.c_str());
    string index_dir_path = std::string(buffer);

    string queries_id_file = index_dir_path + "/qID_l.txt";
    string alldoclens_path = index_dir_path + "/doclens.npy";

    sprintf(buffer, "/home/%s/Dataset/vector-set-similarity-search/Result/answer/", username.c_str());
    string output_path = std::string(buffer);

    string queries_path = index_dir_path + "/query_embeddings.npy";
    cnpy::NpyArray queriesArray = cnpy::npy_load(queries_path);

    size_t n_queries = queriesArray.shape[0];
    size_t vec_per_query = queriesArray.shape[1];
    size_t len = queriesArray.shape[2];

    cout << "Dimension: " << len << "\n"
         << "Number of queries: " << n_queries << "\n"
         << "Vector per query " << vec_per_query << "\n";
    uint16_t values_per_query = vec_per_query * len;
    valType *loaded_query_data = queriesArray.data<valType>();

    // load qid mapping file
    auto qid_map = load_qids(queries_id_file);

    cout << "queries id loaded\n";

    // load documents
    DocumentScorer document_scorer(alldoclens_path, index_dir_path, vec_per_query);


    ofstream out_file; // file with final output
    sprintf(buffer, "%s/%s-emvb-top%d-%s-%s.tsv", output_path.c_str(), dataset.c_str(), topk,
            build_index_suffix.c_str(),
            retrieval_suffix.c_str());
    string result_output_fname = std::string(buffer);
    out_file.open(result_output_fname);

    std::vector<double> search_time_l(n_queries);
    std::vector<double> cand_doc_retrieval_time_l(n_queries);
    std::vector<double> doc_filtering_time_l(n_queries);
    std::vector<double> second_stage_time_l(n_queries);
    std::vector<double> doc_scoring_time_l(n_queries);

    std::vector<uint32_t> n_cand_doc_retrieval_l(n_queries);
    std::vector<uint32_t> n_doc_filtering_l(n_queries);
    std::vector<uint32_t> n_second_stage_l(n_queries);

    cout << "SEARCH STARTED\n";
    for (int query_id = 0; query_id < n_queries; query_id++) {
        auto start = chrono::high_resolution_clock::now();
        globalIdxType q_start = query_id * values_per_query;

        // PHASE 1: candidate documents retrieval
        auto phase_start = chrono::high_resolution_clock::now();
        auto candidate_docs = document_scorer.find_candidate_docs(loaded_query_data, q_start, nprobe, thresh);
        const double cand_doc_retrieval_time = 1e-6 * chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - phase_start).count();

        // PHASE 2: candidate document filtering
        phase_start = chrono::high_resolution_clock::now();
        auto selected_docs = document_scorer.compute_hit_frequency(candidate_docs, thresh, n_doc_to_score);
        const double doc_filtering_time = 1e-6 * chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - phase_start).count();

        //  PHASE 3: second stage filtering
        phase_start = chrono::high_resolution_clock::now();
        auto selected_docs_2nd = document_scorer.second_stage_filtering(loaded_query_data, q_start, selected_docs,
                                                                        out_second_stage);
        const double second_stage_time = 1e-6 * chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - phase_start).count();

        // PHASE 4: document scoring
        phase_start = chrono::high_resolution_clock::now();
        auto query_res = document_scorer.compute_topk_documents_selected(loaded_query_data, q_start, selected_docs_2nd,
                                                                         topk, thresh_query);
        const double doc_scoring_time = 1e-6 * chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - phase_start).count();

        const double query_time = 1e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - start).count();

        search_time_l[query_id] = query_time;

        cand_doc_retrieval_time_l[query_id] = cand_doc_retrieval_time;
        doc_filtering_time_l[query_id] = doc_filtering_time;
        second_stage_time_l[query_id] = second_stage_time;
        doc_scoring_time_l[query_id] = doc_scoring_time;

        n_cand_doc_retrieval_l[query_id] = candidate_docs.size();
        n_doc_filtering_l[query_id] = selected_docs.size();
        n_second_stage_l[query_id] = selected_docs_2nd.size();

        for (int i = 0; i < topk; i++) {
            out_file << qid_map[query_id] << "\t" << get<0>(query_res[i]) << "\t" << i + 1 << "\t"
                     << get<1>(query_res[i]) << endl;
        }
        out_file.flush();
    }

    out_file.flush();
    out_file.close();

    sprintf(buffer, "%s/%s-emvb-performance-top%d-%s-%s.tsv", output_path.c_str(), dataset.c_str(), topk,
            build_index_suffix.c_str(),
            retrieval_suffix.c_str());
    string performance_output_fname = std::string(buffer);
    out_file.open(performance_output_fname);
    out_file << std::setprecision(3) << "search_time" << "\t" << "cand_doc_retrieval_time" << "\t"
             << "doc_filtering_time" << "\t" << "second_stage_time" << "\t" << "doc_scoring_time" << "\t"
             << "n_cand_doc_retrieval" << "\t" << "n_doc_filtering" << "\t" << "n_second_stage" << "\n" << std::endl;

    for (uint32_t qID = 0; qID < n_queries; qID++) {
        out_file << search_time_l[qID] << "\t" << cand_doc_retrieval_time_l[qID] << "\t"
                 << doc_filtering_time_l[qID] << "\t" << second_stage_time_l[qID] << "\t" << doc_scoring_time_l[qID]
                 << "\t"
                 << n_cand_doc_retrieval_l[qID] << "\t" << n_doc_filtering_l[qID] << "\t" << n_second_stage_l[qID]
                 << "\n";
    }

    cout << "Average Elapsed Time per query "
         << std::accumulate(search_time_l.begin(), search_time_l.end(), 0.0) / n_queries << "ms\n";

    return 0;
}
