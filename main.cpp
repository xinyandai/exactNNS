#include <iostream>


#include "include/search.h"

using namespace std;



int main(int argc, const char **argv) {

    //TODO parameters should be encapsulated in one struct

    // set default parameters value for testing
    string baseFile = "/home/xinyan/programs/gqr/data/audio/audio_base.fvecs";
    string queryFile = "/home/xinyan/programs/gqr/data/audio/audio_query.fvecs";
    size_t num_codebook = 2;
    size_t clusterK = 30;
    int topK = 10;
    size_t max_iteration = 500;

    //use parameters
    unordered_map<string, string> params = lshbox::parseParams(argc, argv);

    if (params.size() < 7)
    {
        std::cerr << "Usage: "
                  << "./exactNNS   "
                  << "--base_file=xxx "
                  << "--query_file=xxx "
                  << "--num_codebook=xxx"
                  << "--clusterK=xxx"
                  << "--topK=xxx"
                  << "--max_iteration=xxx"
                  << std::endl;

    } else {
        baseFile = params["base_file"];
        queryFile = params["query_file"];

        num_codebook = stoul(params["num_codebook"]);
        clusterK = stoul(params["clusterK"]);
        topK = stoi(params["topK"]);
        max_iteration = stoul(params["base_file"]);
    }

    execute<float >(baseFile, queryFile, num_codebook, clusterK, topK , max_iteration);

    return 0;
}