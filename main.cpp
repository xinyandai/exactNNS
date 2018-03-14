#include <iostream>


#include "include/exactnns.h"


using namespace std;


template <typename DataType>
int execute(const string& dataFile, const string& queryFile) {

    lshbox::Matrix<DataType> data;
    lshbox::Matrix<DataType> query;


    lshbox::loadFvecs(data, dataFile);
    lshbox::loadFvecs(query, queryFile);


    size_t num_codebook = 2;
    size_t clusterK = 32;
    int topK = 10;
    size_t max_iteration = 500;

    ExactNNS<DataType>(data, query, num_codebook, clusterK, topK, max_iteration);

}


int main(int argc, char *argv[]) {

    string dataFile = "/home/xinyan/programs/gqr/data/audio/audio_base.fvecs";
    string queryFile = "/home/xinyan/programs/gqr/data/audio/audio_query.fvecs";

    execute<float >(dataFile, queryFile);

    return 0;
}