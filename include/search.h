#pragma once

//
// Created by Xinyan Dai on 4/8/18. 
// Contact email: xinyan.dai@outlook.com
//
//
#include "exactnns.h"
#include "kmeans/kmeans_radius.h"

template <typename DataType>
int execute(const string& dataFile,
            const string& queryFile,
            const size_t num_codebook,
            const size_t clusterK,
            const int topK,
            const size_t max_iteration) {

    // train data(items) and queries
    lshbox::Matrix<DataType> data;
    lshbox::Matrix<DataType> query;

    // load data
    lshbox::loadFvecs(data, dataFile);
    lshbox::loadFvecs(query, queryFile);

    // search will be done inside this construct function
    ExactNNS<DataType, KBalls<DataType> >(data, query, metric::euclidDistance<DataType>, num_codebook, clusterK, topK, max_iteration);

}
