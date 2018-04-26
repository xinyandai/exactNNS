//
// Created by darxan on 2018/4/26.
//
#pragma once
#ifndef EXACTNNS_PQ_H_H
#define EXACTNNS_PQ_H_H


//
//
// Created by Xinyan Dai on 3/13/18.
//
// Exact Nearest Neighbor Search with The Inverted Multi Index
//
//

#include <iostream>
#include <queue>
#include <unordered_map>

#include "../util.h"
#include "../kmeans/kmeans_abstract.h"
#include "../metric.h"
#include "../imisequence.h"

/**
 * sub-KMeans: multiple sub-KMeans consist of one PQ by merging(joining)
 * @tparam DataType
 */
template <typename DataType, typename KMeansType>
class ProductQuantization {
public:
    /**
     *
     * @param data
     * @param query
     * @param num_codebook
     * @param clusterK number of cluster in each sub-KMeans
     * @param topK
     * @param max_iteration
     */
    ProductQuantization(lshbox::Matrix<DataType>& data,
             lshbox::Matrix<DataType>& query,
             std::function< DataType (const DataType*, const DataType*, size_t) > cluster_dist,
             size_t num_codebook, size_t clusterK, size_t max_iteration)
            : data_(data),
              query_(query),
              num_codebook_(num_codebook),
              cluster_dist_(cluster_dist),
              clusterK_(clusterK),
              max_iterations_(max_iteration) {
        // 1. calculate dimension in each code book
        calculateCodeBook();
        // 2. pre process
        preProcess();
        // 3. build indexes with PQ
        buildIndex();
        // 4. merge sub-clusters
        productJoinClusters();
    }

    virtual ~ ExactNNS() {
    }

protected:
    // TODO
    virtual void preProcess() ;

    /**
     * build multi index for each sub-code-book with sub-KMeans
     */
    virtual void buildIndex() ;

    /**
     * join cluster for all combinations in all code book.
     *
     */
    void productJoinClusters();

    /**
     * calculate dimension of each code book
     */
    void calculateCodeBook();


protected:
    lshbox::Matrix<DataType>& data_;
    lshbox::Matrix<DataType>& query_;
    /**
     * key: merged(joined) cluster id.
     * value: merged(joined) cluster.
     */
    std::unordered_map<unsigned long ,  Cluster<DataType> > tables_;
    /***distance function for KMeans**/
    std::function<DataType (const DataType*, const DataType*, size_t) > cluster_dist_;
    /***sub-KMeans collection for all code books.**/
    vector<KMeansType > kMeans_;

    /***wrapped points used by kMeans.**/
    vector<vector<Point<DataType > > > points_;

    size_t num_codebook_ ;
    /***number of clusters in each sub-KMeans**/
    size_t clusterK_ ;

    size_t max_iterations_;

    /**
     * default dimension of code book except the last code book.
     * the last code book is no larger than codebook_subdim_max(=data_.getDim()/num_codebook_)
     */
    size_t codebook_subdim_max;

    /***all dimensions**/
    vector<size_t > codebook_subdimension;
};


template <typename DataType, typename KMeansType>
void ProductQuantization<DataType, KMeansType>::preProcess() {
    // 2.1 mean
    // 2.2 eigen allocate or optimized product quantization
}


template <typename DataType, typename KMeansType>
void ProductQuantization<DataType, KMeansType>::buildIndex() {

    kMeans_.reserve(num_codebook_);
    // initialize sub-KMeans
    for (int i = 0; i < num_codebook_; ++i) {
        kMeans_.push_back( KMeansType (clusterK_, (size_t)data_.getSize(), codebook_subdimension[i], max_iterations_, cluster_dist_) );
    }
    // wrap point for each sub-KMeans
    for (int code_book_index = 0; code_book_index < num_codebook_; ++code_book_index) {

        vector<Point<DataType> > subPoints;
        subPoints.reserve((size_t)data_.getSize());

        for (int point_id = 0; point_id < data_.getSize(); ++point_id) {

            subPoints.push_back(Point<DataType>(point_id, & data_[point_id][code_book_index * codebook_subdim_max]));
        }
        points_.push_back(subPoints);
    }

    // building indexes
    for (int i = 0; i < num_codebook_; ++i) {
        KMeansType& means = kMeans_[i];
        // run KMeans
        means.run(points_[i]);
    }

}


template <typename DataType, typename KMeansType>
void ProductQuantization<DataType, KMeansType>::productJoinClusters() {

    /**
     *  join function for two code book.
     */
    std::function<void (int, const Cluster<DataType>& )> mergeRecursive;

    mergeRecursive = [&] (int codebook_index, const Cluster<DataType>& cluster) {

        if (codebook_index == num_codebook_) {

            tables_.emplace(std::make_pair(cluster.getID(), cluster));
        } else {

            KMeansType& subKMeans = kMeans_[codebook_index];
            for (int i = 0; i < clusterK_; ++i) {
                // joined with the codebook_index'th code book
                Cluster<DataType> mergedCluster = cluster.merge( subKMeans.getClusters()[i], clusterK_ );
                // then joined with the (codebook_index+1)'th code book
                mergeRecursive(codebook_index+1, mergedCluster );
            }
        }
    };

    // start with kMeans[0]
    KMeansType& subKMeans = kMeans_[0];

    for (int i = 0; i < subKMeans.getClusters().size(); ++i) {

        // initialize cluster with al clusters in kMeans[0],
        const Cluster<DataType>& initialCluster =  subKMeans.getClusters()[i];
        // then merge clusters in kMeans[1], then merge cluster in kMeans[2] ....
        mergeRecursive(1, initialCluster );
    }

}


template <typename DataType, typename KMeansType>
void ProductQuantization<DataType, KMeansType>::calculateCodeBook() {

    codebook_subdim_max = data_.getDim() / num_codebook_;
    codebook_subdimension.insert(codebook_subdimension.end(), num_codebook_, codebook_subdim_max);

    if (data_.getDim() % num_codebook_) {

        codebook_subdimension[num_codebook_-1] = data_.getDim() % num_codebook_;
    }
}


#endif //EXACTNNS_PQ_H_H
