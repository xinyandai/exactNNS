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

#include "util.h"
#include "kmeans/kmeans_abstract.h"
#include "metric.h"
#include "imisequence.h"
#include "PQ/PQ.h"

/**
 * PQ: product quatization
 * sub-KMeans: multiple sub-KMeans consist of one PQ by merging(joining)
 * @tparam DataType
 */
template <typename DataType, typename KMeansType>
class ExactNNS : public ProductQuantization {
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
    ExactNNS(lshbox::Matrix<DataType>& data,
             lshbox::Matrix<DataType>& query,
             std::function< DataType (const DataType*, const DataType*, size_t) > cluster_dist,
             size_t num_codebook, size_t clusterK, int topK, size_t max_iteration)
            : ProductQuantization(data, query, cluster_dist, num_codebook, clusterK, max_iteration),
              data_(data),
              query_(query),
              topK_(topK),
              upper_bound_log_("logs/log_upper_bound_.log"),
              lower_bound_log_("logs/log_lower_bound_.log"),
              percent_log_("logs/log_percent_.log") {

        //query
        search();
        // write result
        writeResult();
    }

    ~ ExactNNS() {
        upper_bound_log_.close();
        lower_bound_log_.close();
        percent_log_.close();
    }
protected:

    /**
     * write topK result into text file
     */
    void writeResult();

    /**
     * calculate distance from query to cluster's center for each cluster in each code book
     * distance = ||q-c||^2  + ||c-x||^2 - 2*||q-c||*||c-x||
     *          = query_center * query_center + radius * radius - 2 * query_center * radius;
     * @param queryPoint
     * @param distToClusters
     */
    void calculateQueryCenterDist(DataType* queryPoint, vector<vector<pair<size_t , DataType> > >& distToClusters) ;

    /**
     * probe one bucket for one query, them update state.
     * @param imiSequence
     * @param distToClusters
     * @param maxHeap topK result will be stored in heap
     * @param queryPoint
     * @param upperBound  will be update
     * @param lowerBound will be update
     */
    virtual void inline probeOneBucket(IMISequence& imiSequence,
                        vector<vector<pair<size_t , DataType> > > & distToClusters,
                        priority_queue<DistDataMax<int > >& maxHeap,
                        DataType* queryPoint,
                        long double& upperBound,
                        long double& lowerBound) ;

    /**
     * search topK nearest neighbor for queryPoint, and result is saved in maxHeap.
     * @param queryPoint
     * @param maxHeap
     */
    long double searchOnePoint(DataType* queryPoint, priority_queue<DistDataMax<int > >& maxHeap) ;

    /**
     * search reult for each query point, and result is saved in maxHeaps.
     */
    void search() ;

protected:

    lshbox::Matrix<DataType>& data_;
    lshbox::Matrix<DataType>& query_;

    ofstream upper_bound_log_;
    ofstream lower_bound_log_;
    ofstream percent_log_;

    /***result is save in max heaps according to the distance from each query point to each retrieved data.**/
    vector<priority_queue<DistDataMax<int > > > maxHeaps_;

    int topK_;


};




template <typename DataType, typename KMeansType>
void ExactNNS<DataType, KMeansType>::writeResult() {
    string lshboxBenchFileName = "logs/nn.lshbox";
    // lshbox file
    ofstream lshboxFout(lshboxBenchFileName);
    if (!lshboxFout) {
        cout << "cannot create output file " << lshboxBenchFileName << endl;
    }
    lshboxFout << maxHeaps_.size() << "\t" << topK_ << endl;
    for (int query_id = 0; query_id < maxHeaps_.size(); ++query_id) {
        lshboxFout << query_id << "\t";
        priority_queue<DistDataMax<int > >& maxHeap = maxHeaps_[query_id];

        vector<DistDataMax<int> > tempList;
        while (maxHeap.size()>0) {
            DistDataMax<int> kDist = maxHeap.top();
            maxHeap.pop();
            tempList.push_back(kDist);
        }
        for (int i = tempList.size()-1; i >=0 ; --i) {
            DistDataMax<int> kDist = tempList[i];
            lshboxFout << kDist.data() << "\t" << kDist.dist() << "\t";
        }

        lshboxFout << endl;
    }
    lshboxFout.close();
    cout << "lshbox groundtruth are written into " << lshboxBenchFileName << endl;
}


template <typename DataType, typename KMeansType>
void ExactNNS<DataType, KMeansType>::calculateQueryCenterDist(DataType* queryPoint, vector<vector<pair<size_t , DataType> > >& distToClusters) {
    for (int codeBookID = 0; codeBookID < num_codebook_; ++codeBookID) {
        distToClusters[codeBookID].reserve(codebook_subdimension[codeBookID]);
    }

    for (int codeBookID = 0; codeBookID < num_codebook_; ++codeBookID) {

        KMeansType& means = kMeans_[codeBookID];
        vector<pair<size_t , DataType> >& dists = distToClusters[codeBookID];

        for (int clusterId = 0; clusterId < means.getClusters().size(); ++clusterId) {
            const Cluster<DataType>& cluster = means.getClusters()[clusterId];
            DataType query_center = metric::euclidDistance(
                    &queryPoint[codeBookID*codebook_subdim_max],
                    cluster.getCentralValues().data(),
                    codebook_subdimension[codeBookID]
            );
            DataType radius = cluster.getRadius();
            DataType dist_lower_bound;
            if (query_center > radius) {
                dist_lower_bound = (query_center - radius) * (query_center - radius);
            } else {
                dist_lower_bound = 0;
            }

            dists.push_back(std::make_pair(clusterId, dist_lower_bound));
        }

        std::sort(
                dists.begin(),
                dists.end(),
                [](const pair<size_t , DataType>& a, const pair<size_t , DataType>& b) {
                    return a.second < b.second;
                });
    }
}

/**
 * probe one bucket for one query, them update state.
 * @param imiSequence
 * @param distToClusters
 * @param maxHeap topK result will be stored in heap
 * @param queryPoint
 * @param upperBound  will be update
 * @param lowerBound will be update
 */
template <typename DataType, typename KMeansType>
void ExactNNS<DataType, KMeansType>::probeOneBucket(IMISequence& imiSequence,
                                   vector<vector<pair<size_t , DataType> > > & distToClusters,
                                   priority_queue<DistDataMax<int > >& maxHeap,
                                   DataType* queryPoint,
                                   long double& upperBound,
                                   long double& lowerBound) {

    auto next = imiSequence.next();
    unsigned long key = 0;
    for (int codeBookIndex = 0; codeBookIndex < num_codebook_; ++codeBookIndex) {
        unsigned clusterID = next.second[codeBookIndex];
        pair<size_t , DataType>& dists = distToClusters[codeBookIndex][clusterID];
        key = key * clusterK_ + dists.first;
    }

    if (tables_.find(key) == tables_.end()) {
        assert(false);
    }
    const Cluster<DataType>& nextCluster = tables_.find(key)->second;

    DistDataMax<int > kDist = maxHeap.top();

    for (int i = 0; i < nextCluster.getClusterSize(); ++i) {
        DataType distance = metric::euclidDistance(data_[nextCluster.getPoint(i).getID()], queryPoint, (size_t)query_.getDim());

        if (distance < kDist.dist()) {
            if (maxHeap.size() == topK_) {
                maxHeap.pop();
            }
            maxHeap.push(DistDataMax<int >(distance, nextCluster.getPoint(i).getID()));
            kDist = maxHeap.top();
        }
    }

    upperBound = metric::squareEuclidDistance(data_[kDist.data()], queryPoint, (size_t)query_.getDim());
    lowerBound = next.first;
}


template <typename DataType, typename KMeansType>
long double ExactNNS<DataType, KMeansType>::searchOnePoint(DataType* queryPoint, priority_queue<DistDataMax<int > >& maxHeap) {

    // (cluster_id, distance) from query to cluster center for each codebook
    // and each center in each codebook
    vector<vector<pair<size_t , DataType> > > distToClusters(num_codebook_);
    calculateQueryCenterDist(queryPoint, distToClusters);

    // lambda to get distance
    auto distor = [&](vector<unsigned > clusterIDS) {

        DataType dist_square = 0.0;
        for (int codebookIndex = 0; codebookIndex < num_codebook_; ++codebookIndex) {

            dist_square += distToClusters[codebookIndex][clusterIDS[codebookIndex]].second;
        }

        return dist_square;
    };

    IMISequence imiSequence(num_codebook_, vector<size_t >(num_codebook_, this->clusterK_), distor);

    long double upperBound = std::numeric_limits<long double>::max();
    long double lowerBound = - std::numeric_limits<long double>::max();

    maxHeap.push( DistDataMax<int >(std::numeric_limits<DataType>::max(), 0 ) );

    int bucketNum;
    for (bucketNum = 0; imiSequence.hasNext() && lowerBound<upperBound; bucketNum++) {

        probeOneBucket(imiSequence, distToClusters, maxHeap, queryPoint, upperBound, lowerBound);
        upper_bound_log_ << upperBound << "\t";
        lower_bound_log_ << lowerBound << "\t";
    }
    upper_bound_log_  << "\n";
    lower_bound_log_  << "\n";

    return bucketNum / (long double)std::pow(clusterK_, num_codebook_);
}


template <typename DataType, typename KMeansType>
void ExactNNS<DataType, KMeansType>::search() {
    // insert n empty queue
    maxHeaps_.insert(maxHeaps_.end(), query_.getSize(), priority_queue<DistDataMax<int > > ());

    long double average_percent = 0.0;

    for (int point_id = 0; point_id < query_.getSize(); ++point_id) {

        long double percent = searchOnePoint(query_[point_id], maxHeaps_[point_id]);

        percent_log_ << point_id << "\t : \t" << percent << std::endl;
        average_percent += percent;
    }

    percent_log_ << "average\t : \t" << average_percent / query_.getSize() << std::endl;

}

