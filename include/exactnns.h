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
#include "kmeans/kmeans.h"
#include "metric.h"
#include "imisequence.h"

/**
 * PQ: product quatization
 * sub-KMeans: multiple sub-KMeans consist of one PQ by merging(joining)
 * @tparam DataType
 */
template <typename DataType>
class ExactNNS {
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
             size_t num_codebook, size_t clusterK, int topK, size_t max_iteration)
            : data_(data),
              query_(query),
              num_codebook_(num_codebook),
              clusterK_(clusterK),
              max_iterations_(max_iteration),
              topK_(topK){
        // 1. calculate dimension in each code book
        calculateCodeBook();
        // 2. pre process
        preProcess();
        // 3. build indexes with PQ
        // 2-dimension code, which means size of product quantization = 2
        buildIndex();
        // 4. merge sub-clusters
        productJoinClusters();
        //query
        search();
        // write result
        writeResult();
    }

protected:
    // TODO
    void preProcess() {
        // 2.1 mean
        // 2.2 eigen allocate or optimized product quantization
    }

    /**
     * write topK result into text file
     */
    void writeResult() {
        string lshboxBenchFileName = "nn.lshbox";
        // lshbox file
        ofstream lshboxFout(lshboxBenchFileName);
        if (!lshboxFout) {
            cout << "cannot create output file " << lshboxBenchFileName << endl;
        }
        lshboxFout << maxHeaps.size() << "\t" << topK_ << endl;
        for (int query_id = 0; query_id < maxHeaps.size(); ++query_id) {
            lshboxFout << query_id << "\t";
            priority_queue<DistDataMax<int > >& maxHeap = maxHeaps[query_id];

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


    /**
     * calculate distance from query to cluster's center for each cluster in each code book
     * distance = ||q-c||^2  + ||c-x||^2 - 2*||q-c||*||c-x||
     *          = query_center * query_center + radius * radius - 2 * query_center * radius;
     * @param queryPoint
     * @param distToClusters
     */
    void calculateQueryCenterDist(DataType* queryPoint, vector<vector<pair<size_t , DataType> > >& distToClusters) {
        for (int codeBookID = 0; codeBookID < num_codebook_; ++codeBookID) {
            distToClusters[codeBookID].reserve(codebook_subdimension[codeBookID]);
        }

        for (int codeBookID = 0; codeBookID < num_codebook_; ++codeBookID) {

            KMeans<DataType>& means = kMeans[codeBookID];
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
    void inline probeOneBucket(IMISequence& imiSequence,
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

        if (tables.find(key) == tables.end()) {
            assert(false);
        }
        const Cluster<DataType>& nextCluster = tables.find(key)->second;

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

    /**
     * search topK nearest neighbor for queryPoint, and result is saved in maxHeap.
     * @param queryPoint
     * @param maxHeap
     */
    long double searchOnePoint(DataType* queryPoint, priority_queue<DistDataMax<int > >& maxHeap) {

        // (cluster_id, distance) from query to cluster center for each codebook and each center in each codebook
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
        //
        maxHeap.push( DistDataMax<int >(std::numeric_limits<DataType>::max(), 0 ) );

        int bucketNum;
        for (bucketNum = 0; imiSequence.hasNext() && lowerBound<upperBound; bucketNum++) {

            probeOneBucket(imiSequence, distToClusters, maxHeap, queryPoint, upperBound, lowerBound);
        }

        return bucketNum / (long double)std::pow(clusterK_, num_codebook_);
    }

    /**
     * search reult for each query point, and result is saved in maxHeaps.
     */
    void search() {
        // insert n empty
        maxHeaps.insert(maxHeaps.end(), query_.getSize(), priority_queue<DistDataMax<int > > ());

        long double average_percent = 0.0;

        for (int point_id = 0; point_id < query_.getSize(); ++point_id) {

            long double percent = searchOnePoint(query_[point_id], maxHeaps[point_id]);

            std::cout << "percent : " << percent << std::endl;
            average_percent += percent;
        }

        std::cout << "average percent : " << average_percent / query_.getSize() << std::endl;

    }

    /**
     * build multi index for each sub-code-book with sub-KMeans
     */
    void buildIndex() {

        kMeans.reserve(num_codebook_);

        // initialize sub-KMeans
        for (int i = 0; i < num_codebook_; ++i) {

            kMeans.push_back( KMeans<DataType> (clusterK_, (size_t)data_.getSize(), codebook_subdimension[i], max_iterations_, metric::euclidDistance<DataType>) );
        }

        // wrap point for each sub-KMeans
        for (int code_book_index = 0; code_book_index < num_codebook_; ++code_book_index) {

            vector<Point<DataType> > subPoints;
            subPoints.reserve((size_t)data_.getSize());

            for (int point_id = 0; point_id < data_.getSize(); ++point_id) {

                subPoints.push_back(Point<DataType>(point_id, & data_[point_id][code_book_index * codebook_subdim_max]));
            }
            points.push_back(subPoints);
        }

        // building indexes
        for (int i = 0; i < num_codebook_; ++i) {
            KMeans<DataType>& means = kMeans[i];
            // run KMeans
            means.run(points[i]);
            // calculate radius in each sub-KMeans
            means.calculateRadius(metric::euclidDistance<DataType>);
        }

    }

    /**
     * join cluster for all combinations in all code book.
     *
     */
    void productJoinClusters() {

        /**
         *  join function for two code book.
         */
        std::function<void (int, const Cluster<DataType>& )> mergeRecursive;

        /**
         * join function.
         */
        mergeRecursive = [&] (int codebook_index, const Cluster<DataType>& cluster) {

            if (codebook_index == num_codebook_) {
                // joined with "num_codebook_" clusters, put into tables because this is one kind of valid combination
                // cluster.getID() = cluster[0] * clusterK_^(num_codebook_-1)
                //                 + cluster[1] * clusterK_^(num_codebook_-2)
                //                 ...
                //                 + cluster[num_codebook_-1]
                tables.emplace(std::make_pair(cluster.getID(), cluster));

            } else {

                KMeans<DataType>& subKMeans = kMeans[codebook_index];
                for (int i = 0; i < clusterK_; ++i) {
                    // joined with the codebook_index'th code book
                    Cluster<DataType> mergedCluster = cluster.merge( subKMeans.getClusters()[i], clusterK_ );
                    // then joined with the (codebook_index+1)'th code book
                    mergeRecursive(codebook_index+1, mergedCluster );
                }
            }
        };

        // start with kMeans[0]
        KMeans<DataType>& subKMeans = kMeans[0];

        for (int i = 0; i < subKMeans.getClusters().size(); ++i) {

            // initialize cluster with al clusters in kMeans[0],
            const Cluster<DataType>& initialCluster =  subKMeans.getClusters()[i];
            // then merge clusters in kMeans[1], then merge cluster in kMeans[2] ....
            mergeRecursive(1, initialCluster );
        }

    }

    /**
     * calculate dimension of each code book
     */
    void calculateCodeBook() {

        codebook_subdim_max = data_.getDim() / num_codebook_;
        codebook_subdimension.insert(codebook_subdimension.end(), num_codebook_, codebook_subdim_max);
        if (data_.getDim() % num_codebook_) {
            codebook_subdimension[num_codebook_-1] = data_.getDim() % num_codebook_;
        }
    }


protected:
    lshbox::Matrix<DataType>& data_;
    lshbox::Matrix<DataType>& query_;

    /**
     * key value table.
     * key: merged(joined) cluster id.
     * value: merged(joined) cluster.
     */
    std::unordered_map<unsigned long ,  Cluster<DataType> > tables;
    /**
     * sub-KMeans collection for all code books.
     */
    vector<KMeans<DataType> > kMeans;

    /**
     * result is save in max heaps according to the distance from each query point to each retrieved data.
     */
    vector<priority_queue<DistDataMax<int > > > maxHeaps;

    /**
     * wrapped points for kMeans.
     */
    vector<vector<Point<DataType > > > points;

    size_t num_codebook_ ;
    /**
     * number of clusters in each sub-KMeans
     */
    size_t clusterK_ ;

    size_t max_iterations_;

    int topK_;

    /**
     * default dimension of code book except the last code book.
     * the last code book is no larger than codebook_subdim_max(=data_.getDim()/num_codebook_)
     */
    size_t codebook_subdim_max;
    /**
     * all dimensions
     */
    vector<size_t > codebook_subdimension;
};