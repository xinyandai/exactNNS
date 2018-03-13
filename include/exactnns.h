//
// Created by xinyan on 3/13/18.
//

#include <iostream>
#include <queue>
#include <unordered_map>

#include "util.h"
#include "kmeans.h"
#include "metric.h"
#include "imisequence.h"

template <typename DataType>
class ExactNNS {
public:
    ExactNNS(lshbox::Matrix<DataType>& data,
             lshbox::Matrix<DataType>& query,
             size_t num_codebook, size_t clusterK, int topK, size_t max_iteration)
            : data_(data),
              query_(query),
              num_codebook_(num_codebook),
              clusterK_(clusterK),
              max_iterations_(max_iteration),
              topK_(topK){
        // 1.
        calculateCodeBook();
        // 2. pre process
        preProcess();
        // 3. build indexes with PQ
        // 2-dimension code, which means size of product quantization = 2
        buildIndex();
        // 4. merge sub-clusters
        mergeProduct();
        //query
        search();

        // write result
        writeResult();
    }

    void preProcess() {
        // 2.1 mean
        // 2.2 eigen allocate or optimized product quantization
    }

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

            while (maxHeap.size()>0) {
                DistDataMax<int> kDist = maxHeap.top();
                maxHeap.pop();

                lshboxFout << kDist.data() << "\t" << kDist.dist() << "\t";
            }
            lshboxFout << endl;
        }
        lshboxFout.close();
        cout << "lshbox groundtruth are written into " << lshboxBenchFileName << endl;
    }

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
                DataType dist_lower_bound = query_center * query_center + radius * radius - 2 * query_center * radius;

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

    void searchOnePoint(DataType* queryPoint, priority_queue<DistDataMax<int > >& maxHeap) {

        vector<vector<pair<size_t , DataType> > > distToClusters(num_codebook_);// dist from query to cluster center for each codebook and each center in each codebook

        calculateQueryCenterDist(queryPoint, distToClusters);

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

        for (int bucketNum = 0; lowerBound<upperBound && imiSequence.hasNext(); bucketNum++) {
            auto next = imiSequence.next();
            unsigned long key = 0;
            for (int codeBookIndex = num_codebook_-1; codeBookIndex >= 0; --codeBookIndex) {
                unsigned clusterID = next.second[codeBookIndex];
                pair<size_t , DataType>& dists = distToClusters[codeBookIndex][clusterID];
                key = key * clusterK_ + dists.first;
            }

            if (tables.find(key) == tables.end()) {
                assert(true);
            }
            const Cluster<DataType>& nextCluster = tables.find(key)->second;

            DistDataMax<int > kDist = maxHeap.top();

            for (int i = 0; i < nextCluster.getClusterSize(); ++i) {
                DataType distance = metric::euclidDistance(data_[nextCluster.getPoint(i).getID()], queryPoint, (size_t)query_.getDim());

                if (distance < kDist.dist()) {
                    maxHeap.pop();
                    maxHeap.push(DistDataMax<int >(distance, nextCluster.getPoint(i).getID()));
                    kDist = maxHeap.top();
                }
            }

            upperBound = metric::squareEuclidDistance(data_[kDist.data()], queryPoint, (size_t)query_.getDim());
            lowerBound = next.first;

        }

    }

    void search() {
        // insert n empty
        maxHeaps.insert(maxHeaps.end(), query_.getSize(), priority_queue<DistDataMax<int > > ());

        for (int point_id = 0; point_id < query_.getSize(); ++point_id) {

            searchOnePoint(query_[point_id], maxHeaps[point_id]);
        }
    }

    void buildIndex() {

        kMeans.reserve(num_codebook_);

        for (int i = 0; i < num_codebook_; ++i) {

            kMeans.push_back( KMeans<DataType> (clusterK_, (size_t)data_.getSize(), codebook_subdimension[i], max_iterations_, metric::squareEuclidDistance<DataType>) );
        }


        for (int codebook_index = 0; codebook_index < num_codebook_; ++codebook_index) {

            vector<Point<DataType> > subPoints;
            subPoints.reserve((size_t)data_.getSize());
            for (int point_id = 0; point_id < data_.getSize(); ++point_id) {

                subPoints.push_back(Point<DataType>(point_id, & data_[point_id][codebook_index * codebook_subdim_max]));
            }
            points.push_back(subPoints);
        }


        for (int i = 0; i < num_codebook_; ++i) {
            KMeans<DataType>& means = kMeans[i];

            means.run(points[i]);
            means.calculateRadius(metric::euclidDistance<DataType>);
        }

    }

    void mergeRecursive(int codebook_index, const Cluster<DataType>& cluster) {
        if (codebook_index == -1) {
            std::cout << "cluster: " << cluster.getID() << " size: "<< cluster.getClusterSize()<< "\n";
            for (int i = 0; i < cluster.getClusterSize(); ++i) {
                std::cout << "     point id: " << cluster.getPoint(i).getID() << "\n";
            }

            tables.emplace(std::make_pair(cluster.getID(), cluster));
        } else {

            KMeans<DataType>& subKMeans = kMeans[codebook_index];
            for (int i = 0; i < clusterK_; ++i) {

                Cluster<DataType> mergedCluster = cluster.merge( subKMeans.getClusters()[i], clusterK_ );
                mergeRecursive(codebook_index-1, mergedCluster );
            }
            std::cout << std::endl;
        }
    }

    void mergeProduct() {

        KMeans<DataType>& subKMeans = kMeans[num_codebook_-1];

        for (int i = 0; i < subKMeans.getClusters().size(); ++i) {

            const Cluster<DataType>& mergedCluster =  subKMeans.getClusters()[i];
            mergeRecursive((int)num_codebook_-2, mergedCluster );
        }

    }

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

    std::unordered_map<unsigned long ,  Cluster<DataType> > tables;

    vector<KMeans<DataType> > kMeans;
    vector<priority_queue<DistDataMax<int > > > maxHeaps;
    vector<vector<Point<DataType > > > points;

    size_t num_codebook_ ; // how much codebook
    size_t clusterK_ ;
    size_t max_iterations_;
    int topK_;

    size_t codebook_subdim_max;
    vector<size_t > codebook_subdimension;
};