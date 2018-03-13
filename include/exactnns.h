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

        // 3. query
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
            priority_queue<DistDataMax<Point<DataType> > >& maxHeap = maxHeaps[query_id];

            while (maxHeap.size()>0) {
                DistDataMax<Point<DataType>> kDist = maxHeap.top();
                maxHeap.pop();

                lshboxFout << kDist.data().getID() << "\t" << kDist.dist() << "\t";
            }
            lshboxFout << endl;
        }
        lshboxFout.close();
        cout << "lshbox groundtruth are written into " << lshboxBenchFileName << endl;
    }

    void search() {
        // insert n empty
        maxHeaps.insert(maxHeaps.end(), query_.getSize(), priority_queue<DistDataMax<Point<DataType> > > ());

        for (int point_id = 0; point_id < query_.getSize(); ++point_id) {

            DataType* queryPoint = query_[point_id];

            auto distor = [&](vector<unsigned > clusterIDS) {

                DataType dist_square = 0.0;
                for (int codebookIndex = 0; codebookIndex < num_codebook_; ++codebookIndex) {

                    KMeans<DataType>& currentKMeans = kMeans[codebookIndex];
                    const Cluster<DataType>& cluster = currentKMeans.getClusters()[clusterIDS[codebookIndex]];

                    DataType query_center = metric::euclidDistance(
                            &queryPoint[codebookIndex*codebook_max_size],
                            cluster.getCentralValues().data(),
                            codebook_size[codebookIndex]
                    );
                    DataType radius = cluster.getRadius();
                    DataType dist_lower_bound = query_center * query_center + radius * radius - 2 * query_center * radius;

                    dist_square += dist_lower_bound;
                }
                return dist_square;
            };
            IMISequence imiSequence(num_codebook_, codebook_size, distor);

            long double upperBound = std::numeric_limits<long double>::max();
            long double lowerBound = - std::numeric_limits<long double>::max();
            priority_queue<DistDataMax<Point<DataType> > >& maxHeap = maxHeaps[point_id];
            maxHeap.push( DistDataMax<Point<DataType> >(std::numeric_limits<DataType>::max(), points[0][0] ) );

            for (int bucketNum = 0; lowerBound<upperBound && imiSequence.hasNext(); bucketNum++) {
                auto next = imiSequence.next();
                const Cluster<DataType>& nextCluster = tables.find(lshbox::to_long(next.second, codebook_max_size))->second;

                DistDataMax<Point<DataType>> kDist = maxHeap.top();

                for (int i = 0; i < nextCluster.getClusterSize(); ++i) {
                    DataType distance = metric::euclidDistance(nextCluster.getPoint(i).getValues(), queryPoint, (size_t)query_.getDim());

                    if (distance < kDist.dist()) {
                        maxHeap.pop();
                        maxHeap.push(DistDataMax<Point<DataType> >(distance, nextCluster.getPoint(i)));
                        kDist = maxHeap.top();
                    }
                }

                upperBound = metric::squareEuclidDistance(data_[kDist.data().getID()], queryPoint, (size_t)query_.getDim());
                lowerBound = next.first;

            }

        }
    }

    void buildIndex() {

        kMeans.reserve(num_codebook_);

        for (int i = 0; i < num_codebook_; ++i) {

            kMeans.push_back( KMeans<DataType> (clusterK_, (size_t)data_.getSize(), codebook_size[i], max_iterations_, metric::squareEuclidDistance<DataType>) );
        }


        for (int codebook_index = 0; codebook_index < num_codebook_; ++codebook_index) {

            vector<Point<DataType> > subPoints;
            subPoints.reserve((size_t)data_.getSize());
            for (int point_id = 0; point_id < data_.getSize(); ++point_id) {

                subPoints.push_back(Point<DataType>(point_id, & data_[point_id][codebook_index * codebook_max_size]));
            }
            points.push_back(subPoints);
        }


        for (int i = 0; i < num_codebook_; ++i) {
            KMeans<DataType>& means = kMeans[i];

            means.run(points[i]);
            means.calculateRadius(metric::euclidDistance<DataType>);
        }

    }

    void mergeProduct() {

        // merge codebooks
        std::function<void (size_t, Cluster<DataType>& )> merger;

        merger = [&](size_t codebook_index, Cluster<DataType>& cluster) {

            if (codebook_index == -1) {

                tables.emplace(std::make_pair(cluster.getID(), cluster));
            } else {

                KMeans<DataType>& subKMeans = kMeans[codebook_index];
                for (int i = 0; i < clusterK_; ++i) {

                    Cluster<DataType> mergedCluster = cluster.merge( subKMeans.getClusters()[i], clusterK_ );
                    merger(codebook_index-1, mergedCluster );
                }
                std::cout << std::endl;
            }

        };

        Cluster<DataType> emptyCluster(0L, vector<DataType>());
        merger(num_codebook_-1, emptyCluster);

    }

    void calculateCodeBook() {
        codebook_max_size = data_.getDim() / num_codebook_;
        codebook_size.insert(codebook_size.end(), num_codebook_, codebook_max_size);
        if (data_.getDim() % num_codebook_) {
            codebook_size[num_codebook_-1] = data_.getDim() % num_codebook_;
        }
    }


protected:
    lshbox::Matrix<DataType>& data_;
    lshbox::Matrix<DataType>& query_;

    std::unordered_map<unsigned long ,  Cluster<DataType> > tables;

    vector<KMeans<DataType> > kMeans;
    vector<priority_queue<DistDataMax<Point<DataType> > > > maxHeaps;
    vector<vector<Point<DataType > > > points;

    size_t num_codebook_ ; // how much codebook
    size_t clusterK_ ;
    size_t max_iterations_;
    int topK_;

    size_t codebook_max_size;
    vector<size_t > codebook_size;
};