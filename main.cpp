#include <iostream>
#include <queue>
#include <unordered_map>

#include "include/util.h"
#include "include/kmeans.h"
#include "include/metric.h"
#include "include/imisequence.h"


using namespace std;


template <typename DataType>
void preProcess(lshbox::Matrix<DataType>& data, lshbox::Matrix<DataType>& query) {
    // 2.1 mean
    // 2.2 eigen allocate or optimized product quantization
}



template <typename DataType>
void buildIndex(lshbox::Matrix<DataType>& data, lshbox::Matrix<DataType>& query) {
    size_t num_codebook = 2; // how much codebook
    size_t K = 4;
    size_t max_iterations = 100;

    size_t codebook_default_dimension = data.getDim() / num_codebook;
    // dimensions mean the dimension in each codebook
    vector<size_t > codebook_dimensions(num_codebook, codebook_default_dimension);
    if (data.getDim() % num_codebook) {
        codebook_dimensions[num_codebook-1] = data.getDim() % num_codebook;
    }

    vector<KMeans<DataType> > kMeans;
    for (int i = 0; i < num_codebook; ++i) {

        kMeans.push_back( KMeans<DataType> (K, (size_t)data.getSize(), codebook_dimensions[i], max_iterations, metric::squareEuclidDistance<DataType>) );
    }

    vector<vector<Point<DataType > > > points;
    for (int codebook_index = 0; codebook_index < num_codebook; ++codebook_index) {

        vector<Point<DataType> > subPoints;
        subPoints.reserve((size_t)data.getSize());
        for (int point_id = 0; point_id < data.getSize(); ++point_id) {

            subPoints.push_back(Point<DataType>(point_id, & data[point_id][codebook_index * codebook_default_dimension]));
        }
        points.push_back(subPoints);
    }


    for (int i = 0; i < num_codebook; ++i) {
        KMeans<DataType>& means = kMeans[i];

        means.run(points[i]);
        means.calculateRadius(metric::euclidDistance<DataType>);
    }

    // merge codebooks
    std::unordered_map<unsigned long ,  Cluster<DataType> > tables;
    std::function<void (size_t, Cluster<DataType>& )> merger;

    merger = [&](size_t codebook_index, Cluster<DataType>& cluster) {

        if (codebook_index == -1) {

            tables.emplace(std::make_pair(cluster.getID(), cluster));
        } else {

            KMeans<DataType>& subKMeans = kMeans[codebook_index];
            for (int i = 0; i < K; ++i) {

                Cluster<DataType> mergedCluster = cluster.merge( subKMeans.getClusters()[i], K );
                merger(codebook_index-1, mergedCluster );
            }
            std::cout << std::endl;
        }

    };

    Cluster<DataType> emptyCluster(0L, vector<DataType>());
    merger(num_codebook-1, emptyCluster);


    vector<priority_queue<DistDataMax<Point<DataType> > > > maxHeaps(query.getSize());

    for (int point_id = 0; point_id < query.getSize(); ++point_id) {

        DataType* queryPoint = query[point_id];

        auto distor = [&](vector<unsigned > clusterIDS) {

            DataType dist_square = 0.0;
            for (int codebookIndex = 0; codebookIndex < num_codebook; ++codebookIndex) {

                KMeans<DataType>& currentKMeans = kMeans[codebookIndex];
                const Cluster<DataType>& cluster = currentKMeans.getClusters()[clusterIDS[codebookIndex]];

                DataType query_center = metric::euclidDistance(
                        &queryPoint[codebookIndex*codebook_default_dimension],
                        cluster.getCentralValues().data(),
                        codebook_dimensions[codebookIndex]
                );
                DataType radius = cluster.getRadius();
                DataType dist_lower_bound = query_center * query_center + radius * radius - 2 * query_center * radius;

                dist_square += dist_lower_bound;
            }
            return dist_square;
        };
        IMISequence imiSequence(num_codebook, codebook_dimensions, distor);

        long double upperBound = std::numeric_limits<long double>::max();
        long double lowerBound = std::numeric_limits<long double>::min();
        priority_queue<DistDataMax<Point<DataType> > >& maxHeap = maxHeaps[point_id];
        maxHeap.push( DistDataMax<Point<DataType> >(std::numeric_limits<DataType>::max(), points[0][0] ) );

        for (int bucketNum = 0; lowerBound<upperBound && imiSequence.hasNext(); bucketNum++) {
            auto next = imiSequence.next();
            const Cluster<DataType>& nextCluster = tables.find(lshbox::to_long(next.second, codebook_default_dimension))->second;

            DistDataMax<Point<DataType>> kDist = maxHeap.top();

            for (int i = 0; i < nextCluster.getClusterSize(); ++i) {
                DataType distance = metric::euclidDistance(nextCluster.getPoint(i).getValues(), queryPoint, (size_t)query.getDim());

                if (distance < kDist.dist()) {
                    maxHeap.pop();
                    maxHeap.push(DistDataMax<Point<DataType> >(distance, nextCluster.getPoint(i)));
                    kDist = maxHeap.top();
                }
            }

            upperBound = metric::squareEuclidDistance(data[kDist.data().getID()], queryPoint, (size_t)query.getDim());
            lowerBound = next.first;

        }

    }


    string lshboxBenchFileName = "nn.lshbox";
    // lshbox file
    ofstream lshboxFout(lshboxBenchFileName);
    if (!lshboxFout) {
        cout << "cannot create output file " << lshboxBenchFileName << endl;
    }
    lshboxFout << maxHeaps.size() << "\t" << K << endl;
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

template <typename DataType>
int execute(const string& dataFile, const string& queryFile) {

    lshbox::Matrix<DataType> data;
    lshbox::Matrix<DataType> query;

    // 1. load retrieved data and query data
    lshbox::loadFvecs(data, dataFile);
    lshbox::loadFvecs(query, queryFile);

    // 2. pre process
    preProcess(data, query);

    // 3. build indexes with PQ
    // 2-dimension code, which means size of product quantization = 2
    buildIndex(data, query);

    // 3. query
    
}

template <typename DataType>
int kMeansTest() {

    srand(0);

    size_t total_points, dimensions, K, max_iterations, has_name;
    cin >> total_points >> dimensions >> K >> max_iterations >> has_name;

    vector<Point<DataType > > points;
    string point_name;

    vector<DataType> values;

    values.reserve(total_points*dimensions +1);
    points.reserve(total_points+1);

    for(int i = 0; i < total_points; i++) {

        for(int j = 0; j < dimensions; j++) {
            DataType value;
            cin >> value;
            values.push_back(value);
        }
        if (has_name) {
            cin >> point_name;
        }
    }

    for (int i = 0; i < total_points; ++i) {
        Point<DataType >  p(i, & ( values[ dimensions * i]) );
        points.push_back(p);
    }

    KMeans<DataType >  kMeans(K, total_points, dimensions, max_iterations, metric::euclidDistance<DataType>);
    kMeans.run(points);

}


int main(int argc, char *argv[]) {

//    kMeansTest<double>();

    string dataFile = "/home/xinyan/programs/data/audio/audio_base.fvecs";
    string queryFile = "/home/xinyan/programs/data/audio/audio_query.fvecs";

    execute<float >(dataFile, queryFile);

    return 0;
}