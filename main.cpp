#include <iostream>

#include "include/util.h"
#include "include/kmeans.h"
#include "include/metric.h"
#include "include/imisequence.h"
#include "include/heap_element.h"

using namespace std;

template <typename DataType>
void preProcess(lshbox::Matrix<DataType>& data, lshbox::Matrix<DataType>& query) {
    // 2.1 mean
    // 2.2 eigen allocate or optimized product quantization
}

template <typename DataType>
void buildIndex(lshbox::Matrix<DataType>& data, lshbox::Matrix<DataType>& query) {
    size_t num_codeword = 2; // how much codeword
    size_t K = 32;
    size_t max_iterations = 100;

    // dimensions mean the dimension in each codeword
    vector<size_t > codeword_dimensions(num_codeword, data.getDim() / num_codeword);
    if (data.getDim() % num_codeword) {
        codeword_dimensions[num_codeword-1] = data.getDim() % num_codeword;
    }

    vector<KMeans<DataType> > kMeans;
    for (int i = 0; i < num_codeword; ++i) {

        kMeans.push_back( KMeans<DataType> (K, (size_t)data.getSize(), codeword_dimensions[i], max_iterations, metric::euclidDistance<DataType>) );
    }

    vector<vector<Point<DataType > > > points;
    for (int codeword_index = 0; codeword_index < num_codeword; ++codeword_index) {

        vector<Point<DataType> > subPoints;
        subPoints.reserve((size_t)data.getSize());
        for (int point_id = 0; point_id < data.getSize(); ++point_id) {

            subPoints.push_back(Point<DataType>(point_id, & data[point_id][codeword_index * codeword_dimensions[0]]));
        }
        points.push_back(subPoints);
    }

    for (int i = 0; i < num_codeword; ++i) {
        kMeans[i].run(points[i]);
    }

    for (int point_id = 0; point_id < query.getSize(); ++point_id) {

        IMISequence imiSequence(num_codeword, codeword_dimensions, [&](vector<unsigned > clusterIDS) {

            DataType dist_square = 0.0;
            for (int codewordIndex = 0; codewordIndex < num_codeword; ++codewordIndex) {

                dist_square += metric::squareEuclidDistance(
                        & data[point_id][codewordIndex * codeword_dimensions[0]],
                        kMeans[codewordIndex].getCulsters()[clusterIDS[codewordIndex]].getCentralValues().data(),
                        codeword_dimensions[codewordIndex]
                );
            }
            return dist_square;
        });

        long double upperBound = 0;
        long double lowerBound = 0;
        Heap minHeap;

        while (lowerBound<upperBound && imiSequence.hasNext()) {
            minHeap.pushAll()
            upperBound = minHeap[K];
            lowerBound = currentbucket;
        }

    }
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