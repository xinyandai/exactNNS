#include <iostream>
#include <queue>

#include "include/exactnns.h"


using namespace std;


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

template <typename DataType>
int execute(const string& dataFile, const string& queryFile) {

    lshbox::Matrix<DataType> data;
    lshbox::Matrix<DataType> query;


    lshbox::loadFvecs(data, dataFile);
    lshbox::loadFvecs(query, queryFile);


    size_t num_codebook = 2;
    size_t clusterK = 4;
    int topK = 10;
    size_t max_iteration = 100;

    ExactNNS<DataType>(data, query, num_codebook, clusterK, topK, max_iteration);

}


int main(int argc, char *argv[]) {

//    kMeansTest<double>();

    string dataFile = "/home/xinyan/programs/data/test/test_base.fvecs";
    string queryFile = "/home/xinyan/programs/data/test/test_query.fvecs";

    execute<float >(dataFile, queryFile);

    return 0;
}