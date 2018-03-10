#include <iostream>

#include "include/util.h"
#include "include/kmeans.h"
#include "include/metric.h"

using namespace std;

template <typename DataType>
int execute(const string& dataFile, const string& queryFile) {

    // 1. load retrieved data and query data
    lshbox::Matrix<DataType> data;
    lshbox::Matrix<DataType> query;
    lshbox::loadFvecs(data, dataFile);
    lshbox::loadFvecs(query, queryFile);

    // 2. build indexes with PQ
    // 2-dimension code, which means size of product quantization = 2
    size_t size_pq = 2;
    size_t K = 2;
    size_t max_iterations = 100;
    size_t sub_dimension = data.getDim() / size_pq;
    // reserve size = size_pq
    vector<size_t > dimensions(size_pq, sub_dimension);
    // make sure sum of all dimensions equal data's origin dimension.
    if (data.getDim() % size_pq) {
        dimensions[size_pq-1] = data.getDim() % size_pq;
    }

    vector<KMeans<DataType> > subKMeans;
    subKMeans.reserve(size_pq);
    for (int i = 0; i < size_pq; ++i) {
        subKMeans.push_back( KMeans<DataType> (K, (size_t)data.getSize(), dimensions[i], max_iterations, metric::euclidDistance<DataType>) );
    }
    vector<vector<Point<DataType > > > points;
    points.reserve(size_pq);

    for (int sub_vector_index = 0; sub_vector_index < size_pq; ++sub_vector_index) {

        vector<Point<DataType> > subPoints;
        subPoints.reserve((size_t)data.getSize());
        for (int point_id = 0; point_id < data.getSize(); ++point_id) {

            subPoints.push_back(Point<DataType>(point_id, & data[point_id][sub_vector_index * sub_dimension]));
        }
        points.push_back(subPoints);
    }
    for (int i = 0; i < size_pq; ++i) {
        subKMeans[i].run(points[i]);
    }

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