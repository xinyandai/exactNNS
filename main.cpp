#include <iostream>

#include "include/util.h"
#include "include/kmeans.h"

using namespace std;

template <typename DATATYPE>
int execute(const string& baseFormat, const string& dataFile, const string& queryFile) {

    // 1. load retrieved data and query data
    lshbox::Matrix<DATATYPE> data;
    lshbox::Matrix<DATATYPE> query;
    if (baseFormat == "fvecs") {
        lshbox::loadFvecs(data, dataFile);
        lshbox::loadFvecs(query, queryFile);
    } else {
        std::cout << "wrong input data format " << baseFormat << std::endl;
    }

    // 2. build indexes with PQ


    // 3. query

}


int kmeansTest() {
    srand(0);

    size_t total_points, dimensions, K, max_iterations, has_name;
    cin >> total_points >> dimensions >> K >> max_iterations >> has_name;

    vector<Point> points;
    string point_name;

    vector<double> values;

    values.reserve(total_points*dimensions +1);
    points.reserve(total_points+1);

    for(int i = 0; i < total_points; i++) {

        for(int j = 0; j < dimensions; j++) {
            double value;
            cin >> value;
            values.push_back(value);
        }
        if (has_name) {
            cin >> point_name;
        }
    }


    for (int i = 0; i < total_points; ++i) {
        Point p(i, & ( values[ dimensions * i]) );
        points.push_back(p);
    }


    KMeans kmeans(K, total_points, dimensions, max_iterations);
    kmeans.run(points);


    std::cout <<  std::endl;
}


int main(int argc, char *argv[]) {

    kmeansTest();
    return 0;
}