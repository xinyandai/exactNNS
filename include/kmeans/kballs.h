//
// Created by xinyan on 3/14/18.
//

//
// author: developed by Xinyan Dai
// date: Mar 9 2018
// based on: https://github.com/marcoscastro/kmeans/blob/master/kmeans.cpp
//

#pragma once
#include "point.h"
#include "cluster.h"
#include "kmeans.h"
#include "../miniball/miniball.hpp"

using namespace std;


template <typename DataType>
class KBalls : KMeans
{
public:
    /**
     *
     * @param K
     * @param num_points        the number of point in train data.
     * @param dimension         dimension of each point.
     * @param max_iterations
     * @param distance          a function to calculate distance between point and center, and is used to determine which cluster the point belongs to.
     */
    KBalls(size_t K,
           size_t num_points,
           size_t dimension,
           size_t max_iterations,
           std::function<DataType (const DataType*, const DataType*, size_t) > distance)
            : KMeans(K, num_points, dimension, max_iterations, distance)     {
    }


protected:

    /**
     * recalculating the center of each cluster
     */
    void recenter() {
        for(int i = 0; i < K_; i++) {

            Cluster& cluster = clusters_[i];

            vector<const DataType * const > points;
            for (int j = 0; j < cluster.getClusterSize(); ++j) {
                points.push_back(cluster.getPoint(j).getValues());
            }

            // define the types of iterators through the points and their coordinates
            // ----------------------------------------------------------------------
            typedef vector<const DataType * const >::const_iterator PointIterator;
            typedef const DataType* CoordIterator;

            // create an instance of Miniball
            // ------------------------------
            typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > MiniBall;
            MiniBall mb ((int)dimension_, points.begin(), points.end());

        }
    }


};