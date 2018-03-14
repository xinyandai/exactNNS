//
// Created by xinyan on 3/14/18.
//
#include "kballs.h"



template <typename DataType>
class KMeans : public KBalls<DataType> {

public:
    /**
     *
     * @param K
     * @param num_points        the number of point in train data.
     * @param dimension         dimension of each point.
     * @param max_iterations
     * @param distance          a function to calculate distance between point and center, and is used to determine which cluster the point belongs to.
     */
    KMeans(size_t K,
           size_t num_points,
           size_t dimension,
           size_t max_iterations,
           std::function<DataType (const DataType*, const DataType*, size_t) > distance)

            : KBalls<DataType>(K, num_points, dimension, max_iterations, distance)     {

    }

};