//
// Created by xinyan on 3/14/18.
//
#include "kmeans_abstract.h"

template <typename DataType>
class StandardKMeans : public AbstractKMeans<DataType> {

public:
    /**
     *
     * @param K
     * @param num_points        the number of point in train data.
     * @param dimension         dimension of each point.
     * @param max_iterations
     * @param distance          a function to calculate distance between point and center, and is used to determine which cluster the point belongs to.
     */
    StandardKMeans(size_t K,
           size_t num_points,
           size_t dimension,
           size_t max_iterations,
           std::function<DataType (const DataType*, const DataType*, size_t) > distance)
            : AbstractKMeans<DataType>(K, num_points, dimension, max_iterations, distance)     {
    }


protected:

    /**
      * recalculating the center of each cluster
      */
    virtual void recenter() {
        for(int i = 0; i < AbstractKMeans<DataType>::K_; i++) {
            size_t total_points_cluster = AbstractKMeans<DataType>::clusters_[i].getClusterSize();

            for(int j = 0; j < AbstractKMeans<DataType>::dimension_; j++) {

                DataType sum = 0.0;

                if(total_points_cluster > 0) {

                    for(int p = 0; p < total_points_cluster; p++) {
                        sum += AbstractKMeans<DataType>::clusters_[i].getPoint(p).getValue(j);
                    }
                    AbstractKMeans<DataType>::clusters_[i].setCentralValue(j, sum / total_points_cluster);
                }
            }
        }
    }


    /**
     * return ID of nearest center (uses euclidean distance)
     * @param point
     * @return
     */
    virtual int chooseCenter(Point<DataType >&  point) {

        DataType min_dist;
        int id_cluster_center = 0;

        min_dist = AbstractKMeans<DataType>::distor_(AbstractKMeans<DataType>::clusters_[0].getCentralValues().data(), point.getValues(), AbstractKMeans<DataType>::dimension_);

        for(int i = 1; i < AbstractKMeans<DataType>::K_; i++) {
            DataType dist = AbstractKMeans<DataType>::distor_(AbstractKMeans<DataType>::clusters_[i].getCentralValues().data(), point.getValues(), AbstractKMeans<DataType>::dimension_);

            if(dist < min_dist) {
                min_dist = dist;
                id_cluster_center = i;
            }
        }

        return id_cluster_center;
    }
};