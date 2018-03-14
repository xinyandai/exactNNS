//
// Created by xinyan on 3/14/18.
//
#include "abstractkmeans.h"

template <typename DataType>
class KBalls : public AbstractKMeans<DataType> {

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
            : AbstractKMeans<DataType>(K, num_points, dimension, max_iterations, distance)     {
        std::cout << "k balls !!" << std::endl;
    }


protected:

    /**
     * recalculating the center of each cluster
     */
    void recenter() {
        for(int i = 0; i < KBalls<DataType>::K_; i++) {

            Cluster<DataType>& cluster = KBalls<DataType>::clusters_[i];

            vector<const DataType * const > points;
            for (int j = 0; j < cluster.getClusterSize(); ++j) {
                points.push_back(cluster.getPoint(j).getValues());
            }

            // define the types of iterators through the points and their coordinates
            // ----------------------------------------------------------------------
            typedef typename vector<const DataType * const >::const_iterator PointIterator;
            typedef const DataType* CoordIterator;

            // create an instance of Miniball
            // ------------------------------
            typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > MiniBall;
            MiniBall mb ((int)KBalls<DataType>::dimension_, points.begin(), points.end());

        }
    }


};