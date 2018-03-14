//
// Created by xinyan on 3/14/18.
//
#include "kmeans_standard.h"

template <typename DataType>
class KBalls : public StandardKMeans<DataType> {

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
            : StandardKMeans<DataType>(K, num_points, dimension, max_iterations, distance)     {

    }


protected:

    /**
     * recalculating the center of each cluster
     */
    virtual void recenter() {

        for(int i = 0; i < AbstractKMeans<DataType>::K_; i++) {

            Cluster<DataType>& cluster = AbstractKMeans<DataType>::clusters_[i];

            vector<const DataType * > points;
            for (int j = 0; j < cluster.getClusterSize(); ++j) {
                points.push_back(cluster.getPoint(j).getValues());
            }

            // define the types of iterators through the points and their coordinates
            // ----------------------------------------------------------------------
            typedef typename vector<const DataType * >::const_iterator PointIterator;
            typedef const DataType* CoordIterator;

            // create an instance of Miniball
            // ------------------------------
            typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > MiniBall;
            MiniBall mb ((int)AbstractKMeans<DataType>::dimension_, points.begin(), points.end());

            const DataType* center = mb.center();


            for (int k = 0; k < AbstractKMeans<DataType>::dimension_; ++k) {
                cluster.setCentralValue(k, center[k]);
            }
        }

        StandardKMeans<DataType>::calculateRadius();
    }


//    /**
//     *
//     * @param point
//     * @return
//     */
//    virtual int chooseCenter(Point<DataType >&  point) {
//        DataType min_dist;
//
//        int id_inner_radius  = -1;
//        int id_beyond_radius = -1;
//
//        DataType min_inner_dist = std::numeric_limits<DataType>::max();
//        DataType min_beyond_dist = std::numeric_limits<DataType>::max();
//
//
//        for(int i = 0; i < AbstractKMeans<DataType>::K_; i++) {
//
//            Cluster<DataType>& cluster = AbstractKMeans<DataType>::clusters_[i];
//            DataType dist = AbstractKMeans<DataType>::distor_(cluster.getCentralValues().data(), point.getValues(), AbstractKMeans<DataType>::dimension_);
//
//            if (dist < min_beyond_dist) {
//
//                min_beyond_dist = dist;
//                id_beyond_radius = i;
//            }
//
//            if (dist < min_inner_dist && dist < cluster.getRadius()) {
//                min_inner_dist = dist;
//                id_inner_radius = i;
//            }
//
//        }
//
//        return id_inner_radius == -1? id_beyond_radius : id_inner_radius;
//    }
//
//    /**
//     * use standard KMeans to initialize cluster centers.
//     * @param points
//     */
//    virtual void initialCenters(vector<Point<DataType > > & points) {
//
//        StandardKMeans<DataType>::initialCenters(points);
//
//        //iterate to run k-means: reallocate and re-calculate center
//        for (int iter = 0; iter < 2; ++iter) {
//
//            if (StandardKMeans<DataType>::associate(points)) {
//                break;
//            }
//            StandardKMeans<DataType>::recenter();
//        }
//
//        StandardKMeans<DataType>::calculateRadius();
//    }
};