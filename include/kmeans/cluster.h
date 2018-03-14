//
// Created by xinyan on 3/14/18.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <functional>
#include <climits>

#include "point.h"

using namespace std;

template <typename DataType>
class Cluster
{

protected:

    size_t cluster_id_;
    /**
     * maximun distance from any point to center
     */
    DataType radius_;
    /**
     * center of this cluster
     */
    vector<DataType> central_values_;
    /**
     * all point belongs to this cluster.
     */
    vector<Point<DataType > > points_;

public:
    /**
     *
     * @param id_cluster
     * @param point the point we used to init cluster.
     * @param dimension the dimension of center / point.
     */
    Cluster(size_t id_cluster, const Point<DataType > & point, size_t dimension)
            : cluster_id_(id_cluster), points_(1, point) {

        // initial center with "point"
        central_values_.reserve(dimension);
        for (size_t i = 0; i < dimension; ++i) {
            central_values_.push_back(point.getValues()[i]);
        }

    }

    Cluster(size_t id, const vector<DataType>& central_values):cluster_id_(id), central_values_(central_values) {}

    DataType calculateRadius(std::function<DataType (const DataType*, const DataType*, size_t) > distor) {

        radius_ = - std::numeric_limits<double>::max();

        for (int i = 0; i < points_.size(); ++i) {
            DataType dist = distor(points_[i].getValues(), central_values_.data(), central_values_.size());
            if (dist > radius_) {
                radius_ = dist;
            }
        }

        return radius_;
    }

    /**
    *
    * @param cluster
    * @param decimal
    * @return
    */
    Cluster merge(const Cluster& cluster, size_t decimal) const {
        vector<DataType> mergedCenter;

        mergedCenter.insert(mergedCenter.end(), this->central_values_.begin(), this->central_values_.end());
        mergedCenter.insert(mergedCenter.end(), cluster.central_values_.begin(), cluster.central_values_.end());

        Cluster mergedCluster = Cluster(this->cluster_id_ * decimal + cluster.cluster_id_, mergedCenter);

        for (int j = 0; j < this->getClusterSize(); ++j) {
            for (int i = 0; i < cluster.getClusterSize(); ++i) {
                if (cluster.getPoint(i).getID() == this->getPoint(j).getID()) {

                    mergedCluster.addPoint(this->getPoint(j));

                    // next j
                    ++j;
                    // initialize i = 0 after ++i
                    i = -1;
                }
            }
        }

        return mergedCluster;
    }

    DataType getRadius() const {
        return radius_;
    }

    void addPoint(const Point<DataType > & point) {
        points_.push_back(point);
    }

    bool removePoint(int id_point) {

        for(int i = 0; i < points_.size(); i++) {
            if(points_[i].getID() == id_point) {

                points_.erase(points_.begin() + i);
                return true;
            }
        }

        return false;
    }

    DataType getCentralValue(int index) const {
        return central_values_[index];
    }

    const vector<DataType >& getCentralValues() const {
        return central_values_;
    }

    void setCentralValue(int index, DataType value) {
        central_values_[index] = value;
    }

    const Point<DataType > & getPoint(int index) const {
        return points_[index];
    }

    const vector<Point<DataType > >& getPoints() const {
        return points_;
    }

    size_t getClusterSize() const {
        return points_.size();
    }

    size_t getID() const {
        return cluster_id_;
    }

};
