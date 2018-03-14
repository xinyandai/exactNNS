//
// Created by xinyan on 3/14/18.
//

#pragma once


#define DEFAULT_CLUSTER_ID (-1)

template <typename DataType>
class Point
{
protected:
    int cluster_id_;
    int point_id_;
    /**
     * one line of data : vector
     */
    const DataType * values_;

public:
    Point(int id_point, const DataType *  values)
            : point_id_(id_point),  values_(values), cluster_id_(DEFAULT_CLUSTER_ID) {}

    Point(const Point<DataType > & p): point_id_(p.point_id_), values_(p.values_), cluster_id_(p.cluster_id_) {}

    void setCluster(int id_cluster) {
        this->cluster_id_ = id_cluster;
    }

    int getCluster() const {
        return cluster_id_;
    }

    int getID() const {
        return point_id_;
    }

    DataType getValue(int index) const {
        return values_[index];
    }

    const DataType * const  getValues() const {
        return values_;
    }
};


