//
// author: developed by Xinyan Dai
// date: Mar 9 2018
// based on: https://github.com/marcoscastro/kmeans/blob/master/kmeans.cpp
//

#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#define DEFAULT_CLUSTER_ID (-1)
using namespace std;

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

template <typename DataType>
class Cluster
{

protected:
    int cluster_id_;
    vector<DataType> central_values_;
    vector<Point<DataType > > points_;

public:
    Cluster(int id_cluster, const Point<DataType > & point, size_t dimension)
            : cluster_id_(id_cluster), points_(1, point) {

        central_values_.reserve(dimension);
        for (size_t i = 0; i < dimension; ++i) {
            central_values_.push_back(point.getValues()[i]);
        }
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

    size_t getClusterSize() const {
        return points_.size();
    }

    int getID() const {
        return cluster_id_;
    }

};

template <typename DataType>
class KMeans
{
protected:
    size_t K_; // number of clusters
    size_t dimension_;
    size_t num_points_;
    size_t max_iterations_;
    vector<Cluster<DataType > > clusters_;

    /**
     * return ID of nearest center (uses euclidean distance)
     * @param point
     * @return
     */
    int getIDNearestCenter(Point<DataType >  point) {
        DataType sum = 0.0;
        DataType min_dist;
        int id_cluster_center = 0;

        for(int i = 0; i < dimension_; i++) {
            sum += pow(clusters_[0].getCentralValue(i) - point.getValue(i), 2.0);
        }

        min_dist = sqrt(sum);

        for(int i = 1; i < K_; i++) {
            DataType dist;
            sum = 0.0;

            for(int j = 0; j < dimension_; j++) {
                sum += pow(clusters_[i].getCentralValue(j) -
                           point.getValue(j), 2.0);
            }

            dist = sqrt(sum);

            if(dist < min_dist) {
                min_dist = dist;
                id_cluster_center = i;
            }
        }

        return id_cluster_center;
    }

    /**
     * choose K distinct values for the centers of the clusters
     * @param points
     * @param prohibited_indexes
     */
    void initialCenters(vector<Point<DataType > > & points, vector<size_t >& prohibited_indexes) {

        for(int i = 0; i < K_; i++) {

            while(true) {

                size_t index_point = rand() % num_points_;

                if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
                        index_point) == prohibited_indexes.end()) {

                    prohibited_indexes.push_back(index_point);
                    points[index_point].setCluster(i);

                    Cluster<DataType >  cluster(i, points[index_point], dimension_);
                    clusters_.push_back(cluster);
                    break;
                }
            }
        }
    }

    /**
     * recalculating the center of each cluster
     */
    void recenter() {
        for(int i = 0; i < K_; i++) {
            size_t total_points_cluster = clusters_[i].getClusterSize();

            for(int j = 0; j < dimension_; j++) {

                DataType sum = 0.0;

                if(total_points_cluster > 0) {

                    for(int p = 0; p < total_points_cluster; p++) {
                        sum += clusters_[i].getPoint(p).getValue(j);
                    }
                    clusters_[i].setCentralValue(j, sum / total_points_cluster);
                }
            }
        }
    }

    /**
     * @param points
     * @return true if no action performed
     */
    bool associate(vector<Point<DataType > > & points) {
        bool done = true;

        // associates each point to the nearest center
        for(int i = 0; i < num_points_; i++) {
            int id_old_cluster = points[i].getCluster();
            int id_nearest_center = getIDNearestCenter(points[i]);

            if(id_old_cluster != id_nearest_center) {

                if(id_old_cluster != DEFAULT_CLUSTER_ID) {
                    clusters_[id_old_cluster].removePoint(points[i].getID());
                }

                points[i].setCluster(id_nearest_center);
                clusters_[id_nearest_center].addPoint(points[i]);
                done = false;
            }
        }

        return done;
    }

    /**
     * shows elements of clusters
     */
    void showClusters() {

        for(int i = 0; i < K_; i++) {

            size_t total_points_cluster =  clusters_[i].getClusterSize();

            cout << "Cluster " << clusters_[i].getID() + 1 << endl;

            for(int j = 0; j < total_points_cluster; j++) {
                cout << "Point " << clusters_[i].getPoint(j).getID() + 1 << ": ";
                for(int p = 0; p < dimension_; p++)
                    cout << clusters_[i].getPoint(j).getValue(p) << " ";
                cout << endl;
            }

            cout << "Cluster values: ";

            for(int j = 0; j < dimension_; j++)
                cout << clusters_[i].getCentralValue(j) << " ";

            cout << "\n\n";
        }
    }


public:
    KMeans(size_t K, size_t num_points, size_t dimension, size_t max_iterations)
            : K_(K), num_points_(num_points), dimension_(dimension), max_iterations_(max_iterations) {}

    void run(vector<Point<DataType > > & points) {

        // check requirement
        if(K_ > num_points_){
            std::cout << "K_ is bigger than number of points" << std::endl;
            return;
        }

        // initial center
        vector<size_t > prohibited_indexes;
        initialCenters(points, prohibited_indexes);


        for (int iter = 0; iter < max_iterations_; ++iter) {

            showClusters();

            cout << "============================================\n"
                 << "iteration " << iter+1 << "\n";

            // k clusters are created by associating every observation with the nearest mean.
            if (associate(points)) {
                break;
            }

            // The centroid of each of the k clusters becomes the new mean.
            recenter();

        }

        showClusters();
    }

};