//
// author: developed by Xinyan Dai
// date: Mar 9 2018
// based on: https://github.com/marcoscastro/kmeans/blob/master/kmeans.cpp
//

#pragma once
#include "point.h"
#include "cluster.h"


using namespace std;


template <typename DataType>
class KMeans
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
    KMeans(size_t K,
           size_t num_points,
           size_t dimension,
           size_t max_iterations,
           std::function<DataType (const DataType*, const DataType*, size_t) > distance)
            :
            K_(K), num_points_(num_points), dimension_(dimension), max_iterations_(max_iterations), distor_(distance) {}

    void run(vector<Point<DataType > > & points) {

        // check requirement
        if(K_ > num_points_){
            std::cout << "K_ is bigger than number of points" << std::endl;
            return;
        }
        // initial center
        initialCenters(points);
        //iterate to run k-means: reallocate and re-calculate center
        for (int iter = 0; iter < max_iterations_; ++iter) {

            // k clusters are created by associating every observation with the nearest mean.
            if (associate(points)) {
                std::cout << "break in iteration : " << iter << std::endl;
                break;
            }
            // The centroid of each of the k clusters becomes the new mean.
            recenter();
        }

        for (int i = 0; i < clusters_.size(); ++i) {
            clusters_[i].calculateRadius(distor_);
        }

        showClusters();
    }

    void calculateRadius(std::function<DataType (const DataType*, const DataType*, size_t) > distor) {
        for (int j = 0; j < clusters_.size(); ++j) {
            Cluster<DataType>& cluster = clusters_[j];
            cluster.calculateRadius(distor);
        }
    }

    const vector<Cluster<DataType > > & getClusters() const {
        return clusters_;
    }
protected:
    size_t K_; // number of clusters
    size_t dimension_;
    size_t num_points_;
    size_t max_iterations_;
    vector<Cluster<DataType > > clusters_;
    std::function<DataType (const DataType*, const DataType*, size_t) > distor_;


    /**
     * return ID of nearest center (uses euclidean distance)
     * @param point
     * @return
     */
    size_t getIDNearestCenter(Point<DataType >&  point) {

        DataType min_dist;
        size_t id_cluster_center = 0;

        min_dist = distor_(clusters_[0].getCentralValues().data(), point.getValues(), dimension_);

        for(size_t i = 1; i < K_; i++) {
            DataType dist = distor_(clusters_[i].getCentralValues().data(), point.getValues(), dimension_);

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
     */
    void initialCenters(vector<Point<DataType > > & points) {

        vector<size_t > prohibited_indexes;

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
    virtual void recenter() {
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
     * re allocate each point to its nearest cluster center.
     * @param points
     * @return true if no action performed
     */
    virtual bool associate(vector<Point<DataType > > & points) {
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

            cout << "Cluster " << clusters_[i].getID() << "   \t";

            for(int j = 0; j < total_points_cluster; j++) {
                cout << clusters_[i].getPoint(j).getID() << " ";
            }
            cout << endl;

        }
        cout << "\n";
    }

};