//
// Created by xinyan on 3/8/18.
//

#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <queue>
#include <functional>
#include "util/heap_element.h"
using std::vector;
using std::pair;
using std::function;
using std::priority_queue;

// stands for inverted multi index
class IMISequence{
public:
    typedef vector<unsigned > Coord;
    IMISequence(
            unsigned dimension,
            const vector<unsigned > & lengths,
            const function<float (Coord)>& func)
            :  _dimension(dimension), _lengths(lengths) {

        distor_ = func;
        enHeap(vector<unsigned >(dimension, 0));

        unsigned point_size = 1;
        for (int i = 0; i < _dimension; ++i) {
            point_size *= _lengths[i];
        }
        visited_ = vector<bool >(point_size, false);
//        visited_[0] = true;
//  only the first element visited
    }

    bool hasNext() const {
        return !minHeap_.empty();
    }

    inline unsigned position(const Coord &coord) {
        unsigned index = coord[0];
        for (int i = 1; i < _dimension; ++i) {
            index *= _lengths[i-1];
            index += coord[i];
        }
        return index;
    }
    /**
     * time complexity O(_dimension)
     * @param coord
     * @return coord is already visited
     */
    inline bool visited(const Coord& coord) {
        return visited_[position(coord)];
    }

    /**
     *
     * time complexity O(_dimension*_dimension)
     * @param coord
     * @return if all point before coord is visited
     */
    inline bool shouldEnHeap(const Coord& coord) {

        for (int i = 0; i < _dimension; ++i) {
            if (coord[i]>=1) { // preCoord existed

                Coord preOne = coord;
                preOne[i]--;
                if (!visited(preOne)) {
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * time complexity O(_dimension*_dimension*_dimension)
     * @return
     */
    pair<float, Coord > next() {
        const Coord p = minHeap_.top().data();

        float dist = minHeap_.top().dist();
        minHeap_.pop();

        visited_[position(p)] = true;

        for (int i = 0; i < _dimension; ++i) {
            // check the next element in (i+1)'th dimension
            unsigned newIndex = p[i] + 1;

            if(newIndex  < _lengths[i]) { // next element exists

                Coord nextEnHeap = p;
                nextEnHeap[i] = newIndex;
                if (shouldEnHeap(nextEnHeap)) {
                    enHeap(nextEnHeap);
                }
            }
        }

        return std::make_pair(dist, p);
    }
private:
    // dist stores float value, data stores the index in the given leftVec or rightVec;

    unsigned  _dimension;
    vector<unsigned> _lengths;

    vector<bool > visited_;
    function<float (Coord)> distor_;
    priority_queue<DistDataMin<Coord>> minHeap_;

    void enHeap(const Coord& coord) {
        float dist = distor_(coord);
        minHeap_.emplace(DistDataMin<Coord>(dist, coord));
    }
};


