#pragma once
#include <cmath>
#include <iostream>

namespace metric {

    template <typename DataType>
    DataType squareEuclidDistance(const DataType *vec1, const DataType *vec2, size_t dim_) {
        DataType dist_ = 0.0;
        for (int i = 0; i < dim_; ++i) {
            dist_ += (vec1[i]-vec2[i]) * (vec1[i]-vec2[i]);
        }
        return dist_;
    }

    template <typename DataType>
    DataType euclidDistance(const DataType* vec1, const DataType * vec2, size_t dim_) {
        return sqrt(squareEuclidDistance(vec1, vec2, dim_));
    }

    template <typename DataType>
    DataType innerProductDistance(const DataType* vec1, const DataType * vec2, size_t dim_) {
        DataType dist_ = 0.0;
        for (unsigned i = 0; i != dim_; ++i)
        {
            dist_ += vec1[i] * vec2[i];
        }
        return - dist_;
    }

    template <typename DataType>
    DataType angularDistance(const DataType* vec1, const DataType * vec2, size_t dim_) {
        DataType dist_ = 0.0;
        float norm_1 = 0.0;
        float norm_2 = 0.0;
        for (unsigned i = 0; i != dim_; ++i)
        {
            dist_ += vec1[i] * vec2[i];
            norm_1 += vec1[i] *  vec1[i];
            norm_2 += vec2[i] * (vec2[i]);
        }
        return acos( dist_ / std::sqrt(norm_1*norm_2) );
    }

    template <typename DataType>
    DataType consineDistance(const DataType* vec1, const DataType * vec2, size_t dim_) {
        DataType dist_ = 0.0;
        float norm_1 = 0.0;
        float norm_2 = 0.0;
        for (unsigned i = 0; i != dim_; ++i)
        {
            dist_  += vec1[i] * vec2[i];
            norm_1 += vec1[i] * vec1[i];
            norm_2 += vec2[i] * vec2[i];
        }
        return - ( dist_ / std::sqrt(norm_1*norm_2) );
    }
}