#pragma once

#include "OceanTensor.hpp"

#include <cmath>
#include <stdexcept>

namespace Loss {
    template <int DIM>
    OceanTensor::myTensor<double, DIM> squaredLoss(
        OceanTensor::myTensor<double, DIM> &pred,
        OceanTensor::myTensor<double, DIM> &y,
        bool derivative=false
    )
    {
        OceanTensor::myTensor<double, DIM> newTensor(pred);
        auto &meta_pred = newTensor.getMetadata();
        auto &meta_y = y.getMetadata();

        if (!meta_pred.isEqual(meta_y.shape()))
            throw std::runtime_error("Shape not equal.");
        for (size_t i = 0; i < newTensor.size(); i++) {
            newTensor[i] = std::pow(newTensor[i] - y[i], 2) * 0.5;
        }
        return newTensor;
    }
}
