#pragma once

#include "OceanTensor.hpp"
#include <cmath>

namespace Act {
    template <int DIM>
    OceanTensor::myTensor<double, DIM> sigmoid(
        const OceanTensor::myTensor<double, DIM> &tensor,
        bool derivative=false
    )
    {

        OceanTensor::myTensor<double, DIM> outTensor(tensor);

        for (size_t i = 0; i < outTensor.size(); i++) {
            outTensor[i] = 1.0 / (1.0 + std::exp((-1.0) * tensor[i]));
        }
        if (derivative) {
            return (outTensor * (-1) + 1) * outTensor;
        }
        return outTensor;
    }
}

template OceanTensor::myTensor<double, 2> Act::sigmoid(const OceanTensor::myTensor<double, 2> &, bool);
