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
            return ((outTensor * (-1)) + 1) * outTensor;
        }
        return outTensor;
    }

    template <int DIM>
    OceanTensor::myTensor<double, DIM> relu(
        const OceanTensor::myTensor<double, DIM> &tensor,
        bool derivative=false
    )
    {
        OceanTensor::myTensor<double, DIM> outTensor(tensor);

        if (derivative) {
            for (size_t i = 0; i < outTensor.size(); i++)
                outTensor[i] = outTensor[i] > 0;
        } else {
            for (size_t i = 0; i < outTensor.size(); i++)
                outTensor[i] = std::max(outTensor[i], 0.0);
        }
        return outTensor;
    }

    template <int DIM>
    OceanTensor::myTensor<double, DIM> tanh(
        const OceanTensor::myTensor<double, DIM> &tensor,
        bool derivative=false
    )
    {
        OceanTensor::myTensor<double, DIM> outTensor(tensor);

        if (derivative) {
            for (size_t i = 0; i < outTensor.size(); i++)
                outTensor[i] = 1 - std::pow(std::tanh(outTensor[i]), 2);
        } else {
            for (size_t i = 0; i < outTensor.size(); i++)
                outTensor[i] = std::tanh(outTensor[i]);
        }
        return outTensor;
    }
}
