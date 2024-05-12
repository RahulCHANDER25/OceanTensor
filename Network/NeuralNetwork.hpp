#pragma once

#include "../Tensor/OceanTensor.hpp"

namespace Network {
    /// @brief Interface for all the Neural Networks
    class INeuralNetwork {
        public:
        INeuralNetwork() = default;

        ~INeuralNetwork() = default;

        virtual OceanTensor::myTensor<double, 2> forward(OceanTensor::myTensor<double, 2> &inputs) = 0;

    //    virtual void backward(OceanTensor::myTensor<double, 2> &error) = 0;
    };
}
