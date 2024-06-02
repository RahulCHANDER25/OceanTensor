#pragma once

#include "../Tensor/OceanTensor.hpp"
#include <fstream>
#include <ios>

namespace Network {
    /// @brief Interface for all the Neural Networks
    class INeuralNetwork {
        public:
        INeuralNetwork() = default;

        ~INeuralNetwork() = default;

        virtual Matrix2f forward(Matrix2f &inputs) = 0;

        virtual void save(const std::string &, std::_Ios_Openmode mode=std::ios::app) = 0;

        virtual void load(std::ifstream &) = 0;

    //    virtual void backward(OceanTensor::myTensor<double, 2> &error) = 0;
    };
}
