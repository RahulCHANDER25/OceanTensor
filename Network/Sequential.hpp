#pragma once

#include "LinearNetwork.hpp"
#include "NeuralNetwork.hpp"

#include <initializer_list>
#include <vector>

namespace Network {
    /// @brief Fully connected layer of a neural network
    class Sequential: public INeuralNetwork {
    public:
        Sequential() = default;

        Sequential(std::initializer_list<LinearNetwork> linears): m_layers()
        {
            for (auto lin: linears) {
                m_layers.push_back(lin);
            }
        }

        ~Sequential() = default;

        virtual OceanTensor::myTensor<double, 2> forward(OceanTensor::myTensor<double, 2> &inputs) override
        {
            auto out = inputs;

            for (auto &layer : m_layers) {
                out = layer.forward(out);
            }
            return out;
        }

    private:
        std::vector<LinearNetwork> m_layers;
    };
}
