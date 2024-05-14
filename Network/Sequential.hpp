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


        virtual void backward(OceanTensor::myTensor<double, 2> &target)
        {
            auto output = m_layers[m_layers.size() - 1].m_output;
            auto delta = (output - target);
            delta *= ((output * (-1)) + 1) * output;

            auto size = m_layers.size() - 1;
            for (int i = m_layers.size() - 1; i >= 0; i--) {
                auto &curr = m_layers[i];


                if (i == size) {
                    auto &shape = curr.m_weight.getMetadata().shape();
                    delta = delta.replicate({shape[0], shape[1]}).transposed();
                }

                // std::cout << delta << std::endl;
                // Grad for weight
                curr.m_gradW = curr.m_input * delta;
                curr.m_gradW *= 0.5;

                delta = delta.transposed();
                delta *= curr.m_weight.transposed();
                std::cout << "Current Iteration: " << std::endl;
                std::cout << curr.m_weight << std::endl;
                std::cout << curr.m_weight - curr.m_gradW << std::endl;
            }
        }

    private:
        std::vector<LinearNetwork> m_layers;
    };
}
