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

            auto size = m_layers.size() - 1;
            for (int i = m_layers.size() - 1; i >= 0; i--) {

                auto &curr = m_layers[i];
                auto &shape = curr.m_weight.getMetadata().shape();

                if (i == size) { // First time we need to put it in correct shape.
                    delta = delta.replicate({shape[1], shape[0]}).transposed();
                }
                auto out = curr.m_output.replicate({shape[1], shape[0]}).transposed();

                delta *= ((out * (-1)) + 1) * out;

                // Grad for weight

                // std::cout << "For bias => " << delta << curr.m_bias << std::endl;
                for (int j = 0; j < curr.m_bias.getMetadata().shapeAt(0); j++) {
                    curr.m_gradB({j, 0}) = delta({j, 0});
                }
                curr.m_gradW = curr.m_input * delta;
                curr.m_gradW *= 0.5;

                delta *= curr.m_weight;
                delta.transpose();

                auto &deltaShape = delta.getMetadata().shape();
                for (int i = 0; i < deltaShape[0]; i++) {
                    double row = 0.f;
                    for (int j = 0; j < deltaShape[1]; j++) {
                        row += delta({i, j});
                    }
                    for (int j = 0; j < deltaShape[1]; j++) {
                        delta({i, j}) = row;
                    }
                }
            }
            for (auto &layer: m_layers) {
                layer.backward(target);
            }
        }

    private:
        std::vector<LinearNetwork> m_layers;
    };
}
