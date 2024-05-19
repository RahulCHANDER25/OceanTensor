#pragma once

#include "InitType.hpp"
#include "NeuralNetwork.hpp"

#include "Activation.hpp"
#include "OceanTensor.hpp"
#include <functional>

namespace Network {
    /// @brief Fully connected layer of a neural network
    class LinearNetwork: public INeuralNetwork {
    public:
        LinearNetwork(
            int in,
            int out,
            std::function<Matrix2f (const Matrix2f &, bool)> act=Act::sigmoid<2>
        ):
            INeuralNetwork(),
            m_weight{out, in},
            m_bias{out, 1},
            m_gradB{out, 1},
            m_gradW({out, in}, ZEROS),
            m_delta({out, in}, ZEROS),
            m_act(act)
        {
        }

        ~LinearNetwork() = default;

        virtual OceanTensor::myTensor<double, 2> forward(OceanTensor::myTensor<double, 2> &inputs) override
        {
            // Tracking input.
            auto Wshapes = m_weight.getMetadata().shape();
            m_input = inputs.replicate({Wshapes[0], Wshapes[1]});

            auto z = m_weight.matMul(inputs) + m_bias;
            m_output = (m_act(z, false));

            // Tracking delta activation.
            // m_delta.clear(); // This too
            // m_delta = m_act(z, true);

            return m_output;
        }

        void backward(OceanTensor::myTensor<double, 2> &) // Normalement c ok mais recheck
        {
            this->m_weight -= this->m_gradW;
            this->m_bias -= this->m_gradB;
            // std::cout << "BACKWARD" << std::endl;
            // std::cout << m_weight << std::endl;
            // std::cout << m_bias << std::endl;
            return;
        }

        OceanTensor::myTensor<double, 2> m_weight;
        OceanTensor::myTensor<double, 2> m_bias;
        OceanTensor::myTensor<double, 2> m_gradB;
        OceanTensor::myTensor<double, 2> m_gradW;
        OceanTensor::myTensor<double, 2> m_delta;

        // All of that type are temporary I have to change them !
        OceanTensor::myTensor<double, 2> m_input;
        OceanTensor::myTensor<double, 2> m_output;
        std::function<Matrix2f(const Matrix2f &, bool)> m_act;
    private:
    };
}