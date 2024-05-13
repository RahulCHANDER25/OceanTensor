#pragma once

#include "InitType.hpp"
#include "NeuralNetwork.hpp"

#include "Activation.hpp"
#include "OceanTensor.hpp"
#include <chrono>
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
            m_weight({out, in}),
            m_bias({out, 1}),
            m_gradW({out, in}, ZEROS),
            m_delta({out, 1}, ZEROS),
            m_act(act)
        {
        }

        ~LinearNetwork() = default;

        virtual OceanTensor::myTensor<double, 2> forward(OceanTensor::myTensor<double, 2> &inputs) override
        {
            /*
                std::cout << "Weight: " << m_weight << std::endl;
                std::cout << "Bias: " << m_bias << std::endl;

                auto Wshapes = m_weight.getMetadata().shape();
                m_input = inputs.transposed().replicate({Wshapes[0], Wshapes[1]}); // Important !
            */

            auto outNet(m_act(m_weight.matMul(inputs) + m_bias, false));

            /*
                m_delta.clear(); // This too
                m_delta = ((outNet * (-1)) + 1) * outNet;
            */

            return outNet;
        }

        void backward(OceanTensor::myTensor<double, 2> &error) // Normalement c ok mais recheck
        {
            auto Wshapes = m_weight.getMetadata().shape();
            auto derivLoss = (error * 2).sqrt();

            // std::cout << "There is derivLoss, Net over Activation and Weight over Net" << std::endl;
            // std::cout << m_delta << std::endl;

            m_delta *= derivLoss;

            auto gradDelta  = m_delta.replicate({Wshapes[0], Wshapes[1]});
            gradDelta.transpose();

            // std::cout << m_input << gradDelta << std::endl;

            m_gradW = (m_input * gradDelta) * 0.5;

            // std::cout << m_gradW << std::endl;
            // std::cout << m_weight << std::endl;
            // std::cout << m_weight - m_gradW << std::endl;
            return;
        }

        OceanTensor::myTensor<double, 2> m_weight;
        OceanTensor::myTensor<double, 2> m_bias;
        OceanTensor::myTensor<double, 2> m_gradW;
        OceanTensor::myTensor<double, 2> m_delta;

        // Maybe have a reference to input ? // Check for optimization how to do it.
        OceanTensor::myTensor<double, 2> m_input;
        std::function<Matrix2f(const Matrix2f &, bool)> m_act;
    private:
    };
}