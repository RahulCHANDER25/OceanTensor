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
            m_weight({out, in}),
            m_bias({out, 1}),
            m_grad({out, in}, ZEROS),
            m_delta({out, 1}, ZEROS),
            m_act(act)
        {
        }

        ~LinearNetwork() = default;

        virtual OceanTensor::myTensor<double, 2> forward(OceanTensor::myTensor<double, 2> &inputs) override
        {
            std::cout << "Weight: " << m_weight << std::endl;
            std::cout << "Bias: " << m_bias << std::endl;
            m_input = inputs;
            auto outNet(m_act(m_weight.matMul(inputs) + m_bias, false));
            m_grad.clear();
            std::cout << outNet.transposed() << std::endl;
            // m_grad += outNet.transposed();
            // std::cout << m_grad << std::endl;
            return outNet;
        }

        void backward(OceanTensor::myTensor<double, 2> &error)
        {
            // double loss = error.sum();
            auto derivLoss = (error * 2).sqrt();

            std::cout << "There is derivLoss, Net over Activation and Weight over Net" << std::endl;
            std::cout << derivLoss << std::endl;
            std::cout << "" << std::endl;
            std::cout << m_input << std::endl;
        }

        OceanTensor::myTensor<double, 2> m_weight;
        OceanTensor::myTensor<double, 2> m_bias;
        OceanTensor::myTensor<double, 2> m_grad;
        OceanTensor::myTensor<double, 2> m_delta;

        // Maybe have a reference to input ? // Check for optimization how to do it.
        OceanTensor::myTensor<double, 2> m_input;
        std::function<Matrix2f(const Matrix2f &, bool)> m_act;
    private:
    };
}