#pragma once

#include "InitType.hpp"
#include "NeuralNetwork.hpp"

#include "Activation.hpp"
#include "OceanTensor.hpp"
#include <fstream>
#include <functional>

namespace Network {
    /// @brief Fully connected layer of a neural network
    class LinearNetwork: public INeuralNetwork {
    public:
        LinearNetwork() = default;

        LinearNetwork(const LinearNetwork &lin):
            INeuralNetwork(),
            m_weight(lin.m_weight),
            m_bias(lin.m_bias),
            m_gradB(lin.m_gradB),
            m_gradW(lin.m_gradW),
            m_act(lin.m_act),
            in(lin.in),
            out(lin.out)
        {}

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
            m_act(act),
            in(in),
            out(out)
        {
        }

        ~LinearNetwork() = default;

        virtual Matrix2f forward(Matrix2f &inputs) override;

        void backward(Matrix2f &) // Normalement c ok mais recheck
        {
            this->m_weight -= this->m_gradW;
            this->m_bias -= this->m_gradB;
            return;
        }

        virtual void save(
            const std::string &filepath,
            std::_Ios_Openmode mode=std::ios::app
        ) override;

        virtual void load(std::ifstream &) override;

        Matrix2f m_weight;
        Matrix2f m_bias;
        Matrix2f m_gradB;
        Matrix2f m_gradW;

        // All of that type are temporary I have to change them !
        Matrix2f m_input;
        Matrix2f m_output;
        std::function<Matrix2f(const Matrix2f &, bool)> m_act;
        int in;
        int out;
    private:
    };
}