#pragma once

#include "Linear/LinearNetwork.hpp"
#include "NeuralNetwork.hpp"

#include <initializer_list>
#include <vector>

namespace Network {
    /// @brief Fully connected layer of a neural network
    class Sequential: public INeuralNetwork {
    public:
        Sequential() = default;

        Sequential(
            std::initializer_list<LinearNetwork> linears,
            double ALPHA=0.5
        ):
            m_layers(),
            ALPHA(ALPHA)
        {
            for (auto lin: linears) {
                m_layers.push_back(std::move(lin));
            }
        }

        ~Sequential() = default;

        virtual Matrix2f forward(Matrix2f &inputs) override
        {
            auto out = inputs;

            for (auto &layer : m_layers) {
                out = layer.forward(out);
            }
            return out;
        }

        virtual void backward(Matrix2f &target)
        {
            size_t idx_last = m_layers.size() - 1;
            auto &last = m_layers[idx_last];
            auto output = last.m_act(last.m_output, false);

            auto delta = (output - target);

            auto size = m_layers.size() - 1;
            for (int i = m_layers.size() - 1; i >= 0; i--) {

                auto &curr = m_layers[i];
                auto &shape = curr.m_weight.getMetadata().shape();

                if (i == size) { // First time we need to put it in correct shape.
                    delta = delta.replicate({shape[1], shape[0]}).transposed();
                }
                auto out = curr.m_act(curr.m_output, true);
                out = out.replicate({shape[1], shape[0]}).transposed();

                delta *= out;

                for (int j = 0; j < curr.m_bias.getMetadata().shapeAt(0); j++) {
                    curr.m_gradB({j, 0}) = delta({j, 0});
                }
                curr.m_gradW = curr.m_input * delta;
                curr.m_gradW *= ALPHA;

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

        virtual void save(const std::string &filepath, std::_Ios_Openmode mode=std::ios::trunc) override;

        virtual void load(std::ifstream &ifs) override;

        int in() const { return m_layers.front().in; }
        int out() const { return m_layers.back().out; }

    private:
        std::vector<LinearNetwork> m_layers;
        const double ALPHA = 0.5;
    };
}
