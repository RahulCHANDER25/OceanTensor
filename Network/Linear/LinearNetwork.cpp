/*
** EPITECH PROJECT, 2024
** OceanTensor
** File description:
** LinearNetwork
*/

#include "LinearNetwork.hpp"
#include <fstream>

Matrix2f Network::LinearNetwork::forward(Matrix2f &inputs)
{
    auto Wshapes = m_weight.getMetadata().shape();
    m_input = inputs.replicate({Wshapes[0], Wshapes[1]});

    auto z = m_weight.matMul(inputs) + m_bias;
    m_output = (m_act(z, false));

    return m_output;
}

void Network::LinearNetwork::save(
    const std::string &filepath,
    std::_Ios_Openmode mode
)
{
    m_weight.save(filepath);
    m_bias.save(filepath);
}

void Network::LinearNetwork::load(std::ifstream &ifs)
{
    m_weight.load(ifs);
    m_bias.load(ifs);
}
