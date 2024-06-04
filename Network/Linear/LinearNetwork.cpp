/*
** EPITECH PROJECT, 2024
** OceanTensor
** File description:
** LinearNetwork
*/

#include "LinearNetwork.hpp"
#include "Activation.hpp"
#include <fstream>

Matrix2f Network::LinearNetwork::forward(Matrix2f &inputs)
{
    auto Wshapes = m_weight.getMetadata().shape();
    m_input = inputs.replicate({Wshapes[0], Wshapes[1]});

    m_output = m_weight.matMul(inputs) + m_bias;
    auto a = (m_act(m_output, false));

    return a;
}

void Network::LinearNetwork::save(
    const std::string &filepath,
    std::_Ios_Openmode mode
)
{
    std::ofstream ofs(filepath, mode);

    ofs.write(reinterpret_cast<char *>(&in), sizeof(int));
    ofs.write(reinterpret_cast<char *>(&out), sizeof(int));
    ofs.close();

    m_weight.save(filepath);
    m_bias.save(filepath);
}

void Network::LinearNetwork::load(std::ifstream &ifs)
{
    ifs.read(reinterpret_cast<char *>(&in), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&out), sizeof(int));

    m_weight.load(ifs);
    m_bias.load(ifs);
    m_act = Act::sigmoid<2>;
}
