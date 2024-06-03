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

    auto z = m_weight.matMul(inputs) + m_bias;
    m_output = (m_act(z, false));

    return m_output;
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

    std::cout << "SAVE WEIGHT" << std::endl;
    m_weight.save(filepath);
    std::cout << "SAVE BIAS" << std::endl;
    m_bias.save(filepath);
}

void Network::LinearNetwork::load(std::ifstream &ifs)
{
    ifs.read(reinterpret_cast<char *>(&in), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&out), sizeof(int));

    std::cout << "IN AND OUT " << in << " " << out << std::endl;
    std::cout << "WEIGHTS" << std::endl;
    m_weight.load(ifs);
    std::cout << "File state => " << ifs.tellg() << std::endl;
    std::cout << "BIAS" << std::endl;
    m_bias.load(ifs);
    m_act = Act::sigmoid<2>;
}
