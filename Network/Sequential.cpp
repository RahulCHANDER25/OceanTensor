/*
** EPITECH PROJECT, 2024
** OceanTensor
** File description:
** Sequential
*/

#include "Sequential.hpp"
#include "Linear/LinearNetwork.hpp"
#include <cstddef>
#include <fstream>
#include <stdexcept>

void Network::Sequential::save(const std::string &filepath, std::_Ios_Openmode mode)
{
    std::ofstream ofs(filepath, mode);

    ofs.write("<OceanTensor>", strlen("<OceanTensor>"));

    size_t size = m_layers.size();
    ofs.write(reinterpret_cast<char *>(&size), sizeof(size_t));
    ofs.close();

    for (auto &lin: m_layers) {
        lin.save(filepath);
    }
}

void Network::Sequential::load(std::ifstream &ifs)
{
    char magic[4096] = {0};

    ifs.read(magic, strlen("<OceanTensor>"));

    if (std::string(magic) != "<OceanTensor>")
        throw std::runtime_error("[ERROR] Wrong format of file.");
    m_layers.clear();

    size_t nb_layers = 0;

    ifs.read(reinterpret_cast<char *>(&nb_layers), sizeof(size_t));
    for (size_t i = 0; i < nb_layers; i++) {
        LinearNetwork net;
        net.load(ifs);
        m_layers.push_back(std::move(net));
    }
}

