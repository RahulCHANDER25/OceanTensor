/*
** EPITECH PROJECT, 2024
** OceanTensor
** File description:
** trainings
*/

#pragma once

#include "Loss.hpp"
#include "Sequential.hpp"

void fit_training(
    Network::Sequential &seq,
    const std::string &saveName="",
    std::function<double(double)> func=cos,
    std::function<Matrix2f(Matrix2f &, Matrix2f &, bool)> lossFunc=Loss::squaredLoss<2>
);
