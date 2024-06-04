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
    const size_t EPOCH,
    const std::string &saveName="",
    std::function<double(double)> func=cos,
    std::function<Matrix2f(Matrix2f &, Matrix2f &, bool)> lossFunc=Loss::squaredLoss<2>
);

void func_predict_training(
    Network::Sequential &seq,
    const size_t EPOCH,
    const std::string &saveName="",
    std::function<double(double)> func=cos,
    std::function<Matrix2f(Matrix2f &, Matrix2f &, bool)> lossFunc=Loss::meanSquaredLoss<2>
);
