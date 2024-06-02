/*
** EPITECH PROJECT, 2024
** OceanTensor
** File description:
** fit_training
*/

#include "OceanTensor.hpp"
#include "Sequential.hpp"
#include <cmath>

void fit_training(
    Network::Sequential &seq,
    const std::string &saveName,
    std::function<double(double)> func,
    std::function<Matrix2f(Matrix2f &, Matrix2f &, bool)> lossFunc
)
{
    OceanTensor::myTensor<double, 2> input({seq.in(), 1});
    OceanTensor::myTensor<double, 2> y({seq.out(), 1});
    for (int i = 0; i < input.size(); i++) {
        y[i] = cos(input[i]);
    }

    OceanTensor::myTensor<double, 2> pred;
    for (int i = 0; i < 100000; i++) {
        pred = seq.forward(input);
        if (i % 100 == 0) {
            std::cout << "----------- EPOCH " << i << "-----------" <<std::endl;
            std::cout << "Pred: " << pred[0] << " VS " << "y (cosinus): " << y[0] << std::endl;
            auto loss = lossFunc(pred, y, false);
            std::cout << "Total Loss: " << loss.sum() << std::endl << std::endl;;
            if (loss.sum() < 10e-6) {
                for (int i = 0; i < y.size(); i++)
                    std::cout << "Y = " << y[i] << "\tPRED = " << pred[i] << std::endl;
                break;
            }
        }
        seq.backward(y);
    }
    if (!saveName.empty())
        seq.save(saveName);
}
