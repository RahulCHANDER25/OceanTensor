/*
** EPITECH PROJECT, 2024
** OceanTensor
** File description:
** tests
*/

#include "Loss.hpp"
#include "OceanTensor.hpp"
#include "Linear/LinearNetwork.hpp"
#include "Sequential.hpp"

/// @brief Example from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
void formatNets(Network::LinearNetwork &net, Network::LinearNetwork &net2)
{
    // First Layer weight values
    net.m_weight({0, 0}) = 0.15;
    net.m_weight({0, 1}) = 0.20;
    net.m_weight({1, 0}) = 0.25;
    net.m_weight({1, 1}) = 0.30;

    // First Layer biases values
    net.m_bias[0] = 0.35;
    net.m_bias[1] = 0.35;

    // Second Layer weight values
    net2.m_weight({0, 0}) = 0.40;
    net2.m_weight({0, 1}) = 0.45;
    net2.m_weight({1, 0}) = 0.50;
    net2.m_weight({1, 1}) = 0.55;

    // Second Layer biases values
    net2.m_bias[0] = 0.60;
    net2.m_bias[1] = 0.60;
}

void test_Sequential()
{
    Network::LinearNetwork net(2, 2);
    Network::LinearNetwork netOut(2, 2);

    formatNets(net, netOut);

    Network::Sequential seq{
        std::move(net),
        std::move(netOut)
    };
    OceanTensor::myTensor<double, 2> input({2, 1}, IN_RANGE);
    OceanTensor::myTensor<double, 2> pred({10, 1}, IN_RANGE);
    OceanTensor::myTensor<double, 2> y({2, 1});

    // Input values
    input[0] = 0.05;
    input[1] = 0.10;

    // Prediction
    y[0] = 0.01;
    y[1] = 0.99;

    std::cout << "Prediction: " << seq.forward(input) << std::endl;
    seq.backward(y);
}


void dot_test()
{
    OceanTensor::myTensor<int, 1> vect1({3});
    OceanTensor::myTensor<int, 1> vect2({3});

    auto res = vect1.dot(vect2);

    std::cout << "-- LEFT --" << std::endl;
    std::cout << vect1 << std::endl;

    std::cout << "-- RIGHT --" << std::endl;
    vect2.dump_arr();

    std::cout << "-- Dot --" << std::endl;
    std::cout << res << std::endl;
}

void matrix_test()
{
    OceanTensor::myTensor<int, 2> mat1({3, 3});
    OceanTensor::myTensor<int, 2> mat2({3, 3});

    auto mat3 = mat1.matMul(mat2);

    std::cout << "-- LEFT --" << std::endl;
    mat1.dump_arr();

    std::cout << "-- RIGHT --" << std::endl;
    mat2.dump_arr();

    std::cout << "-- MATMUL --" << std::endl;
    mat3.dump_arr();
}

void tensor_test()
{
    OceanTensor::myTensor<int, 3> tensor({3, 3, 3});
    OceanTensor::myTensor<int, 3> tensor2({3, 3, 3});

    OceanTensor::myTensor<int, 2> t({3, 3});

    OceanTensor::myTensor<int, 3> res1 = tensor + 2;
    OceanTensor::myTensor<int, 3> res2 = tensor - 4;
    OceanTensor::myTensor<int, 3> res3 = tensor * 32;

    std::cout << "-- LEFT --" << std::endl;
    tensor.dump_arr();
    std::cout << "-- RIGHT --" << std::endl;
    tensor2.dump_arr();

    std::cout << "---- ADD ----" << std::endl;
    res1.dump_arr();
    std::cout << "---- MINUS ----" << std::endl;
    res2.dump_arr();
    std::cout << "---- MULTIPLY ----" << std::endl;
    res3.dump_arr();
    std::cout << "--------------------------------" << std::endl;
}

/// @brief Example from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
void linear_test()
{
    Network::LinearNetwork net(2, 2);
    Network::LinearNetwork netOut(2, 2);
    OceanTensor::myTensor<double, 2> input({2, 1});
    OceanTensor::myTensor<double, 2> y({2, 1});

    /* -------------------- INIT OF NETWORK -------------------- */

    // Prediction
    y[0] = 0.01;
    y[1] = 0.99;

    // Input values
    input[0] = 0.05;
    input[1] = 0.10;

    // First Layer weight values
    net.m_weight({0, 0}) = 0.15;
    net.m_weight({0, 1}) = 0.20;
    net.m_weight({1, 0}) = 0.25;
    net.m_weight({1, 1}) = 0.30;

    // First Layer biases values
    net.m_bias[0] = 0.35;
    net.m_bias[1] = 0.35;

    // Second Layer weight values
    netOut.m_weight({0, 0}) = 0.40;
    netOut.m_weight({0, 1}) = 0.45;
    netOut.m_weight({1, 0}) = 0.50;
    netOut.m_weight({1, 1}) = 0.55;

    // Second Layer biases values
    netOut.m_bias[0] = 0.60;
    netOut.m_bias[1] = 0.60;

    /* -------------------- COMPUTE FORWARD -------------------- */

    // Input and desired output
    std::cout << "Input: " << input << std::endl;
    std::cout << "Y: " << y << std::endl;

    // Going through the first net
    auto net1Forward = net.forward(input);
    std::cout << "Net output + Activation: \n" << net1Forward << std::endl;

    // Going through the second net
    auto pred = netOut.forward(net1Forward);
    std::cout << "Net output + Activation: \n" << pred << std::endl;

    // Compute the loss net
    auto loss = Loss::squaredLoss(pred, y, false);
    std::cout << "Loss: " << loss << std::endl;;
    std::cout << "Total Loss: " << loss.sum() << std::endl << std::endl;;

    /* -------------------- COMPUTE BACKWARD -------------------- */

    netOut.backward(loss);
    // if first layer ==> Apply simple backpropagation
    // else apply with one layer before
}

void copy_constructor_test()
{
    OceanTensor::myTensor<double, 2> input({2, 1});
    OceanTensor::myTensor<double, 2> cpy;

    cpy = input;
    std::cout << input << std::endl;
}

void test_save()
{
    // Network::LinearNetwork net(10, 10);
    // Network::LinearNetwork test;

    // std::ifstream ifs("test.txt");

    // net.m_weight.transpose();
    // std::cout << net.m_weight << std::endl;
    // ifs.close();
    // net.save("test.txt");
    // ifs.open("test.txt");
    // test.load(ifs);
    // std::cout << test.m_weight << std::endl;
}
