#include <iostream>
#include "OceanTensor.hpp"

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

int main()
{
    dot_test();
    matrix_test();
    tensor_test();
    return 0;
}
