#include <iostream>
#include "OceanTensor.hpp"
 
int main()
{
    OceanTensor::myTensor<int, 3> t({3, 5, 2});

    // t.dump_arr();
    // t.transpose();
    // t.dump_arr();

    OceanTensor::myTensor<int, 2> toAdd({3, 2});
    OceanTensor::myTensor<int, 2> toAdd2({3, 2});

    OceanTensor::myTensor<int, 2> res1 = toAdd + toAdd2;
    OceanTensor::myTensor<int, 2> res2 = toAdd - toAdd2;
    OceanTensor::myTensor<int, 2> res3 = toAdd * toAdd2;

    std::cout << "--------------------------------" << std::endl;
    res1.dump_arr();
    std::cout << "--------------------------------" << std::endl;
    res2.dump_arr();
    std::cout << "--------------------------------" << std::endl;
    res3.dump_arr();
    std::cout << "--------------------------------" << std::endl;

    // OceanTensor<int, 1> np2({2});
    //OceanTensor<int, 2> np3({4, 1});

    //np1.dump_arr();
    //np1.transpose();
    //np1.dump_arr();
    
    //std::cout << np1({1, 1}) << std::endl;

    // OceanTensor<int, 2>* mul_test = np1 + zeros;
    return 0;
}

