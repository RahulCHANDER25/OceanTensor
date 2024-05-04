#include <iostream>
#include "OceanTensor.hpp"
 
int main()
{
   OceanTensor::myTensor<int, 3> tensor({3, 3, 3});
   OceanTensor::myTensor<int, 3> tensor2({3, 3, 3});

   OceanTensor::myTensor<int, 3> res1 = tensor + tensor2;
   OceanTensor::myTensor<int, 3> res2 = tensor - tensor2;
   OceanTensor::myTensor<int, 3> res3 = tensor * tensor2;

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
    return 0;
}

