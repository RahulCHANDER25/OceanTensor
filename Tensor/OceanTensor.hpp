#pragma once
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <memory>
#include <array>

// #include "operations/mathOp.hpp"
// #include "operations/tensorOp.hpp"
 #include "MetaData.hpp"

// /!\
//
//     We want user to be able to put their shapes in the class/functions parameters
//     Check how to put operations in separate files
// /!\

// Finish transpose and think about transpose() and TransposeInPlace()

// Be able to do the do_op with respect of the strides/shapes ==> Simple math calcul.
// Class Neural Network

#include "Array.hpp"

namespace OceanTensor {
    template <typename T, int DIM>
    class myTensor {
    public:

        myTensor(std::initializer_list<int> shape): m_arr(), m_data(shape)
        {
            this->m_arr = std::make_unique<myArray<T>>(m_data.size());
        }

        myTensor(const myTensor<T, DIM> &tensor):
            m_arr(std::make_unique<myArray<T>>(*tensor.m_arr)),
            m_data(tensor.m_data)
        {
        }

        ~myTensor() = default;

        /* ---- All the functions for tensor ---- */

        T& at(int index) const
        {
            return (*this->m_arr)[index];
        }

        void dump_arr()
        {
            if (DIM != 1) {
                format_dump(0, 0);
            } else {
                std::cout << "[ ";
                for (int i = 0; i < m_arr->size(); i++) std::cout << (*m_arr)[i] << " ";
                std::cout << "]" << std::endl;
            }
        }

        void transpose(const std::initializer_list<int> &shapes = {})
        {
            m_data.transpose(shapes);
        }

        void reshape(const std::initializer_list<int> &new_shape)
        {
            m_data.reshape(new_shape);
        }

        T& operator()(std::initializer_list<int> indexes)
        {
            return (*m_arr)[m_data.toIndex({m_data.shape(), indexes})];
        }

        // Check Operator for +, -, *, ...
        myTensor operator*(myTensor<T, DIM> &ocTensor) { return this->do_op(ocTensor, std::multiplies<T>()); }
        myTensor operator+(myTensor<T, DIM> &ocTensor) { return this->do_op(ocTensor, std::plus<T>()); }
        myTensor operator-(myTensor<T, DIM> &ocTensor) { return this->do_op(ocTensor, std::minus<T>()); }
        myTensor operator/(myTensor<T, DIM> &ocTensor) { return this->do_op(ocTensor, std::divides<T>()); }

    private:
        void format_dump(int track_dim, int idx_start)
        {
            if (track_dim == (DIM - 1)) {
                for (int k = 0; k < track_dim; k++) std::cout << "\t";
                std::cout << "[ ";
                for (int k = 0; k < m_data.shapeAt(track_dim); k++) {
                    std::cout << this->at(idx_start + k * m_data.strideAt(track_dim)) << " ";
                }
                std::cout << "] " << std::endl;
                return;
            }
            for (int k = 0; k < track_dim; k++) std::cout << "\t";
            std::cout << "[" << std::endl;

            for (int i = 0; i < m_data.shapeAt(track_dim); i++) {
                format_dump(
                    track_dim + 1,
                    idx_start + i * m_data.strideAt(track_dim)
                );
            }
            for (int k = 0; k < track_dim; k++) std::cout << "\t";
            std::cout << "]" << std::endl;
        }

        template <typename F>
        myTensor do_op(myTensor<T, DIM> &ocTensor, F op)
        {
            myTensor<T, DIM> tensor(*this);

            if (!m_data.isEqual(ocTensor.m_data.shape())){
                std::cout << "[ERROR] Shape error while trying to do an operation" << std::endl;
            }
            auto it_this = tensor.m_data.begin();
            auto it_oth = ocTensor.m_data.begin();
            for (int i = 0; i < this->m_data.size(); i++) {
                int idx_this = this->m_data.toIndex(it_this);
                int idx_oth = ocTensor.m_data.toIndex(it_oth);
                (*tensor.m_arr)[idx_this] = op(tensor.at(idx_this), ocTensor.at(idx_oth));
                ++it_this;
                ++it_oth;
            }
            return tensor;
        }

        std::unique_ptr<myArray<T>> m_arr;
        Metadata m_data;
    };
}
