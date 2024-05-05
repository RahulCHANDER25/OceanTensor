#pragma once
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <memory>
#include <array>
#include <numeric>

#include <type_traits>

#include "MetaData.hpp"

// /!\
//
//     We want user to be able to put their shapes in the class/functions parameters
//     Check how to put operations in separate files
// /!\

// Finish transpose and think about transpose() and TransposeInPlace()

// Class Neural Network

#include "Array.hpp"

namespace OceanTensor {
    template <typename T, int DIM>
    class myTensor {
    public:

        myTensor(std::initializer_list<int> shape, bool inRange=true): m_arr(), m_data(shape)
        {
            this->m_arr = std::make_unique<myArray<T>>(m_data.size(), inRange);
        }

        myTensor(const myTensor<T, DIM> &tensor):
            m_arr(std::make_unique<myArray<T>>(*tensor.m_arr)),
            m_data(tensor.m_data)
        {
        }

        myTensor(const myArray<T> &array, const Metadata &data):
            m_arr(std::make_unique<myArray<T>>(std::move(array))),
            m_data(data)
        {
        }

        ~myTensor() = default;

        /* ---- All the functions for tensor ---- */

        T& at(int index) const
        {
            return (*this->m_arr)[index];
        }

        void dump_arr() const
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

        T dot(const myTensor<T, DIM> &ocTensor) requires (DIM == 1)
        {
            if (m_data.size() != ocTensor.m_data.size())
                throw std::runtime_error("Tensor needs to be of same size");
            T res = 0;
            for (size_t i = 0; i < m_data.size(); i++) {
                res += (*m_arr)[i] * (*ocTensor.m_arr)[i];
            }
            return res;
        }

        myTensor<T, DIM> matMul(const myTensor<T, DIM> &oth) requires (DIM == 2)
        {
            myTensor<T, DIM> newTensor({m_data.shapeAt(0), oth.m_data.shapeAt(1)}, false);

            if (m_data.shapeAt(1) != oth.m_data.shapeAt(0)) {
                throw std::runtime_error("Incorrect dimension for matrix.");
            }
            for (int i = 0; i < m_data.shapeAt(0); i++) {
                for (int j = 0; j < m_data.shapeAt(1); j++) {
                    for (int k = 0; k < oth.m_data.shapeAt(1); k++) {
                        newTensor({i, k}) +=
                            this->operator()({i, j}) * oth.operator()({j, k});
                    }
                }
            }
            return newTensor;
        }

        T& operator()(std::initializer_list<int> indexes)
        {
            return (*m_arr)[m_data.toIndex({m_data.shape(), indexes})];
        }

        T operator()(std::initializer_list<int> indexes) const
        {
            return (*m_arr)[m_data.toIndex({m_data.shape(), indexes})];
        }

        myTensor operator*(myTensor<T, DIM> &ocTensor) { return this->do_opNew(ocTensor, std::multiplies<T>()); }
        myTensor operator+(myTensor<T, DIM> &ocTensor) { return this->do_opNew(ocTensor, std::plus<T>()); }
        myTensor operator-(myTensor<T, DIM> &ocTensor) { return this->do_opNew(ocTensor, std::minus<T>()); }
        myTensor operator/(myTensor<T, DIM> &ocTensor) { return this->do_opNew(ocTensor, std::divides<T>()); }

        myTensor operator*(T val) { return myTensor<T, DIM>((*this->m_arr) * val, m_data); }
        myTensor operator+(T val) { return myTensor<T, DIM>((*this->m_arr) + val, m_data); }
        myTensor operator-(T val) { return myTensor<T, DIM>((*this->m_arr) - val, m_data); }
        myTensor operator/(T val) { return myTensor<T, DIM>((*this->m_arr) / val, m_data); }

        myTensor &operator*=(myTensor<T, DIM> &ocTensor) { return do_op(*this, ocTensor, std::multiplies<T>()); }
        myTensor &operator+=(myTensor<T, DIM> &ocTensor) { return do_op(*this, ocTensor, std::plus<T>()); }
        myTensor &operator-=(myTensor<T, DIM> &ocTensor) { return do_op(*this, ocTensor, std::minus<T>()); }
        myTensor &operator/=(myTensor<T, DIM> &ocTensor) { return do_op(*this, ocTensor, std::divides<T>()); }

        myTensor &operator*=(T val) {
            (*this->m_arr) * val;
            return *this;
        }
        myTensor &operator+=(T val)
        {
            (*this->m_arr) + val;
            return *this;
        }
        myTensor &operator-=(T val)
        {
            (*this->m_arr) - val;
            return *this;
        }
        myTensor &operator/=(T val)
        {
            (*this->m_arr) / val;
            return *this;
        }

    private:
        // This function is used for formatting dump of tensors. (Used only by dump_arr).
        void format_dump(int track_dim, int idx_start) const
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
        myTensor<T, DIM> do_opNew(myTensor<T, DIM> &ocTensor, F op)
        {
            myTensor<T, DIM> tensor(*this);

            do_op(tensor, ocTensor, op);
            return tensor;
        }

        template <typename F>
        myTensor<T, DIM> &do_op(myTensor<T, DIM> &tensor, myTensor<T, DIM> &ocTensor, F op)
        {
            if (!tensor.m_data.isEqual(ocTensor.m_data.shape())){
                std::cout << "[ERROR] Shape error while trying to do an operation" << std::endl;
            }
            auto it_this = tensor.m_data.begin();
            auto it_oth = ocTensor.m_data.begin();
            for (int i = 0; i < tensor.m_data.size(); i++) {
                int idx_this = tensor.m_data.toIndex(it_this);
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

template <typename T, int DIM>
std::ostream &operator<<(std::ostream &os, const OceanTensor::myTensor<T, DIM> &t)
{
    t.dump_arr();
    return os;
}
