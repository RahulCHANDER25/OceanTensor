#pragma once
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <memory>
#include <array>

#include "operations/mathOp.hpp"
#include "operations/tensorOp.hpp"
#include "Metadata.hpp"

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

        myTensor(const std::initializer_list<int> &shape): m_arr(), m_data(shape)
        {
            this->m_arr = std::make_unique<myArray<T>>(m_data.getSize());
        }

        myTensor(const myTensor<T, DIM> &tensor):
            m_arr(std::make_unique<myArray<T>>(*tensor.m_arr, tensor.m_data.getSize())),
            m_data(tensor.m_data)
        {
        }

        ~myTensor()
        {
        }

        /* ---- All the functions for tensor ---- */

        T* get_raw_data() const
        {
            return this->m_arr->get_data();
        }

        T& at(int index) const {
            return this->m_arr->value_at(index);
        }

        void dump_arr()
        {
            if (DIM != 1) {
                format_dump(0, 0);
            } else {
                T *data = this->get_raw_data();
                std::cout << "[ ";
                for (int i = 0; i < m_data.getSize(); i++) std::cout << data[i] << " ";
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

        T& operator()(const std::initializer_list<int> &indexes)
        {
            return m_data.indexesToIndex<DIM>(indexes);
        }

        myTensor operator*(myTensor<T, DIM> &npCat) {
            return this->do_op(npCat, &op::mul);
        }

        myTensor operator+(myTensor<T, DIM> &npCat) {
            return this->do_op(npCat, &op::add);
        }

        myTensor operator-(myTensor<T, DIM> &npCat) {
            return this->do_op(npCat, &op::sub);
        }

        myTensor operator/(myTensor<T, DIM> &npCat) {
            return this->do_op(npCat, &op::div);
        }

        myTensor operator%(myTensor<T, DIM> &npCat) {
            return this->do_op(npCat, &op::mod);
        }

    private:
        void format_dump(int track_dim, int idx_start)
        {
            if (track_dim == (DIM - 1)) {
                for (int k = 0; k < track_dim; k++) std::cout << "\t";
                std::cout << "[ ";
                for (int k = 0; k < m_data.getShapeValueAt(track_dim); k++) {
                    std::cout << this->at(idx_start + k * m_data.getStridesValueAt(track_dim)) << " ";
                }
                std::cout << "] " << std::endl;
                return;
            }
            for (int k = 0; k < track_dim; k++) std::cout << "\t";
            std::cout << "[" << std::endl;

            for (int i = 0; i < m_data.getShapeValueAt(track_dim); i++) {
                format_dump(
                    track_dim + 1,
                    idx_start + i * m_data.getStridesValueAt(track_dim)
                );
            }
            for (int k = 0; k < track_dim; k++) std::cout << "\t";
            std::cout << "]" << std::endl;
        }

        myTensor do_op(myTensor<T, DIM> &npCat, T (*op)(T, T))
        {
            myTensor<T, DIM> tensor(*this);
            auto &shape = m_data.getShape();

            if (!metadataOp::_isShapeEqual(shape, npCat.m_data.getShape())){
                std::cout << "[ERROR] Shape error while trying to do an operation" << std::endl;
            }
            auto it_this = tensor.m_data.begin<DIM>();
            auto it_oth = npCat.m_data.begin<DIM>();
            for (int i = 0; i < this->m_data.getSize(); i++) {
                int idx_this = this->m_data.indexesToIndex(it_this);
                int idx_oth = npCat.m_data.indexesToIndex(it_oth);
                tensor.m_arr->value_at(idx_this) = op(tensor.at(idx_this), npCat.at(idx_oth));
                ++it_this;
                ++it_oth;
            }
            return tensor;
        }

        std::unique_ptr<myArray<T>> m_arr;
        Metadata m_data;
    };
}
