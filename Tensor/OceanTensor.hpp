#pragma once
#include <initializer_list>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#include "MetaData.hpp"

#include "InitType.hpp"

#include "Array.hpp"

// Add N Row with argument copy line=-1
// Add N Col with argument copy line=-1

namespace OceanTensor {
    template <typename T, int DIM>
    class myTensor {
    public:

        myTensor() = default;

        myTensor(std::initializer_list<int> shape, enum InitType type=RANDOM): m_arr(), m_data(shape)
        {
            this->m_arr = myArray<T>(m_data.size(), type);
        }

        myTensor(const myTensor<T, DIM> &tensor):
            m_arr(myArray<T>(tensor.m_arr)),
            m_data(tensor.m_data)
        {
        }

        myTensor(const myArray<T> &array, const Metadata &data):
            m_arr(std::move(array)),
            m_data(data)
        {
        }

        ~myTensor() = default;

        /* ---- All the functions for tensor ---- */

        T& at(int index) const
        {
            return this->m_arr.at(index);
        }

        size_t size() const { return m_data.size(); }
        Metadata &getMetadata() { return m_data; }

        void dump_arr() const
        {
            if (DIM != 1) {
                format_dump(0, 0);
            } else {
                std::cout << "[ ";
                for (int i = 0; i < m_arr.size(); i++) std::cout << m_arr[i] << " ";
                std::cout << "]" << std::endl;
            }
        }

        /// @brief Modification of the shape and the stride of the tensor
        void transpose(const std::initializer_list<int> &shapes = {})
        {
            m_data.transpose(shapes);
        }

        /// @brief Modification of the shape and the stride of the tensor into a new one 
        myTensor<T, DIM> transposed(const std::initializer_list<int> &shapes = {})
        {
            myTensor<T, DIM> newTensor(*this);
            newTensor.m_data.transpose(shapes);
            return newTensor;
        }

        void reshape(const std::initializer_list<int> &new_shape)
        {
            m_data.reshape(new_shape);
        }

        void normalize()
        {
            T s = sum();

            if (s == 0)
                return;
            for (int i = 0; i < size(); i++) {
                this->m_arr[i] /= s;
            }
        }

        T dot(const myTensor<T, DIM> &ocTensor) requires (DIM == 1)
        {
            if (m_data.size() != ocTensor.m_data.size())
                throw std::runtime_error("Tensor needs to be of same size");
            T res = 0;
            for (size_t i = 0; i < m_data.size(); i++) {
                res += m_arr[i] * ocTensor.m_arr[i];
            }
            return res;
        }

        myTensor<T, DIM> matMul(const myTensor<T, DIM> &oth) requires (DIM == 2)
        {
            myTensor<T, DIM> newTensor({m_data.shapeAt(0), oth.m_data.shapeAt(1)}, ZEROS);

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

        T sum() const
        {
            T res = 0;
            for (int i = 0; i < m_data.size(); i++) {
                res += m_arr[i];
            }
            return res;
        }

        myTensor<T, DIM> sqrt()
        {
            myTensor<T, DIM> newTensor(*this);

            for (int i = 0; i < m_data.size(); i++) {
                newTensor[i] = std::sqrt(newTensor[i]);
            }
            return newTensor;
        }

        void clear()
        {
            for (size_t i = 0; i < m_data.size(); i++) {
                this->m_arr[i] = 0;
            }
        }

        /// @brief Replicate N (size of the new tensor) times all the numbers of the tensors in a new shape.
        ///        Careful if N is not equal to a pow of the size of the current tensor then copy will
        ///        truncated.
        ///        WARNING: This function replicate in accordance of raw buffer not shapes.
        myTensor<T, DIM> replicate(std::initializer_list<int> shape) const
        {
            int sizeThis = size();
            myTensor<T, DIM> ten{shape};

            for (int i = 0; i < ten.size(); i++) {
                ten[i] = this->m_arr[i % sizeThis];
            }
            return ten;
        }

        T& operator()(std::initializer_list<int> indexes)
        {
            return m_arr[m_data.toIndex({m_data.shape(), indexes})];
        }

        T operator()(std::initializer_list<int> indexes) const
        {
            return m_arr[m_data.toIndex({m_data.shape(), indexes})];
        }

        T &operator[](size_t idx)
        {
            return m_arr[idx];
        }

        T operator[](size_t idx) const
        {
            return m_arr[idx];
        }

        myTensor<T, DIM> operator=(const myTensor<T, DIM> &ocTensor) 
        {
            this->m_arr = ocTensor.m_arr;
            this->m_data = ocTensor.m_data;
            return *this;
        }

        myTensor<T, DIM> operator=(myTensor<T, DIM> &&ocTensor)
        {
            this->m_arr = std::move(ocTensor.m_arr);
            this->m_data = std::move(ocTensor.m_data);
            return *this;
        }

        myTensor operator*(myTensor<T, DIM> &ocTensor) { return this->do_opNew(ocTensor, std::multiplies<T>()); }
        myTensor operator+(myTensor<T, DIM> &ocTensor) { return this->do_opNew(ocTensor, std::plus<T>()); }
        myTensor operator-(myTensor<T, DIM> &ocTensor) { return this->do_opNew(ocTensor, std::minus<T>()); }
        myTensor operator/(myTensor<T, DIM> &ocTensor) { return this->do_opNew(ocTensor, std::divides<T>()); }

        myTensor operator*(T val) { return myTensor<T, DIM>(this->m_arr * val, m_data); }
        myTensor operator+(T val) { return myTensor<T, DIM>(this->m_arr + val, m_data); }
        myTensor operator-(T val) { return myTensor<T, DIM>(this->m_arr - val, m_data); }
        myTensor operator/(T val) { return myTensor<T, DIM>(this->m_arr / val, m_data); }

        myTensor &operator*=(const myTensor<T, DIM> &ocTensor) { return do_op(*this, ocTensor, std::multiplies<T>()); }
        myTensor &operator+=(const myTensor<T, DIM> &ocTensor) { return do_op(*this, ocTensor, std::plus<T>()); }
        myTensor &operator-=(const myTensor<T, DIM> &ocTensor) { return do_op(*this, ocTensor, std::minus<T>()); }
        myTensor &operator/=(const myTensor<T, DIM> &ocTensor) { return do_op(*this, ocTensor, std::divides<T>()); }

        myTensor &operator*=(T val)
        {
            this->m_arr *= val;
            return *this;
        }
        myTensor &operator+=(T val)
        {
            this->m_arr += val;
            return *this;
        }
        myTensor &operator-=(T val)
        {
            this->m_arr += val;
            return *this;
        }
        myTensor &operator/=(T val)
        {
            this->m_arr /= val;
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
        myTensor<T, DIM> &do_op(myTensor<T, DIM> &tensor, const myTensor<T, DIM> &ocTensor, F op)
        {
            if (!tensor.m_data.isEqual(ocTensor.m_data.shape())){
                throw std::runtime_error("[ERROR] Shape error while trying to do an operation");
            }
            auto it_this = tensor.m_data.begin();
            auto it_oth = ocTensor.m_data.begin();
            for (int i = 0; i < tensor.m_data.size(); i++) {
                int idx_this = tensor.m_data.toIndex(it_this);
                int idx_oth = ocTensor.m_data.toIndex(it_oth);
                tensor.m_arr[idx_this] = op(tensor.at(idx_this), ocTensor.at(idx_oth));
                ++it_this;
                ++it_oth;
            }
            return tensor;
        }

        myArray<T> m_arr;
        Metadata m_data;
    };
}

typedef OceanTensor::myTensor<double, 2> Matrix2f;

template <typename T, int DIM>
std::ostream &operator<<(std::ostream &os, const OceanTensor::myTensor<T, DIM> &t)
{
    t.dump_arr();
    return os;
}
