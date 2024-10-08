#pragma once

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <random>
#include "InitType.hpp"

namespace OceanTensor
{
    template <typename T>
    class Array {
    public:
        Array(): m_data(nullptr), m_size(0) {}

        ~Array()
        {
            if (m_data != nullptr)
                delete[] m_data;
        }

        Array(size_t size, T *data):
            m_data(data),
            m_size(size)
        {
        }

        Array(size_t size, InitType type):
            m_data(nullptr),
            m_size(size)
        {
            m_data = new T[m_size];
            if (!m_data) {
                throw std::runtime_error("Alloc Error");
            }
            switch (type) {
                case ZEROS:
                    this->fillZeros();
                    break;
                case IN_RANGE:
                    this->fillInRange();
                    break;
                case RANDOM:
                    this->fillRandom();
                    break;
            }
        }


        Array(const Array<T> &oth):
            m_data(nullptr),
            m_size(oth.m_size)
        {

            m_data = new T[oth.m_size];
            memcpy(m_data, oth.m_data, oth.m_size * sizeof(T));
        }

        Array(Array<T> &&oth) noexcept :
            m_data(nullptr),
            m_size(oth.m_size)
        {

            m_data = new T[oth.m_size];
            memcpy(m_data, oth.m_data, oth.m_size * sizeof(T));
            delete[] oth.m_data;
            oth.m_size = 0;
            oth.m_data = nullptr;
        }

        Array<T> &operator=(const Array<T> &oth)
        {
            if (m_data != nullptr)
                delete[] m_data;
            m_data = new T[oth.m_size];
            m_size = oth.size();
            memcpy(m_data, oth.m_data, oth.m_size * sizeof(T));
            return *this;
        }

        Array<T> &operator=(Array<T> &&oth)
        {
            if (m_data != nullptr)
                delete[] m_data;
            m_size = oth.size();
            m_data = new T[oth.m_size];
            memcpy(m_data, oth.m_data, oth.m_size * sizeof(T));
            delete[] oth.m_data;
            oth.m_size = 0;
            oth.m_data = nullptr;
            return  *this;
        }

        T &at(size_t idx) const
        {
            if (idx >= m_size)
                throw std::runtime_error("Out of bounds error");
            return m_data[idx];
        }

        [[nodiscard]] size_t size() const { return m_size; }

        T &operator[](size_t idx) { return at(idx); }
        T operator[](size_t idx) const
        {
            if (idx >= m_size)
                throw std::runtime_error("Out of bounds error");
            return m_data[idx];
        }

        Array<T> operator*(T val) const
        {
            Array<T> newArr(*this);

            arrOp(newArr, [&val] (T &curr) -> void { curr *= val; });
            return newArr;
        }
        Array<T> operator+(T val) const
        {
            Array<T> newArr(*this);

            arrOp(newArr, [&val] (T &curr) -> void { curr += val; });
            return newArr;
        }
        Array<T> operator-(T val) const
        {
            Array<T> newArr(*this);

            arrOp(newArr, [&val] (T &curr) -> void { curr -= val; });
            return newArr;
        }
        Array<T> operator/(T val) const
        {
            Array<T> newArr(*this);

            if (val == 0)
                throw std::runtime_error("Zero division error\n");
            arrOp(newArr, [&val] (T &curr) -> void { curr /= val; });
            return newArr;
        }

        Array<T> &operator*=(T val)
        {
            arrOp(*this, [&val] (T &curr) -> void { curr *= val; });
            return *this;
        }
        Array<T> &operator+=(T val)
        {
            arrOp(*this, [&val] (T &curr) -> void { curr += val; });
            return *this;
        }
        Array<T> &operator-=(T val)
        {
            arrOp(*this, [&val] (T &curr) -> void { curr -= val; });
            return *this;
        }
        Array<T> &operator/=(T val)
        {
            if (val == 0)
                throw std::runtime_error("Zero division error\n");
            arrOp(*this, [&val] (T &curr) -> void { curr /= val; });
            return *this;
        }

        T *getRawData() { return m_data; }

        void setRawData(T *data)
        {

            m_data = data;
        }

    private:
        void fillInRange() {
            for (size_t i = 0; i < m_size; i++) {
                m_data[i] = i;
            }
        }

        void fillZeros() {
            for (size_t i = 0; i < m_size; i++) {
                m_data[i] = 0;
            }
        }

        void fillRandom()
        {
            std::random_device dev;
            std::mt19937 gen(dev());
            std::uniform_real_distribution<double> uniform(-1, 1);

            for (size_t i = 0; i < m_size; i++) {
                m_data[i] = uniform(gen);
            }
        }

        template <typename F>
        void arrOp(Array<T> &arr, F func) const
        {
            for (size_t i = 0; i < arr.m_size; i++) {
                func(arr[i]);
            }
        }

        T *m_data;
        size_t m_size;
    };
}
