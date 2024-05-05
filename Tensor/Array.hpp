#pragma once

#include <cstddef>
#include <stdexcept>

namespace OceanTensor
{
    template <typename T>
    class myArray {
    public:
        myArray() = default;

        ~myArray()
        {
            delete[] m_data;
        }

        myArray(size_t size, bool inRange=true):
            m_data(),
            m_size(size)
        {
            m_data = new T[m_size];
            if (!m_data) {
                throw std::runtime_error("Alloc Error");
            }
            if (inRange)
                this->fillInRange();
        }


        myArray(const myArray<T> &oth):
            m_data(),
            m_size(oth.m_size)
        {
            m_data = new T[oth.m_size];
            memcpy(m_data, oth.m_data, oth.m_size * sizeof(T));
        }

        myArray(myArray<T> &&oth) noexcept :
            m_data(),
            m_size(oth.m_size)
        {
            m_data = new T[oth.m_size];
            memcpy(m_data, oth.m_data, oth.m_size);
            delete oth.m_data;
            oth.m_size = 0;
            oth.m_data = nullptr;
        }

        myArray<T> &operator=(const myArray<T> &oth)
        {
            if (oth == *this) {
                return *this;
            }
            if (m_data != nullptr)
                delete m_data;
            m_data = new T[oth.m_size];
            memcpy(m_data, oth.m_data, oth.m_size);
        }

        myArray<T> &operator=(myArray<T> &&oth)
        {
            if (oth == *this) {
                return *this;
            }
            if (m_data != nullptr)
                delete m_data;
            m_data = new T[oth.m_size];
            memcpy(m_data, oth.m_data, oth.m_size);
            delete oth.m_data;
            oth.m_size = 0;
            oth.m_data = nullptr;
        }

        T &at(size_t idx)
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

        myArray<T> operator*(T val) const
        {
            myArray<T> newArr(*this);

            arrOp(newArr, [&val] (T &curr) -> void { curr *= val; });
            return newArr;
        }
        myArray<T> operator+(T val) const
        {
            myArray<T> newArr(*this);

            arrOp(newArr, [&val] (T &curr) -> void { curr += val; });
            return newArr;
        }
        myArray<T> operator-(T val) const
        {
            myArray<T> newArr(*this);

            arrOp(newArr, [&val] (T &curr) -> void { curr -= val; });
            return newArr;
        }
        myArray<T> operator/(T val) const
        {
            myArray<T> newArr(*this);

            if (val == 0)
                throw std::runtime_error("Zero division error\n");
            arrOp(newArr, [&val] (T &curr) -> void { curr /= val; });
            return newArr;
        }

        myArray<T> &operator*=(T val)
        {
            arrOp(*this, [&val] (T &curr) -> void { curr *= val; });
            return *this;
        }
        myArray<T> &operator+=(T val)
        {
            arrOp(*this, [&val] (T &curr) -> void { curr += val; });
            return *this;
        }
        myArray<T> &operator-=(T val)
        {
            arrOp(*this, [&val] (T &curr) -> void { curr -= val; });
            return *this;
        }
        myArray<T> &operator/=(T val)
        {
            if (val == 0)
                throw std::runtime_error("Zero division error\n");
            arrOp(*this, [&val] (T &curr) -> void { curr /= val; });
            return *this;
        }

    private:
        void fillInRange()
        {
            for (size_t i = 0; i < m_size; i++) {
                m_data[i] = i;
            }
        }

        template <typename F>
        void arrOp(myArray<T> &arr, F func) const
        {
            for (size_t i = 0; i < arr.m_size; i++) {
                func(arr[i]);
            }
        }

        T *m_data;
        size_t m_size;
    };
}
