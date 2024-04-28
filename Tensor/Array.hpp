#pragma once

namespace OceanTensor
{
    template <typename T>
    class myArray {
    public:
        myArray();
        myArray(const myArray<T> &oth);

        myArray(myArray<T> &&oth);

        myArray<T> &operator=(const myArray<T> &other);
        myArray<T> &operator=(myArray<T> &&other);

        T &at(size_t idx);

    private:
        void fillInRange(void);

        T *m_data;
    };
}
