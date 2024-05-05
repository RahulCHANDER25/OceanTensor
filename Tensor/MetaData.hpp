#pragma once

#include <initializer_list>
#include <vector>
#include <iostream>

namespace OceanTensor {
    class Metadata {
    public:
        Metadata() = default;

        Metadata(std::initializer_list<int> shape);
        Metadata(const Metadata &oth) = default;

        void transpose(std::initializer_list<int> shape={});

        void reshape(std::initializer_list<int> shapes);

        [[nodiscard]] size_t size() const { return m_size; }
        [[nodiscard]] std::vector<int> &shape() { return m_shape; }
        [[nodiscard]] std::vector<int> shape() const { return m_shape; }
        [[nodiscard]] std::vector<int> &strides() { return m_strides; }
        [[nodiscard]] std::vector<int> strides() const { return m_strides; }

        [[nodiscard]] int strideAt(size_t idx) const { return m_strides[idx]; }
        [[nodiscard]] int shapeAt(size_t idx) const { return m_shape[idx]; }

        class MetaIterator {
        public:
            MetaIterator() = default;
            MetaIterator(const MetaIterator &oth) = default;

            MetaIterator(MetaIterator &&oth) noexcept;

            explicit MetaIterator(const std::vector<int> &shape);
            MetaIterator(const std::vector<int> &shape, std::initializer_list<int> indexes);

            MetaIterator &operator++();

            std::vector<int> &getCurrShape() { return curr_shape; }

            bool end = false;
        private:
            std::vector<int> curr_shape;
            std::vector<int> max_shape;
        };

        [[nodiscard]] MetaIterator begin() const { return MetaIterator{m_shape}; }

        [[nodiscard]] size_t toIndex(MetaIterator it) const;

        bool isEqual(std::vector<int> &oth) { return std::equal(m_shape.begin(), m_shape.end(), oth.begin()); }

    private:
        void init_strides();

        static size_t getSizeByShape(std::vector<int> shape);

        std::vector<int> m_shape;
        std::vector<int> m_strides;
        size_t m_size{};
    };
}

std::ostream &operator<<(std::ostream &os, OceanTensor::Metadata::MetaIterator &it);
