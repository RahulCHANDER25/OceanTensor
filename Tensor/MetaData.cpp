#include "MetaData.hpp"

#include <numeric>
#include <initializer_list>
#include <iostream>
#include <algorithm>

OceanTensor::Metadata::Metadata(std::initializer_list<int> shape):
    m_shape(shape),
    m_strides(),
    m_size()
{
    m_size = getSizeByShape(m_shape);
    init_strides();
}

void OceanTensor::Metadata::init_strides()
{
    int curr_stride = 1;

    m_strides.push_back(curr_stride);
    for (std::size_t i = 0; i < m_shape.size(); i++) {
        if (i == 0)
            continue;
        curr_stride *= m_shape[m_shape.size() - i];
        m_strides.push_back(curr_stride);
    }
    std::reverse(m_strides.begin(), m_strides.end());
}

void OceanTensor::Metadata::transpose(std::initializer_list<int> shape)
{
    if (shape.size() == 0) {
        std::reverse(m_shape.begin(), m_shape.end());
        std::reverse(m_strides.begin(), m_strides.end());
    }
}

void OceanTensor::Metadata::reshape(std::initializer_list<int> shape)
{
    size_t size = getSizeByShape(shape);

    if (size != m_size) {
        throw std::runtime_error("Cannot reshape to this new shape");
    }
    m_shape = shape;
}

size_t OceanTensor::Metadata::getSizeByShape(std::vector<int> shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1, [](int acc, int val) {
        return acc * ((val == 0) ? 1 : val);
    });
}

size_t OceanTensor::Metadata::toIndex(OceanTensor::Metadata::MetaIterator it) const
{
    auto &curr_shape = it.getCurrShape();
    size_t idx = 0;

    auto end_shape = curr_shape.rbegin();
    auto end_strides = m_strides.rbegin();
    while (end_shape != curr_shape.rend() && end_strides != m_strides.rend()) {
        idx += (*end_shape) * (*end_strides);
        end_shape++;
        end_strides++;
    }
    return idx;
}

/*  META ITERATOR CLASS */

OceanTensor::Metadata::MetaIterator::MetaIterator(
    const std::vector<int> &shape
):
    curr_shape(),
    max_shape(shape)
{
    for (size_t i = 0; i < max_shape.size(); i++) {
        curr_shape.push_back(0);
    }
}

OceanTensor::Metadata::MetaIterator::MetaIterator(
        OceanTensor::Metadata::MetaIterator &&oth
) noexcept:
    curr_shape(oth.curr_shape),
    max_shape(oth.max_shape),
    end(oth.end)
{
    oth.max_shape.clear();
    oth.curr_shape.clear();
    oth.end = false;
}

OceanTensor::Metadata::MetaIterator::MetaIterator(
    const std::vector<int> &shape,
    std::initializer_list<int> indexes
):
    curr_shape(indexes),
    max_shape(shape)
{
    if (shape.size() != indexes.size()) {
        throw std::runtime_error("Out of bounds\n");
    }
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] <= curr_shape[i]) {
            throw std::runtime_error("Out of bounds\n");
        }
    }
}

OceanTensor::Metadata::MetaIterator &OceanTensor::Metadata::MetaIterator::operator++()
{
    size_t dim = max_shape.size();
    size_t carry = 0;

    if (end)
        return *this;
    curr_shape[max_shape.size() - 1]++;
    for (int i = dim - 1; i >= 0; i--) {
        if (carry) {
            curr_shape[i] += carry;
            carry = 0;
        }
        if (curr_shape[i] >= max_shape[i]) {
            carry = 1;
            curr_shape[i] = 0;
        }
    }
    if (carry)
        end = true;
    return *this;
}

std::ostream &operator<<(std::ostream &os, OceanTensor::Metadata::MetaIterator &it)
{
    auto &curr = it.getCurrShape();

    os << "Current shape: ";
    for (auto s: curr) {
        os << s << " ";
    }
    os << std::endl;
    return os;
}
