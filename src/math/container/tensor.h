#pragma once
#include <vector>
#include <stdexcept>
#include <initializer_list>

namespace math {
namespace container {

class Vector {
public:
    std::vector<float> data;

    // Constructors
    Vector(size_t n, float val = 0.0f) : data(n, val) {}
    Vector(const std::initializer_list<float>& list) : data(list) {}

    size_t size() const { return data.size(); }

    // Array access
    float& operator[](size_t i) { return data[i]; }
    const float& operator[](size_t i) const { return data[i]; }
};


class Matrix {
public:
    size_t rows;
    size_t cols;
    std::vector<float> data; // Flattened 2D array for cache locality

    // Constructor
    Matrix(size_t r, size_t c, float val = 0.0f) : rows(r), cols(c), data(r * c, val) {}

    // 2D access using M(row, col)
    float& operator()(size_t r, size_t c) { 
        return data[r * cols + c]; 
    }
    const float& operator()(size_t r, size_t c) const { 
        return data[r * cols + c]; 
    }
};

} // namespace container
} // namespace math 