#pragma once
#include "../container/tensor.h"

namespace math {
namespace algorithm {

using container::Matrix;
using container::Vector;

// Neural Network Forward Pass Operations
Matrix matmul(const Matrix& A, const Matrix& B);
Matrix add_bias(const Matrix& X, const Vector& b);
Matrix relu(const Matrix& X);

} // namespace algorithm
} // namespace math