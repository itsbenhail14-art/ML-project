#pragma once
#include "../container/tensor.h"

namespace math {
namespace algorithm {

using container::Matrix;
using container::Vector;

// Forward Pass Operations
Matrix matmul(const Matrix& A, const Matrix& B);
Matrix add_bias(const Matrix& X, const Vector& b);
Matrix relu(const Matrix& X);

// Backward Pass Operations
Matrix transpose(const Matrix& A);
Vector sum_rows(const Matrix& A);
Matrix relu_backward(const Matrix& A, const Matrix& dL_dA);

} // namespace algorithm
} // namespace math