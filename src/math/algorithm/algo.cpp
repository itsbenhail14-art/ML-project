#include "algo.h"
#include <stdexcept>
#include <algorithm>

namespace math {
namespace algorithm {

Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("MatMul dimension mismatch: A.cols must equal B.rows");
    }
    
    Matrix C(A.rows, B.cols, 0.0f);
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t k = 0; k < A.cols; ++k) {
            float a_ik = A(i, k);
            for (size_t j = 0; j < B.cols; ++j) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }
    return C;
}

Matrix add_bias(const Matrix& X, const Vector& b) {
    if (X.cols != b.size()) {
        throw std::invalid_argument("Bias dimension mismatch: b.size() must equal X.cols");
    }
    
    Matrix Y(X.rows, X.cols);
    for (size_t i = 0; i < X.rows; ++i) {
        for (size_t j = 0; j < X.cols; ++j) {
            Y(i, j) = X(i, j) + b[j];
        }
    }
    return Y;
}

Matrix relu(const Matrix& X) {
    Matrix Y(X.rows, X.cols);
    for (size_t i = 0; i < X.data.size(); ++i) {
        Y.data[i] = std::max(0.0f, X.data[i]);
    }
    return Y;
}

Matrix transpose(const Matrix& A) {
    Matrix T(A.cols, A.rows);
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < A.cols; ++j) {
            T(j, i) = A(i, j);
        }
    }
    return T;
}

Vector sum_rows(const Matrix& A) {
    Vector v(A.cols, 0.0f);
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < A.cols; ++j) {
            v[j] += A(i, j);
        }
    }
    return v;
}

Matrix relu_backward(const Matrix& A, const Matrix& dL_dA) {
    Matrix dZ(A.rows, A.cols, 0.0f);
    for (size_t i = 0; i < A.data.size(); ++i) {
        dZ.data[i] = (A.data[i] > 0.0f) ? dL_dA.data[i] : 0.0f;
    }
    return dZ;
}

} // namespace algorithm
} // namespace math