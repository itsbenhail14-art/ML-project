#include "linear.h"
#include "../math/algorithm/algo.h"
#include <random>
#include <cmath>

namespace nn {

Linear::Linear(size_t in_features, size_t out_features) 
    : W(in_features, out_features), b(out_features, 0.0f),
      dW(in_features, out_features, 0.0f), db(out_features, 0.0f),
      X_cache(0, 0) { // X_cache initialized empty, will be resized in forward pass
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_features));

    for (size_t i = 0; i < in_features; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            W(i, j) = dist(gen);
        }
    }
}

math::container::Matrix Linear::forward(const math::container::Matrix& X) {
    X_cache = X; // Cache input for backprop
    math::container::Matrix Z = math::algorithm::matmul(X, W);
    return math::algorithm::add_bias(Z, b);
}

math::container::Matrix Linear::backward(const math::container::Matrix& dL_dZ) {
    // 1. Calculate Weight Gradient: dW = X^T * dL_dZ
    math::container::Matrix X_T = math::algorithm::transpose(X_cache);
    dW = math::algorithm::matmul(X_T, dL_dZ);

    // 2. Calculate Bias Gradient: db = sum(dL_dZ)
    db = math::algorithm::sum_rows(dL_dZ);

    // 3. Calculate Input Gradient (to pass backward): dX = dL_dZ * W^T
    math::container::Matrix W_T = math::algorithm::transpose(W);
    return math::algorithm::matmul(dL_dZ, W_T);
}

} // namespace nn