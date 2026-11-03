#include "linear.h"
#include "../math/algorithm/algo.h"
#include <random>
#include <cmath>

namespace nn {

Linear::Linear(size_t in_features, size_t out_features) 
    : W(in_features, out_features), b(out_features, 0.0f) {
    
    // Initialize weights using standard normal distribution scaled by Kaiming initialization
    std::mt19937 gen(42); // Fixed seed for reproducible debugging
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_features));

    for (size_t i = 0; i < in_features; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            W(i, j) = dist(gen);
        }
    }
}

math::container::Matrix Linear::forward(const math::container::Matrix& X) const {
    math::container::Matrix Z = math::algorithm::matmul(X, W);
    return math::algorithm::add_bias(Z, b);
}

} // namespace nn