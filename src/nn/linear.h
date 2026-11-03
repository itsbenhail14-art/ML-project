#pragma once
#include "../math/container/tensor.h"

namespace nn {

class Linear {
public:
    math::container::Matrix W;
    math::container::Vector b;
    
    // Gradient storage
    math::container::Matrix dW;
    math::container::Vector db;

    // Cache for the backward pass
    math::container::Matrix X_cache;

    Linear(size_t in_features, size_t out_features);
    
    // Note: Removed 'const' because forward() now modifies X_cache
    math::container::Matrix forward(const math::container::Matrix& X);
    
    // Calculates dW, db, and returns dX for the previous layer
    math::container::Matrix backward(const math::container::Matrix& dL_dZ);
};

} // namespace nn