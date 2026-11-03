#pragma once
#include "../math/container/tensor.h"

namespace nn {

class Linear {
public:
    math::container::Matrix W;
    math::container::Vector b;

    Linear(size_t in_features, size_t out_features);
    math::container::Matrix forward(const math::container::Matrix& X) const;
};

} // namespace nn           