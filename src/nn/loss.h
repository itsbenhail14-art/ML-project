#pragma once
#include "../math/container/tensor.h"

namespace nn {

class CrossEntropyLoss {
private:
    // Caching probabilities during the forward pass so we don't recalculate them in backward
    math::container::Matrix probs; 
    
public:
    // Initialize with dummy values so the compiler doesn't complain before first use
    CrossEntropyLoss() : probs(0, 0) {}

    // Forward pass: returns the scalar average loss of the batch
    float forward(const math::container::Matrix& logits, const math::container::Matrix& one_hot_labels);
    
    // Backward pass: returns the gradient dL/dZ to feed back into the network
    math::container::Matrix backward(const math::container::Matrix& one_hot_labels);
};

} // namespace nn