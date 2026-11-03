#pragma once
#include "linear.h"
#include "../math/algorithm/algo.h"

namespace nn {

class NeuralNetwork {
public:
    Linear fc1;
    Linear fc2;
    Linear fc3;

    // Caches to hold activations for the backward pass
    math::container::Matrix A1_cache;
    math::container::Matrix A2_cache;

    // 784 inputs -> 32 hidden -> 32 hidden -> 10 output classes
    NeuralNetwork() : fc1(784, 32), fc2(32, 32), fc3(32, 10), 
                      A1_cache(0,0), A2_cache(0,0) {}

    math::container::Matrix forward(const math::container::Matrix& X);
    
    // The missing backward declaration!
    void backward(const math::container::Matrix& dL_dZ3);
};

} // namespace nn