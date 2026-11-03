#pragma once
#include "linear.h"
#include "../math/algorithm/algo.h"

namespace nn {

class NeuralNetwork {
public:
    Linear fc1;
    Linear fc2;
    Linear fc3;

    // 784 inputs -> 32 hidden -> 32 hidden -> 10 output classes
    NeuralNetwork() : fc1(784, 32), fc2(32, 32), fc3(32, 10) {}
    
    math::container::Matrix forward(const math::container::Matrix& X);
};

} // namespace nn