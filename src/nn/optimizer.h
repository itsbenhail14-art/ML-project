#pragma once
#include "network.h"

namespace nn {

class SGD {
public:
    float learning_rate;

    // Default learning rate of 0.01 is a standard starting point for MNIST
    SGD(float lr = 0.01f);
    
    // Takes the network by reference so it modifies the actual weights in memory
    void step(NeuralNetwork& model);
};

} // namespace nn