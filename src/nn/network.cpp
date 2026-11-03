#include "network.h"

math::container::Matrix nn::NeuralNetwork::forward(const math::container::Matrix& X) {
    // First Hidden Layer + ReLU
    math::container::Matrix Z1 = fc1.forward(X);
    math::container::Matrix A1 = math::algorithm::relu(Z1);
    
    // Second Hidden Layer + ReLU
    math::container::Matrix Z2 = fc2.forward(A1);
    math::container::Matrix A2 = math::algorithm::relu(Z2);
    
    // Output Layer (Raw logits, no activation)
    math::container::Matrix Z3 = fc3.forward(A2);
    return Z3;
}