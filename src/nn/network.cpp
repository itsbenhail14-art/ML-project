#include "network.h"

namespace nn {

math::container::Matrix NeuralNetwork::forward(const math::container::Matrix& X) {
    // First Hidden Layer + ReLU
    math::container::Matrix Z1 = fc1.forward(X);
    A1_cache = math::algorithm::relu(Z1); // Cache for backprop
    
    // Second Hidden Layer + ReLU
    math::container::Matrix Z2 = fc2.forward(A1_cache);
    A2_cache = math::algorithm::relu(Z2); // Cache for backprop
    
    // Output Layer (Raw logits, no activation)
    math::container::Matrix Z3 = fc3.forward(A2_cache);
    return Z3;
}

// The missing backward implementation!
void NeuralNetwork::backward(const math::container::Matrix& dL_dZ3) {
    // Output layer backward
    math::container::Matrix dL_dA2 = fc3.backward(dL_dZ3);
    
    // Second hidden layer backward (through ReLU, then Linear)
    math::container::Matrix dL_dZ2 = math::algorithm::relu_backward(A2_cache, dL_dA2);
    math::container::Matrix dL_dA1 = fc2.backward(dL_dZ2);
    
    // First hidden layer backward (through ReLU, then Linear)
    math::container::Matrix dL_dZ1 = math::algorithm::relu_backward(A1_cache, dL_dA1);
    fc1.backward(dL_dZ1); // We don't need the returned dX from the first layer
}

} // namespace nn