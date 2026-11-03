#include <iostream>
#include <string>
#include "math/container/tensor.h"
#include "nn/network.h"

void print_shape(const math::container::Matrix& m, const std::string& name) {
    std::cout << name << " shape: (" << m.rows << "x" << m.cols << ")\n";
}

int main() {
    // Simulate a batch of 32 MNIST images (32 samples, 784 pixels each)
    math::container::Matrix batch_X = {32, 784, 0.1f}; // 0.1f is dummy pixel data
    print_shape(batch_X, "Input Batch");

    // Initialize the Neural Network
    nn::NeuralNetwork model;

    // Run the forward pass
    math::container::Matrix logits = model.forward(batch_X);

    // The output should perfectly match (batch_size, num_classes) -> (32, 10)
    print_shape(logits, "Output Logits");

    // Print the raw logit scores for the first image in the batch
    std::cout << "\nLogits for image 0:\n";
    for(size_t j = 0; j < 10; ++j) {
        std::cout << logits(0, j) << " ";
    }
    std::cout << "\n";

    return 0;
}