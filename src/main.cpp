#include <iostream>
#include <string>
#include "math/container/tensor.h"
#include "math/algorithm/algo.h"

// Helper function to visualize the matrices in the terminal
void print_matrix(const math::container::Matrix& m, const std::string& name) {
    std::cout << "--- " << name << " (" << m.rows << "x" << m.cols << ") ---\n";
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            std::cout << m(i, j) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // 1. Initialize Input Matrix X (Simulating a batch of 2 samples, 2 features each)
    math::container::Matrix X(2, 2);
    X(0, 0) = 1.0f;  X(0, 1) = 2.0f;
    X(1, 0) = -1.0f; X(1, 1) = 0.5f;

    print_matrix(X, "Input X");

    // 2. Initialize Weights Matrix W (Mapping 2 input features to 3 output features)
    math::container::Matrix W(2, 3);
    W(0, 0) = 0.5f; W(0, 1) = -1.0f; W(0, 2) = 2.0f;
    W(1, 0) = 1.0f; W(1, 1) = 0.5f;  W(1, 2) = -1.0f;

    print_matrix(W, "Weights W");

    // 3. Initialize Bias Vector b (3 elements, one for each output feature)
    // Utilizing the std::initializer_list constructor from your Vector class
    math::container::Vector b = {0.5f, -0.5f, 1.0f}; 

    // 4. Test Forward Pass Step 1: Matrix Multiplication (X @ W)
    math::container::Matrix Z = math::algorithm::matmul(X, W);
    print_matrix(Z, "Matmul Output (X * W)");

    // 5. Test Forward Pass Step 2: Bias Addition (Broadcasting)
    math::container::Matrix Z_bias = math::algorithm::add_bias(Z, b);
    print_matrix(Z_bias, "Biased Output (Z + b)");

    // 6. Test Forward Pass Step 3: ReLU Activation
    // Any negative values from Z_bias should become 0.0
    math::container::Matrix A = math::algorithm::relu(Z_bias);
    print_matrix(A, "Activated Output (ReLU)");

    return 0;
}