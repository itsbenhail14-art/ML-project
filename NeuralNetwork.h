#pragma once
#include "Matrix.h"
#include <cmath>
#include <fstream> // <--- Add this line
#include <string>  // <--- Add this line

// The Activation Function (Turns raw numbers into probabilities between 0 and 1)
inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// The Calculus Derivative of Sigmoid (Used to find the error gradient)
inline double d_sigmoid(double y) {
    return y * (1.0 - y);
}

class NeuralNetwork {
private:
    int input_nodes, hidden_nodes, output_nodes;
    double learning_rate;

    Matrix weights_ih; // Weights connecting Input to Hidden layer
    Matrix weights_ho; // Weights connecting Hidden to Output layer
    Matrix bias_h;     // Biases for Hidden layer
    Matrix bias_o;     // Biases for Output layer

public:
    // Constructor: Set up the network and randomize the starting weights
    NeuralNetwork(int inputs, int hidden, int outputs, double lr) 
        : input_nodes(inputs), hidden_nodes(hidden), output_nodes(outputs), 
          learning_rate(lr),
          weights_ih(hidden, inputs), weights_ho(outputs, hidden),
          bias_h(hidden, 1), bias_o(outputs, 1) 
    {
        weights_ih.randomize();
        weights_ho.randomize();
        bias_h.randomize();
        bias_o.randomize();
    }

    // 1. FORWARD PASS: Take a drawing and guess the number
    std::vector<double> predict(const std::vector<double>& input_array) {
        Matrix inputs = Matrix::fromArray(input_array);

        // Input to Hidden: (Inputs * Weights) + Bias -> Apply Sigmoid
        Matrix hidden = weights_ih.dot(inputs).add(bias_h);
        hidden = hidden.map(sigmoid);

        // Hidden to Output: (Hidden * Weights) + Bias -> Apply Sigmoid
        Matrix outputs = weights_ho.dot(hidden).add(bias_o);
        outputs = outputs.map(sigmoid);

        // Convert the final 1D Matrix column back to a standard C++ vector and return it
        std::vector<double> result(output_nodes);
        for (int i = 0; i < output_nodes; i++) {
            result[i] = outputs.data[i][0];
        }
        return result;
    }

    // 2. BACKWARD PASS: Make a guess, look at the right answer, and update weights
    void train(const std::vector<double>& input_array, const std::vector<double>& target_array) {
        // --- PART A: Make the Guess (Same as predict) ---
        Matrix inputs = Matrix::fromArray(input_array);
        Matrix hidden = weights_ih.dot(inputs).add(bias_h);
        hidden = hidden.map(sigmoid);
        Matrix outputs = weights_ho.dot(hidden).add(bias_o);
        outputs = outputs.map(sigmoid);
        Matrix targets = Matrix::fromArray(target_array);

        // --- PART B: Calculate Errors ---
        // Output Error = Target - Output
        Matrix output_errors = Matrix::subtract(targets, outputs);
        // Hidden Error = Weights (Hidden to Output) Transposed * Output Error
        Matrix weights_ho_t = weights_ho.transpose();
        Matrix hidden_errors = weights_ho_t.dot(output_errors);

        // --- PART C: Calculate Gradients & Update Output Weights ---
        Matrix output_gradients = outputs.map(d_sigmoid);          // Calculus derivative
        output_gradients = output_gradients.multiply(output_errors); // Multiply by error
        output_gradients = output_gradients.multiply(learning_rate); // Apply learning rate

        Matrix hidden_t = hidden.transpose();
        Matrix weight_ho_deltas = output_gradients.dot(hidden_t);
        
        weights_ho = weights_ho.add(weight_ho_deltas); // Update weights
        bias_o = bias_o.add(output_gradients);         // Update biases

        // --- PART D: Calculate Gradients & Update Hidden Weights ---
        Matrix hidden_gradients = hidden.map(d_sigmoid);
        hidden_gradients = hidden_gradients.multiply(hidden_errors);
        hidden_gradients = hidden_gradients.multiply(learning_rate);

        Matrix inputs_t = inputs.transpose();
        Matrix weight_ih_deltas = hidden_gradients.dot(inputs_t);
        
        weights_ih = weights_ih.add(weight_ih_deltas); // Update weights
        bias_h = bias_h.add(hidden_gradients);         // Update biases
    }
    // Save the trained weights to a text file
    bool saveNetwork(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        // Helper lambda function to write a matrix to the file
        auto writeMatrix = [&file](const Matrix& m) {
            for (int i = 0; i < m.rows; i++) {
                for (int j = 0; j < m.cols; j++) {
                    file << m.data[i][j] << " ";
                }
                file << "\n";
            }
        };

        writeMatrix(weights_ih);
        writeMatrix(weights_ho);
        writeMatrix(bias_h);
        writeMatrix(bias_o);

        file.close();
        return true;
    }

    // Load the trained weights from a text file
    bool loadNetwork(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        // Helper lambda function to read a matrix from the file
        auto readMatrix = [&file](Matrix& m) {
            for (int i = 0; i < m.rows; i++) {
                for (int j = 0; j < m.cols; j++) {
                    file >> m.data[i][j];
                }
            }
        };

        readMatrix(weights_ih);
        readMatrix(weights_ho);
        readMatrix(bias_h);
        readMatrix(bias_o);

        file.close();
        return true;
    }
};