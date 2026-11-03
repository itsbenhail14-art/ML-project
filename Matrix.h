#pragma once
#include <vector>
#include <random>
#include <stdexcept>

class Matrix {
public:
    int rows, cols;
    std::vector<std::vector<double>> data;

    // Constructor: creates a matrix of size r x c, filled with 0.0
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    // Fills the matrix with random weights between -1.0 and 1.0
    void randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = dis(gen);
            }
        }
    }

    // Matrix Addition
    Matrix add(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Dimensions must match for addition");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // Matrix Multiplication (Dot Product)
    Matrix dot(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Columns of A must match rows of B");
        }
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    // Transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    // Matrix Subtraction (Target - Guess)
    static Matrix subtract(const Matrix& a, const Matrix& b) {
        Matrix result(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }

    // Element-wise Multiplication (Used for calculating gradients)
    Matrix multiply(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    // Scalar Multiplication (Used to multiply by the learning rate)
    Matrix multiply(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    // Map: Applies a math function (like Sigmoid) to every number in the matrix
    Matrix map(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    // Helper to convert std::vector to a 1D Matrix (Column Vector)
    static Matrix fromArray(const std::vector<double>& arr) {
        Matrix m(arr.size(), 1);
        for (size_t i = 0; i < arr.size(); i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }
};