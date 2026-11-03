#include "loss.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace nn {

float CrossEntropyLoss::forward(const math::container::Matrix& logits, const math::container::Matrix& one_hot_labels) {
    if (logits.rows != one_hot_labels.rows || logits.cols != one_hot_labels.cols) {
        throw std::invalid_argument("Logits and labels dimensions must match.");
    }

    size_t batch_size = logits.rows;
    size_t num_classes = logits.cols;
    
    // Initialize probs matrix cache
    probs = math::container::Matrix(batch_size, num_classes);
    float total_loss = 0.0f;

    for (size_t i = 0; i < batch_size; ++i) {
        // 1. Find max logit in this row for numerical stability
        // (Prevents std::exp from blowing up to infinity)
        float max_val = logits(i, 0);
        for (size_t j = 1; j < num_classes; ++j) {
            max_val = std::max(max_val, logits(i, j));
        }

        // 2. Compute exponentials and their sum
        float sum_exp = 0.0f;
        for (size_t j = 0; j < num_classes; ++j) {
            probs(i, j) = std::exp(logits(i, j) - max_val);
            sum_exp += probs(i, j);
        }

        // 3. Normalize into probabilities and calculate cross-entropy
        for (size_t j = 0; j < num_classes; ++j) {
            probs(i, j) /= sum_exp;
            
            // Only calculate loss for the correct class label (where label is 1.0)
            if (one_hot_labels(i, j) > 0.0f) {
                // Add 1e-7f to prevent std::log(0) which causes NaN errors
                total_loss -= one_hot_labels(i, j) * std::log(probs(i, j) + 1e-7f);
            }
        }
    }

    // Return the average loss across the whole batch
    return total_loss / static_cast<float>(batch_size);
}

math::container::Matrix CrossEntropyLoss::backward(const math::container::Matrix& one_hot_labels) {
    size_t batch_size = probs.rows;
    size_t num_classes = probs.cols;
    
    math::container::Matrix dL_dZ(batch_size, num_classes);
    
    // The gradient of Softmax + Cross Entropy simplifies beautifully to (P - Y) / N
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            dL_dZ(i, j) = (probs(i, j) - one_hot_labels(i, j)) / static_cast<float>(batch_size);
        }
    }
    
    return dL_dZ;
}

} // namespace nn