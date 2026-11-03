#include <iostream>
#include <vector>
#include <string>
#include "data/mnist_loader.h"
#include "nn/network.h"
#include "nn/loss.h"
#include "nn/optimizer.h"

int main() {
    // 1. Hyperparameters
    const size_t batch_size = 64;
    const float learning_rate = 0.05f;
    const int epochs = 5;

    // --- PUT THE PATHS RIGHT HERE ---
    // The "../" moves up out of the 'build' folder, then into 'dataset'
    const std::string img_path = "../dataset/train-images.idx3-ubyte";
    const std::string lbl_path = "../dataset/train-labels.idx1-ubyte";
    // --------------------------------

    std::cout << "Loading MNIST dataset...\n";
    std::vector<data::Batch> batches;
    try {
        batches = data::MNISTLoader::load(img_path, lbl_path, batch_size);
        std::cout << "Loaded " << batches.size() << " batches of size " << batch_size << ".\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << "\n";
        return 1;
    }

    // 2. Initialize the architecture
    nn::NeuralNetwork model;
    nn::CrossEntropyLoss loss_fn;
    nn::SGD optimizer(learning_rate);

    std::cout << "Starting training loop (784 -> 32 -> 32 -> 10)...\n";
    std::cout << "Learning Rate: " << learning_rate << " | Epochs: " << epochs << "\n\n";

    // 3. The Training Loop
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int correct_predictions = 0;
        int total_samples = 0;

        for (size_t i = 0; i < batches.size(); ++i) {
            const auto& batch = batches[i];

            // --- FORWARD PASS ---
            math::container::Matrix logits = model.forward(batch.X);
            float batch_loss = loss_fn.forward(logits, batch.Y);
            epoch_loss += batch_loss;

            // --- METRICS (Accuracy Calculation) ---
            for (size_t r = 0; r < batch_size; ++r) {
                int pred_class = 0;
                int true_class = 0;
                float max_prob = -1.0f;
                
                for (size_t c = 0; c < 10; ++c) {
                    if (logits(r, c) > max_prob) {
                        max_prob = logits(r, c);
                        pred_class = c;
                    }
                    if (batch.Y(r, c) > 0.5f) {
                        true_class = c;
                    }
                }
                if (pred_class == true_class) {
                    correct_predictions++;
                }
                total_samples++;
            }

            // --- BACKWARD PASS ---
            math::container::Matrix dL_dZ = loss_fn.backward(batch.Y);
            model.backward(dL_dZ);

            // --- OPTIMIZE ---
            optimizer.step(model);
        }

        // Print epoch statistics
        float avg_loss = epoch_loss / batches.size();
        float accuracy = (static_cast<float>(correct_predictions) / total_samples) * 100.0f;

        std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                  << "Loss: " << avg_loss << " | "
                  << "Accuracy: " << accuracy << "%\n";
    }

    std::cout << "\nTraining complete!\n";
    return 0;
}