#include <iostream>
#include <vector>
#include <string>
#include <fstream>   // Fixes 'file' errors
#include <cmath>     // Fixes 'exp' errors
#include <algorithm> // For std::max and std::sort
#include <iomanip>   // For std::setprecision formatting
#include <utility>   // Fixes std::pair push_back errors

#include "data/mnist_loader.h"
#include "nn/network.h"
#include "nn/loss.h"
#include "nn/optimizer.h"

int main() {
    // 1. Hyperparameters
    const size_t batch_size = 64;
    const float learning_rate = 0.05f;
    const int epochs = 10;

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
    // --- EXHIBITION MODE ---
    std::cout << "\n=== EXHIBITION MODE ===\n";
    std::cout << "Ready for live inputs. Draw a number and save it as 'drawn.bin'.\n";

    // Clear the input buffer from any previous artifacts before starting the loop
    std::cin.ignore(256, '\n'); 

    while (true) {
        std::cout << "\nPress Enter to predict 'drawn.bin' (or type 'q' to quit)...";
        std::string input;
        std::getline(std::cin, input);
        if (input == "q") break;

        std::ifstream file("drawn.bin", std::ios::binary);
        if (!file) {
            std::cout << "Could not find 'drawn.bin'. Draw a number in the GUI first!\n";
            continue;
        }

        // Load the 784-byte image
        math::container::Matrix X_live(1, 784, 0.0f);
        for (size_t i = 0; i < 784; ++i) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            X_live(0, i) = static_cast<float>(pixel) / 255.0f;
        }
        file.close();

        // 1. Predict
        math::container::Matrix logits = model.forward(X_live);

        // 2. Calculate Softmax Confidence percentages manually
        float max_logit = logits(0, 0);
        for (size_t i = 1; i < 10; ++i) {
            max_logit = std::max(max_logit, logits(0, i));
        }

        float sum_exp = 0.0f;
        std::vector<std::pair<float, int>> confidences;
        for (size_t i = 0; i < 10; ++i) {
            float exp_val = std::exp(logits(0, i) - max_logit);
            confidences.push_back({exp_val, i});
            sum_exp += exp_val;
        }

        for (auto& p : confidences) {
            p.first = (p.first / sum_exp) * 100.0f;
        }

        // Sort descending to get 1st, 2nd, 3rd guesses
        std::sort(confidences.rbegin(), confidences.rend());

        std::cout << "\n--- PREDICTION RESULTS ---\n";
        std::cout << "1st Guess: " << confidences[0].second << " (" << std::fixed << std::setprecision(2) << confidences[0].first << "%)\n";
        std::cout << "2nd Guess: " << confidences[1].second << " (" << confidences[1].first << "%)\n";
        std::cout << "3rd Guess: " << confidences[2].second << " (" << confidences[2].first << "%)\n";

        // 3. Ask for the ground truth to calculate the Loss
        std::cout << "\nWhat number did the judge actually draw? (0-9): ";
        int true_label;
        std::cin >> true_label;
        std::cin.ignore(256, '\n'); // Clear the newline character for the next loop

        math::container::Matrix Y_live(1, 10, 0.0f);
        Y_live(0, true_label) = 1.0f;
        
        // Use your existing CrossEntropyLoss class to compute the error
        float live_loss = loss_fn.forward(logits, Y_live);
        std::cout << "Cross-Entropy Loss for this try: " << live_loss << "\n";
    }
  
    return 0;
}