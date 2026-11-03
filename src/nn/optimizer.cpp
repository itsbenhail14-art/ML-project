#include "optimizer.h"

namespace nn {

SGD::SGD(float lr) : learning_rate(lr) {}

// A helper function to update a single layer's weights and biases
void update_layer(Linear& layer, float lr) {
    // W = W - (learning_rate * dW)
    for (size_t i = 0; i < layer.W.data.size(); ++i) {
        layer.W.data[i] -= lr * layer.dW.data[i];
    }
    
    // b = b - (learning_rate * db)
    for (size_t i = 0; i < layer.b.data.size(); ++i) {
        layer.b.data[i] -= lr * layer.db.data[i];
    }
}

void SGD::step(NeuralNetwork& model) {
    update_layer(model.fc1, learning_rate);
    update_layer(model.fc2, learning_rate);
    update_layer(model.fc3, learning_rate);
}

} // namespace nn