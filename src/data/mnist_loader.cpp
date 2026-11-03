#include "mnist_loader.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <stdexcept>

namespace data {

// MNIST stores integers in Big-Endian format. Modern PCs use Little-Endian. We must swap the bytes.
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

std::vector<Batch> MNISTLoader::load(const std::string& image_path, const std::string& label_path, size_t batch_size) {
    std::ifstream img_file(image_path, std::ios::binary);
    std::ifstream lbl_file(label_path, std::ios::binary);

    if (!img_file.is_open()) throw std::runtime_error("Cannot open MNIST image file: " + image_path);
    if (!lbl_file.is_open()) throw std::runtime_error("Cannot open MNIST label file: " + label_path);

    uint32_t magic_img, num_images, rows, cols;
    uint32_t magic_lbl, num_labels;

    // Read Image Header
    img_file.read(reinterpret_cast<char*>(&magic_img), 4);
    img_file.read(reinterpret_cast<char*>(&num_images), 4);
    img_file.read(reinterpret_cast<char*>(&rows), 4);
    img_file.read(reinterpret_cast<char*>(&cols), 4);
    
    // Read Label Header
    lbl_file.read(reinterpret_cast<char*>(&magic_lbl), 4);
    lbl_file.read(reinterpret_cast<char*>(&num_labels), 4);

    // Swap endianness
    num_images = swap_endian(num_images);
    num_labels = swap_endian(num_labels);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (num_images != num_labels) throw std::runtime_error("Image and label counts do not match!");
    if (rows * cols != 784) throw std::runtime_error("Images are not 28x28!");

    std::vector<Batch> batches;
    size_t num_batches = num_images / batch_size;

    for (size_t b = 0; b < num_batches; ++b) {
        Batch batch { 
            math::container::Matrix(batch_size, 784, 0.0f), 
            math::container::Matrix(batch_size, 10, 0.0f) 
        };

        for (size_t i = 0; i < batch_size; ++i) {
            // Read 784 pixels for this image
            for (size_t p = 0; p < 784; ++p) {
                unsigned char pixel;
                img_file.read(reinterpret_cast<char*>(&pixel), 1);
                batch.X(i, p) = static_cast<float>(pixel) / 255.0f; // Scale 0-255 to 0.0-1.0
            }

            // Read 1 label for this image and convert to one-hot (e.g., 3 -> [0,0,0,1,0,0,0,0,0,0])
            unsigned char label;
            lbl_file.read(reinterpret_cast<char*>(&label), 1);
            batch.Y(i, static_cast<size_t>(label)) = 1.0f;
        }
        batches.push_back(batch);
    }

    return batches;
}

} // namespace data