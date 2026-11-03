#pragma once
#include "../math/container/tensor.h"
#include <string>
#include <vector>

namespace data {

struct Batch {
    math::container::Matrix X; // Images: shape (batch_size, 784)
    math::container::Matrix Y; // Labels: shape (batch_size, 10) one-hot encoded
};

class MNISTLoader {
public:
    // Reads the raw binary files, scales pixels to 0-1, one-hot encodes labels, and chunks into batches
    static std::vector<Batch> load(const std::string& image_path, const std::string& label_path, size_t batch_size);
};

} // namespace data