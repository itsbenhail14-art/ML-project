#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "NeuralNetwork.h"
#include <SFML/Graphics.hpp>

// Helper function to flip bytes (MNIST files are written in "Big Endian" format)
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Function to read the raw MNIST binary files
void readMNIST(std::string image_path, std::string label_path, 
               std::vector<std::vector<double>>& images, 
               std::vector<std::vector<double>>& labels) {
    
    std::ifstream file_img(image_path, std::ios::binary);
    std::ifstream file_lbl(label_path, std::ios::binary);

    if (!file_img.is_open() || !file_lbl.is_open()) {
        std::cout << "Error: Could not open MNIST files. Check your file paths!" << std::endl;
        return;
    }

    int magic_num = 0, num_images = 0, num_rows = 0, num_cols = 0;
    
    // Read Image headers
    file_img.read((char*)&magic_num, sizeof(magic_num));
    file_img.read((char*)&num_images, sizeof(num_images));
    file_img.read((char*)&num_rows, sizeof(num_rows));
    file_img.read((char*)&num_cols, sizeof(num_cols));
    
    num_images = reverseInt(num_images); // Convert to standard format

    // Read Label headers
    int magic_lbl = 0, num_labels = 0;
    file_lbl.read((char*)&magic_lbl, sizeof(magic_lbl));
    file_lbl.read((char*)&num_labels, sizeof(num_labels));
    num_labels = reverseInt(num_labels);

    std::cout << "Successfully found " << num_images << " images. Loading..." << std::endl;

    for (int i = 0; i < num_images; ++i) {
        std::vector<double> image(784);
        for (int p = 0; p < 784; ++p) {
            unsigned char pixel = 0;
            file_img.read((char*)&pixel, sizeof(pixel));
            // Normalize pixel values from 0-255 down to 0.0-1.0
            image[p] = (double)pixel / 255.0; 
        }
        images.push_back(image);

        unsigned char label = 0;
        file_lbl.read((char*)&label, sizeof(label));
        
        // One-Hot Encoding: If the label is '3', make an array where index 3 is 1.0 and the rest are 0.0
        std::vector<double> label_vec(10, 0.0);
        label_vec[label] = 1.0; 
        labels.push_back(label_vec);
        
        // TODO: Print the first image with partially lit block character
       // Print the first image with ANSI true-color brightness
        if (i == 0) {
            std::cout << "\n--- Preview of Image 1 (Label: " << (int)label << ") ---\n";
            for (int row = 0; row < 28; ++row) {
                for (int col = 0; col < 28; ++col) {
                    double val = image[row * 28 + col];
                    int brightness = static_cast<int>(val * 255);
                    
                    // \033[38;2;R;G;Bm sets the foreground True Color RGB
                    std::cout << "\033[38;2;" << brightness << ";" << brightness << ";" << brightness << "m██";
                }
                // Reset terminal color back to default at the end of each row
                std::cout << "\033[0m\n";
            }
            std::cout << "--------------------------------\n\n";
        }
    }

}

int main() {
    std::vector<std::vector<double>> images;
    std::vector<std::vector<double>> labels;

    // 1. Load the data (make sure these file names match what you download!)
    readMNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", images, labels);

    if (images.empty()) return 0; // Stop if files weren't found

    // 2. Initialize the AI: 784 pixels in, 128 hidden neurons, 10 possible outputs, 0.1 learning rate
    NeuralNetwork nn(784, 128, 10, 0.1); 
    
    // 3. Train the network on all 60,000 images
 // Load the pre-trained weights instantly
    if (!nn.loadNetwork("brain.txt")) {
        std::cout << "Error: Could not load brain.txt!" << std::endl;
        return -1;
    }
    std::cout << "Brain loaded successfully! Ready to predict." << std::endl;

  // --- SFML GRAPHICS WINDOW ---
    sf::RenderWindow window(sf::VideoMode({560, 560}), "Handwriting AI Canvas");
    
    // This array holds the 784 pixels the user draws, starting all black (0.0)
    std::vector<double> canvasData(784, 0.0);
    // Create a 20x20 square brush to draw on the screen
    sf::RectangleShape brush(sf::Vector2f({20.0f, 20.0f}));

    while (window.isOpen()) {
        while (const std::optional<sf::Event> event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            // Check if any key was released
            if (const auto* keyReleased = event->getIf<sf::Event::KeyReleased>()) {
                
                // Press Spacebar to clear the canvas
                if (keyReleased->code == sf::Keyboard::Key::Space) {
                    std::fill(canvasData.begin(), canvasData.end(), 0.0);
                    std::cout << "Canvas cleared!" << std::endl;
                }
                
                // Press Enter to make the AI guess the number
                if (keyReleased->code == sf::Keyboard::Key::Enter) {
                    // Pass the 784 pixels to the AI
                    std::vector<double> prediction = nn.predict(canvasData);
                    
                    // Find the number (0-9) with the highest probability
                    int bestGuess = 0;
                    double maxProb = prediction[0];
                    for (int i = 1; i < 10; i++) {
                        if (prediction[i] > maxProb) {
                            maxProb = prediction[i];
                            bestGuess = i;
                        }
                    }
                    std::cout << "\n===============================" << std::endl;
                    std::cout << "AI PREDICTION: " << bestGuess << std::endl;
                    std::cout << "CONFIDENCE: " << (maxProb * 100) << "%" << std::endl;
                    std::cout << "===============================\n" << std::endl;
                }
            }
        }

        // --- DRAWING LOGIC ---
        // If the left mouse button is held down, draw on the canvas
        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
            sf::Vector2i mousePos = sf::Mouse::getPosition(window);
            
            // Convert exact mouse pixel to the 28x28 grid
            int gridX = mousePos.x / 20;
            int gridY = mousePos.y / 20;

            // Make sure the mouse is actually inside the window
            if (gridX >= 0 && gridX < 28 && gridY >= 0 && gridY < 28) {
                // Set that pixel to 1.0 (White)
                canvasData[gridY * 28 + gridX] = 1.0;
                
                // Add a little bit of "fuzz" to surrounding pixels to make the line thicker 
                // (MNIST images usually have soft edges)
                if (gridX > 0) canvasData[gridY * 28 + (gridX - 1)] = std::max(canvasData[gridY * 28 + (gridX - 1)], 0.15);
                if (gridX < 27) canvasData[gridY * 28 + (gridX + 1)] = std::max(canvasData[gridY * 28 + (gridX + 1)], 0.15);
                if (gridY > 0) canvasData[(gridY - 1) * 28 + gridX] = std::max(canvasData[(gridY - 1) * 28 + gridX], 0.15);
                if (gridY < 27) canvasData[(gridY + 1) * 28 + gridX] = std::max(canvasData[(gridY + 1) * 28 + gridX], 0.15);
            }
        }

        // --- RENDERING LOGIC ---
        window.clear(sf::Color::Black); 
        
        // Loop through all 784 pixels and draw the ones that aren't black
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                double pixelValue = canvasData[y * 28 + x];
                if (pixelValue > 0.0) {
                    // Move the brush to the right spot (scaled by 20)
                    brush.setPosition({static_cast<float>(x * 20), static_cast<float>(y * 20)});
                    // Set color based on intensity (0.5 is gray, 1.0 is white)
                    int colorIntensity = static_cast<int>(pixelValue * 255);
                    brush.setFillColor(sf::Color(colorIntensity, colorIntensity, colorIntensity));
                    window.draw(brush);
                }
            }
        }

        window.display();               
    }
    return 0;
}