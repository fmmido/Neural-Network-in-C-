#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>  // For srand and rand

class NeuralNetwork {
private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    std::vector<double> biasInputHidden;
    std::vector<double> biasHiddenOutput;

public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
        // Initialize weights and biases with random values
        weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
        weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
        biasInputHidden.resize(hiddenSize);
        biasHiddenOutput.resize(outputSize);

        // Random initialization
        initializeWeights(weightsInputHidden);
        initializeWeights(weightsHiddenOutput);
        initializeBiases(biasInputHidden);
        initializeBiases(biasHiddenOutput);
    }


    void initializeWeights(std::vector<std::vector<double>>& weights) {
        for (auto& row : weights) {
            for (double& weight : row) {
                weight = getRandomWeight();
            }
        }
    }

    void initializeBiases(std::vector<double>& biases) {
        for (double& bias : biases) {
            bias = getRandomWeight();
        }
    }

    double getRandomWeight() {
        return ((double)rand() / RAND_MAX) * 2 - 1; // Random value between -1 and 1
    }

    std::vector<double> sigmoid(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = 1 / (1 + exp(-x[i]));
        }
        return result;
    }

    std::vector<double> feedforward(const std::vector<double>& inputs) {
        
        // Calculate hidden layer output
        std::vector<double> hiddenOutput(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i) {
            double sum = 0;
            for (int j = 0; j < inputSize; ++j) {
                sum += inputs[j] * weightsInputHidden[j][i];
            }
            sum += biasInputHidden[i];
            hiddenOutput[i] = sum;
        }
        hiddenOutput = sigmoid(hiddenOutput);

        // Calculate output layer output
        std::vector<double> output(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            double sum = 0;
            for (int j = 0; j < hiddenSize; ++j) {
                sum += hiddenOutput[j] * weightsHiddenOutput[j][i];
            }
            sum += biasHiddenOutput[i];
            output[i] = sum;
        }
        output = sigmoid(output);

        return output;
    }
};

int main() {
    srand(123); // Seed random number generator with an integer value

    // Define neural network parameters
    int inputSize = 2;
    int hiddenSize = 3;
    int outputSize = 1;

    // Create neural network
    NeuralNetwork nn(inputSize, hiddenSize, outputSize);

    // Define input
    std::vector<double> inputs = {0.5, 0.3};

    // Perform feedforward pass
    std::vector<double> output = nn.feedforward(inputs);

    // Display output
    std::cout << "Output: ";
    for (double val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
