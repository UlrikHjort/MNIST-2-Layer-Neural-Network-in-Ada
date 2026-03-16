# MNIST Neural Network in Ada

A neural network implementation in Ada for handwritten digit recognition using the MNIST dataset. Simple 2-layer network (784->128->10)

## Overview

This project implements a feedforward neural network from scratch in Ada, demonstrating that Ada is well-suited for machine learning applications. The network achieves **~87% test accuracy** on the MNIST digit classification task.

## Architecture

### Network Structure
- **Input Layer**: 784 neurons (28*28 pixel flattened images)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one per digit 0-9)
- **Total Parameters**: ~101,000 trainable weights and biases

### Key Components

**Activation Functions:**
- **ReLU** (Rectified Linear Unit) for hidden layer: `f(x) = max(0, x)`
- **Softmax** for output layer: converts raw scores to probability distribution

**Loss Function:**
- Cross-entropy loss for multi-class classification

**Optimization:**
- Mini-batch Stochastic Gradient Descent (SGD)
- Backpropagation for gradient computation


## Requirements

- GNAT Ada compiler 


### Data Files
The MNIST dataset in IDX format:
- `train-images-idx3-ubyte` (60,000 training images)
- `train-labels-idx1-ubyte` (60,000 training labels)
- `t10k-images-idx3-ubyte` (10,000 test images)
- `t10k-labels-idx1-ubyte` (10,000 test labels)

Available (compressed) from: https://github.com/UlrikHjort/mnist/tree/master/data 

## Building and Running

### Compilation
```bash
gnatmake mnist_network.adb
```

### Execution
Place the MNIST data files in the "data" directory as same level as the executable:
```bash
./mnist_network
```

## Training Configuration

**Current Hyperparameters:**
- **Learning Rate**: 0.01
- **Batch Size**: 32
- **Epochs**: 5
- **Weight Initialization**: Uniform random [-0.05, 0.05]

## Performance Results


### Test Set Performance
- **Overall Accuracy**: 87.18% (8,718/10,000 correct)

### Per-Digit Accuracy
| Digit | Accuracy |
|-------|----------|
| 0     | 98.5%    |
| 1     | 84.1%    |
| 2     | 84.0%    |
| 3     | 86.3%    |
| 4     | 86.7%    |
| 5     | 86.0%    |
| 6     | 87.7%    |
| 7     | 86.6%    |
| 8     | 84.1%    |
| 9     | 86.6%    |

**Observations:**
- Digit 0 is easiest to classify (distinctive circular shape)
- Digits 1, 2, 8 are most challenging (confusion with 7, 3, 9)


## Code Structure

### Main Procedures
- `Initialize_Weights`: Random initialization of all parameters
- `Forward_Pass`: Computes network output for given input
- `Predict`: Returns predicted digit label
- `Train_Network`: Full training loop with backpropagation
- `Test_Network`: Evaluation on test set with detailed metrics

### Helper Functions
- `ReLU` / `ReLU_Derivative`: Activation function and gradient
- `Sigmoid`: Alternative activation (not currently used)
- `Read_Int32_BE`: Reads 32-bit big-endian integer from binary file
- `Load_Image`: Reads and normalizes 28*28 image
- `Load_Label`: Reads single digit label

## Limitations and Future Improvements

### Current Limitations
- No weight saving/loading (retrains from scratch each run)
- Fixed architecture (hard-coded layer sizes)
- Simple SGD optimizer (no momentum, Adam, etc.)
- No regularization (L2, dropout)
- No data augmentation
- Single hidden layer only

**Memory Usage:**
- Weight matrices: ~400KB (mostly W1: 128*784 floats)
- Activations: ~4KB per image
- Total: <1MB for network parameters


## References

- **[MNIST Database](https://en.wikipedia.org/wiki/MNIST_database)**

- **[IDX File Format](https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html)**

- **[Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)**

- **[GNAT](https://en.wikipedia.org/wiki/GNAT)**

- **[Ada](https://en.wikipedia.org/wiki/Ada_(programming_language))**