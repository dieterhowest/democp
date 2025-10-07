# Assignment 1: Generative Models for MNIST

This project implements a simple generative model for MNIST digit classification.

## Overview

The implementation learns per-pixel mean and variance for each digit class (0-9) and generates new digit images by sampling from class-conditioned Gaussian distributions.

## Features

- **Load MNIST Dataset**: Attempts to load the real MNIST dataset from OpenML, falls back to synthetic data if unavailable
- **Compute Statistics**: Calculates per-pixel mean and variance for each digit class (0-9)
- **Generate New Digits**: Samples from Gaussian distributions conditioned on digit class
- **Visualize Results**: Creates two visualizations:
  - `learned_statistics.png`: Shows the learned mean and variance for each digit
  - `generative_model_results.png`: Compares real samples with generated samples

## Requirements

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

Run the main script:

```bash
python main.py
```

This will:
1. Load or generate the MNIST-like dataset
2. Train the generative model by computing statistics
3. Generate new digit samples
4. Create visualization files

## How It Works

### Generative vs. Discriminative

This approach is **generative** because it models `P(X|Y)` - the probability of features given the class. It learns how to generate/create new examples of each digit class by modeling the underlying data distribution.

In contrast, a **discriminative** model would learn `P(Y|X)` - the probability of a class given features, focusing on decision boundaries between classes rather than data generation.

### Algorithm

1. **Training Phase**:
   - For each digit class (0-9), compute the mean and variance across all training samples
   - This gives us per-pixel statistics that characterize each digit

2. **Generation Phase**:
   - To generate a new sample of digit `d`:
     - Sample from `N(mean_d, variance_d)` for each pixel
     - Clip values to valid pixel range [0, 255]

### Results and Observations

**Most Realistic Digits**: Digits like '1' and '0' typically look more realistic because they have:
- Simpler, more consistent structure
- Less variation in how they're written
- Lower per-pixel variance

**Least Realistic Digits**: Digits like '3', '5', '8', and '9' often appear blurrier because:
- They have higher variance in pixel values
- More diverse writing styles across samples
- The Gaussian assumption fails to capture structured pixel relationships, leading to averaging effects

## Files

- `main.py`: Main implementation with all functionality
- `generative_model_results.png`: Visual comparison of real vs generated digits (generated)
- `learned_statistics.png`: Visualization of learned statistics (generated)
- `README.md`: This file

## Notes

- The implementation includes a synthetic data generator for demonstration when the real MNIST dataset is unavailable
- All generated images are automatically saved in the current directory
- Random seed is set to 42 for reproducibility
