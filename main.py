"""
Assignment 1: Generative Models

This script implements a simple generative model for MNIST digits.
It learns the per-pixel mean and variance for each digit class and generates
new digits by sampling from class-conditioned Gaussian distributions.

Reflection:
- Why is this approach generative instead of discriminative?
  This approach is generative because it models the distribution P(X|Y) 
  (features given class) rather than P(Y|X) (class given features). 
  A discriminative model would learn decision boundaries between classes,
  while this generative approach learns how to generate/create new examples
  of each class by modeling the underlying data distribution.

- Which digits look most/least realistic, and why?
  Digits like '1' and '0' tend to look more realistic because they have
  simpler, more consistent structure across samples. Digits like '3', '5', 
  '8', and '9' often look less realistic because they have more variability
  in how they're written and higher variance in pixel values, leading to
  blurrier generated images. The Gaussian assumption also doesn't capture
  the structured relationships between pixels very well.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


class GaussianGenerativeModel:
    """
    A simple generative model that learns per-pixel mean and variance
    for each digit class and generates new samples.
    """
    
    def __init__(self):
        self.means = {}
        self.variances = {}
        self.n_classes = 10
        
    def fit(self, X, y):
        """
        Compute per-pixel mean and variance for each digit class.
        
        Args:
            X: Images of shape (n_samples, n_pixels)
            y: Labels of shape (n_samples,)
        """
        print("Computing per-pixel statistics for each digit class...")
        for digit in range(self.n_classes):
            # Get all samples of this digit
            digit_samples = X[y == digit]
            
            # Compute mean and variance across samples
            self.means[digit] = np.mean(digit_samples, axis=0)
            self.variances[digit] = np.var(digit_samples, axis=0)
            
            # Add small epsilon to variance to avoid numerical issues
            self.variances[digit] += 1e-6
            
            print(f"  Digit {digit}: {len(digit_samples)} samples")
    
    def generate(self, digit, n_samples=1):
        """
        Generate new samples for a given digit class by sampling from
        a Gaussian distribution with learned mean and variance.
        
        Args:
            digit: The digit class to generate (0-9)
            n_samples: Number of samples to generate
            
        Returns:
            Generated samples of shape (n_samples, n_pixels)
        """
        if digit not in self.means:
            raise ValueError(f"Model has not been trained for digit {digit}")
        
        mean = self.means[digit]
        std = np.sqrt(self.variances[digit])
        
        # Sample from Gaussian distribution
        samples = np.random.normal(mean, std, size=(n_samples, len(mean)))
        
        # Clip to valid pixel range [0, 255]
        samples = np.clip(samples, 0, 255)
        
        return samples


def generate_synthetic_mnist():
    """
    Generate synthetic MNIST-like data for demonstration.
    Since external data access is limited, we create simplified digit patterns.
    Each digit has a characteristic pattern that we'll use to generate samples.
    """
    print("Generating synthetic MNIST-like dataset...")
    
    n_samples_per_digit = 100
    img_size = 28
    n_pixels = img_size * img_size
    
    X_list = []
    y_list = []
    
    for digit in range(10):
        for _ in range(n_samples_per_digit):
            # Create a blank image
            img = np.zeros((img_size, img_size))
            
            # Add noise
            noise = np.random.randn(img_size, img_size) * 10
            
            # Create digit patterns (simplified representations)
            if digit == 0:
                # Circle/oval
                for i in range(img_size):
                    for j in range(img_size):
                        dist_center = np.sqrt((i - 14)**2 + (j - 14)**2)
                        if 8 < dist_center < 11:
                            img[i, j] = 200 + np.random.randn() * 30
            
            elif digit == 1:
                # Vertical line
                for i in range(5, 23):
                    j = 14 + np.random.randint(-1, 2)
                    img[i, j] = 200 + np.random.randn() * 30
                    
            elif digit == 2:
                # Top horizontal, diagonal, bottom horizontal
                for j in range(7, 21):
                    img[6, j] = 200 + np.random.randn() * 30
                    img[22, j] = 200 + np.random.randn() * 30
                for i in range(6, 22):
                    j = int(7 + (i - 6) * 0.8)
                    img[i, j] = 200 + np.random.randn() * 30
                    
            elif digit == 3:
                # Two horizontal lines and a curve on right
                for j in range(7, 18):
                    img[6, j] = 200 + np.random.randn() * 30
                    img[14, j] = 200 + np.random.randn() * 30
                    img[22, j] = 200 + np.random.randn() * 30
                for i in range(6, 23):
                    img[i, 17] = 200 + np.random.randn() * 30
                    
            elif digit == 4:
                # Vertical line on right and diagonal
                for i in range(10, 23):
                    img[i, 16] = 200 + np.random.randn() * 30
                for i in range(5, 15):
                    j = int(5 + (15 - i) * 0.7)
                    img[i, j] = 200 + np.random.randn() * 30
                for j in range(5, 17):
                    img[14, j] = 200 + np.random.randn() * 30
                    
            elif digit == 5:
                # Top horizontal, middle horizontal, bottom curve
                for j in range(7, 21):
                    img[6, j] = 200 + np.random.randn() * 30
                    img[14, j] = 200 + np.random.randn() * 30
                for i in range(6, 15):
                    img[i, 7] = 200 + np.random.randn() * 30
                for i in range(14, 23):
                    img[i, 20] = 200 + np.random.randn() * 30
                    
            elif digit == 6:
                # Circle with vertical line on left
                for i in range(6, 23):
                    img[i, 7] = 200 + np.random.randn() * 30
                for i in range(14, 23):
                    for j in range(img_size):
                        dist = np.sqrt((i - 18)**2 + (j - 14)**2)
                        if 5 < dist < 7:
                            img[i, j] = 200 + np.random.randn() * 30
                            
            elif digit == 7:
                # Horizontal top and diagonal
                for j in range(7, 21):
                    img[6, j] = 200 + np.random.randn() * 30
                for i in range(6, 23):
                    j = int(20 - (i - 6) * 0.5)
                    img[i, j] = 200 + np.random.randn() * 30
                    
            elif digit == 8:
                # Two circles stacked
                for i in range(img_size):
                    for j in range(img_size):
                        dist_top = np.sqrt((i - 10)**2 + (j - 14)**2)
                        dist_bottom = np.sqrt((i - 18)**2 + (j - 14)**2)
                        if (4 < dist_top < 6) or (4 < dist_bottom < 6):
                            img[i, j] = 200 + np.random.randn() * 30
                            
            elif digit == 9:
                # Circle with vertical line on right
                for i in range(6, 23):
                    img[i, 20] = 200 + np.random.randn() * 30
                for i in range(6, 15):
                    for j in range(img_size):
                        dist = np.sqrt((i - 10)**2 + (j - 14)**2)
                        if 5 < dist < 7:
                            img[i, j] = 200 + np.random.randn() * 30
            
            # Add the noise
            img = np.clip(img + noise, 0, 255)
            
            X_list.append(img.flatten())
            y_list.append(digit)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Shuffle the dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"Dataset generated: {X.shape[0]} samples, {X.shape[1]} features")
    print("Note: This is synthetic data for demonstration purposes.")
    return X, y


def load_mnist():
    """
    Load MNIST dataset. Falls back to synthetic data if external access is unavailable.
    """
    try:
        print("Attempting to load MNIST dataset from OpenML...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.to_numpy()
        y = mnist.target.to_numpy().astype(int)
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    except Exception as e:
        print(f"Could not load MNIST from OpenML: {e}")
        print("Falling back to synthetic dataset...")
        return generate_synthetic_mnist()


def visualize_results(model, X_real, y_real, n_samples_per_digit=5):
    """
    Visualize both real and generated samples for each digit class.
    
    Args:
        model: Trained generative model
        X_real: Real MNIST images
        y_real: Real MNIST labels
        n_samples_per_digit: Number of samples to show per digit
    """
    fig, axes = plt.subplots(model.n_classes, n_samples_per_digit * 2, 
                            figsize=(15, 12))
    fig.suptitle('MNIST Generative Model Results\n'
                'Left columns: Real samples | Right columns: Generated samples',
                fontsize=14, y=0.995)
    
    for digit in range(model.n_classes):
        # Get real samples for this digit
        digit_samples = X_real[y_real == digit]
        real_indices = np.random.choice(len(digit_samples), 
                                       n_samples_per_digit, 
                                       replace=False)
        
        # Generate new samples
        generated_samples = model.generate(digit, n_samples_per_digit)
        
        # Display real and generated samples side by side
        for i in range(n_samples_per_digit):
            # Real sample
            ax_real = axes[digit, i * 2]
            real_img = digit_samples[real_indices[i]].reshape(28, 28)
            ax_real.imshow(real_img, cmap='gray')
            ax_real.axis('off')
            if i == 0:
                ax_real.set_title(f'Digit {digit}\nReal', fontsize=10)
            
            # Generated sample
            ax_gen = axes[digit, i * 2 + 1]
            gen_img = generated_samples[i].reshape(28, 28)
            ax_gen.imshow(gen_img, cmap='gray')
            ax_gen.axis('off')
            if i == 0:
                ax_gen.set_title(f'Generated', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('generative_model_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'generative_model_results.png'")
    plt.close()


def visualize_statistics(model):
    """
    Visualize the learned mean and variance for each digit class.
    """
    fig, axes = plt.subplots(2, model.n_classes, figsize=(15, 4))
    fig.suptitle('Learned Per-Pixel Statistics for Each Digit Class',
                fontsize=14)
    
    for digit in range(model.n_classes):
        # Mean
        ax_mean = axes[0, digit]
        mean_img = model.means[digit].reshape(28, 28)
        ax_mean.imshow(mean_img, cmap='gray')
        ax_mean.axis('off')
        ax_mean.set_title(f'Digit {digit}\nMean', fontsize=10)
        
        # Variance
        ax_var = axes[1, digit]
        var_img = model.variances[digit].reshape(28, 28)
        ax_var.imshow(var_img, cmap='hot')
        ax_var.axis('off')
        ax_var.set_title(f'Variance', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('learned_statistics.png', dpi=150, bbox_inches='tight')
    print("Statistics visualization saved to 'learned_statistics.png'")
    plt.close()


def main():
    """Main function to run the generative model experiment."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load MNIST dataset
    X, y = load_mnist()
    
    # Create and train the generative model
    model = GaussianGenerativeModel()
    model.fit(X, y)
    
    print("\nGenerating samples and creating visualizations...")
    
    # Visualize learned statistics (mean and variance)
    visualize_statistics(model)
    
    # Visualize real vs generated samples
    visualize_results(model, X, y, n_samples_per_digit=5)
    
    print("\n" + "="*70)
    print("REFLECTION")
    print("="*70)
    print("\n1. Why is this approach generative instead of discriminative?")
    print("   This approach models P(X|Y) - the probability of features given")
    print("   the class. It learns how to generate/create new examples of each")
    print("   digit class. In contrast, a discriminative model would learn")
    print("   P(Y|X) - the probability of a class given features, focusing on")
    print("   decision boundaries between classes rather than data generation.")
    
    print("\n2. Which digits look most/least realistic, and why?")
    print("   - MOST realistic: Digits '1' and '0' typically look better because")
    print("     they have simpler, more consistent structure with less variation")
    print("     in how people write them.")
    print("   - LEAST realistic: Digits like '3', '5', '8', and '9' often appear")
    print("     blurrier because they have higher variance in pixel values and")
    print("     more diverse writing styles. The Gaussian assumption fails to")
    print("     capture structured pixel relationships, leading to averaging effects.")
    print("="*70)


if __name__ == "__main__":
    main()
