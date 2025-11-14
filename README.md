# PathMNIST Classification

A comparative study of machine learning approaches for medical image classification using the PathMNIST dataset.

## Project Overview

This project evaluates and compares three different machine learning models for classifying histopathology images:
- **Decision Tree** (Classical ML)
- **Multilayer Perceptron (MLP)** (Neural Network)
- **Convolutional Neural Network (CNN)** (Deep Learning)

The goal is to identify which approach performs best on medical imaging data and understand the trade-offs between model complexity, training time, and accuracy.

## Dataset

**PathMNIST** is a subset of the MedMNIST v2 collection containing:
- 28×28 RGB images of medical microscope images
- 9 classes (labels 0-8) representing different tissue types
- Training, validation, and test sets
- ~80,000+ total samples

### Class Distribution
The dataset shows slight class imbalance, with classes 5 and 8 having relatively more samples. Images display subtle color and texture differences between classes, making classification challenging.

## Models & Architecture

### 1. Decision Tree
- **Best Parameters:** `max_depth=10`, `criterion=entropy`
- **Validation Accuracy:** 0.4549
- **Training Time:** ~20 seconds
- **Advantages:** Fast, interpretable, no normalization needed
- **Disadvantages:** Prone to overfitting, struggles with complex patterns

### 2. Multilayer Perceptron (MLP)
- **Architecture:** Input (2352) → Dense(100) → Dense(200) → Output(9)
- **Best Parameters:** `learning_rate=0.01`, `activation=ReLU`, `units=[100, 200]`
- **Test Accuracy:** 0.6603
- **Training Time:** ~96 seconds
- **Advantages:** Simple architecture, good general performance
- **Disadvantages:** Loses spatial information, requires flattened input

### 3. Convolutional Neural Network (CNN)
- **Architecture:** Conv2D → MaxPool → Conv2D → MaxPool → Dropout → Dense(9)
- **Best Parameters:** 
  - First Conv: 64 filters, 3×3 kernel, ReLU
  - Second Conv: 128 filters, 3×3 kernel, ReLU
  - Dropout: 0.4, Learning rate: 0.001
- **Test Accuracy:** 0.8897
- **Training Time:** ~2264 seconds
- **Advantages:** Excellent feature extraction, highest accuracy, preserves spatial structure
- **Disadvantages:** Computationally expensive, requires more training time

## Requirements

```
python>=3.7
numpy
pandas
matplotlib
scikit-learn
tensorflow>=2.0
keras-tuner
medmnist
```

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras-tuner medmnist
```

## Usage

### Running the Complete Pipeline

```python
# Load the notebook
jupyter notebook A2-code-550077552-540561135.ipynb
```

### Key Steps in the Notebook

1. **Data Loading & Exploration**
   - Load PathMNIST dataset
   - Visualize class distribution
   - Examine pixel intensity distribution

2. **Data Preprocessing**
   - Normalize pixel values (0-255 → 0-1)
   - Flatten images for Decision Tree and MLP
   - Maintain 3D shape for CNN

3. **Model Training**
   - Hyperparameter tuning using GridSearchCV (Decision Tree) or Keras Tuner (MLP, CNN)
   - Train final models with optimal parameters
   - Evaluate on test set

4. **Evaluation**
   - Confusion matrices
   - Classification reports (precision, recall, F1-score)
   - Learning curves

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|---------|----------|---------------|
| Decision Tree | 0.46 | 0.46 | 0.45 | 0.45 | 20s |
| MLP | 0.66 | 0.63 | 0.62 | 0.63 | 96s |
| **CNN** | **0.89** | **0.88** | **0.87** | **0.88** | 2264s |

### Key Findings

- **CNN significantly outperforms** traditional methods with 89% accuracy
- **MLP shows moderate improvement** over Decision Tree (66% vs 46%)
- **Trade-off exists** between training time and accuracy
- **Spatial feature preservation** is crucial for medical image classification

### Per-Class Performance (CNN)

The CNN model demonstrates strong performance across all classes:
- Best performing classes: 0, 1, 3, 5 (precision ≥ 0.89)
- Most challenging class: 7 (recall = 0.78)
- Minimal misclassification between visually similar classes

## Hyperparameter Tuning

### Decision Tree
- Used GridSearchCV with 5-fold cross-validation
- Explored max_depth: [5, 10, 15, 20]
- Tested criteria: gini, entropy

### MLP
- Used Keras Tuner RandomSearch (24 trials, 15 epochs each)
- Hidden units: [100, 200]
- Activations: ReLU, sigmoid, tanh
- Learning rates: [0.001, 0.01, 0.1]

### CNN
- Used Keras Tuner RandomSearch (36 trials, 15 epochs each)
- Filters: [32, 64, 128] (Layer 1), [64, 128, 256] (Layer 2)
- Kernel sizes: [3, 5]
- Activations: ReLU, tanh
- Dropout rates: [0.3, 0.5]
- Learning rates: [0.001, 0.01, 0.1]

## Evaluation Metrics

All models are evaluated using:
- **Accuracy:** Overall classification performance
- **Precision:** Reliability of positive predictions per class
- **Recall:** Ability to detect all instances per class
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Visualization of classification patterns
- **Macro/Weighted Averages:** Account for class imbalance

## Conclusion

The **CNN model is the recommended choice** for PathMNIST classification due to:
1. Superior accuracy (89%) and balanced performance across all metrics
2. Effective hierarchical feature learning from medical images
3. Robust handling of subtle texture and color differences
4. Acceptable training time trade-off for production use

While Decision Trees offer speed and interpretability, and MLPs provide a middle ground, the CNN's ability to preserve and extract spatial features makes it ideal for medical imaging tasks.

## Future Work

- Train all 3,888 hyperparameter combinations for CNN with better hardware
- Implement data augmentation (rotation, flipping) to improve generalization
- Add cross-validation to detect overfitting
- Explore deeper architectures (ResNet, EfficientNet)
- Apply transfer learning from pre-trained models
- Visualize learned features using Grad-CAM
- Test generalization on other medical image classification datasets

## References

1. Yang, J., et al. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10, 41.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.

## Authors

Group 328 - COMP5318 Assignment 2
- Student A (550077552)
- Student B (540561135)

## License

This project is part of an academic assignment for educational purposes.
