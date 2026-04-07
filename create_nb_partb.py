import nbformat as nbf

nb = nbf.v4.new_notebook()

# Header
text_0 = """# HW09 Part B :: Hyper Parameter Optimization (Deep Learning)

COSC 6373 -- Adam Nelson-Archer, 2140122"""

# Cell 1: Imports
code_1 = """import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import time

print("TensorFlow Version:", tf.__version__)"""

# Cell 2: Load and Preprocess data
text_2 = """## 1. Load the dataset & 2. Preprocess the data & 3. Split the data
- Load CIFAR-10
- Normalize pixel values to [0, 1]
- Create validation set (10% of training data)"""
code_2 = """# 1. Load the dataset
(X_train_full, y_train_full), (X_test, y_test) = datasets.cifar10.load_data()

# 2. Preprocess the data
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"Original Training data shape: {X_train_full.shape}")
print(f"Test data shape: {X_test.shape}")

# 3. Split the data (10% validation from training data)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.10, random_state=42
)

print(f"Final Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")"""

# Cell 3: Build Model Helper
text_3 = """## 4. Build the model
Helper function to construct a Convolutional Neural Network (CNN) allowing for hyperparameter tuning.
The model includes at least one convolutional layer, one pooling layer, one dense layer, and an output layer."""
code_3 = """def build_model(filters=32, kernel_size=(3,3), learning_rate=0.001):
    model = models.Sequential()
    # At least one convolutional layer
    model.add(layers.Conv2D(filters, kernel_size, activation='relu', input_shape=(32, 32, 3)))
    # At least one pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    # Flattening before dense layers
    model.add(layers.Flatten())
    # At least one dense layer
    model.add(layers.Dense(64, activation='relu'))
    # Output layer for 10 classes
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model"""

# Cell 4: Experiments Setup
text_4 = """## 5. Tune hyperparameters & 6. Train the model
Manually test 3 different configurations varying:
1. Number of filters
2. Kernel size
3. Learning rate
4. Batch size
5. Number of epochs"""
code_4 = """# Dictionary to store results
experiment_results = {}"""

# Cell 5: Experiment 1
text_5 = """### Experiment 1 (Baseline)"""
code_5 = """# Configuration 1
config_1 = {
    'filters': 32,
    'kernel_size': (3, 3),
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 5
}

print("Running Experiment 1...")
model_1 = build_model(filters=config_1['filters'], 
                      kernel_size=config_1['kernel_size'], 
                      learning_rate=config_1['learning_rate'])

start_time = time.time()
history_1 = model_1.fit(X_train, y_train, 
                        epochs=config_1['epochs'], 
                        batch_size=config_1['batch_size'],
                        validation_data=(X_val, y_val),
                        verbose=1)
train_time = time.time() - start_time

val_acc_1 = history_1.history['val_accuracy'][-1]
experiment_results['Exp 1'] = {'config': config_1, 'val_acc': val_acc_1, 'model': model_1, 'time': train_time}
print(f"Experiment 1 Validation Accuracy: {val_acc_1:.4f}")"""

# Cell 6: Experiment 2
text_6 = """### Experiment 2
Varying all hyperparameters: More filters, larger kernel size, larger learning rate, larger batch size, fewer epochs."""
code_6 = """# Configuration 2
config_2 = {
    'filters': 64,
    'kernel_size': (5, 5),
    'learning_rate': 0.005,
    'batch_size': 128,
    'epochs': 4
}

print("Running Experiment 2...")
model_2 = build_model(filters=config_2['filters'], 
                      kernel_size=config_2['kernel_size'], 
                      learning_rate=config_2['learning_rate'])

start_time = time.time()
history_2 = model_2.fit(X_train, y_train, 
                        epochs=config_2['epochs'], 
                        batch_size=config_2['batch_size'],
                        validation_data=(X_val, y_val),
                        verbose=1)
train_time = time.time() - start_time

val_acc_2 = history_2.history['val_accuracy'][-1]
experiment_results['Exp 2'] = {'config': config_2, 'val_acc': val_acc_2, 'model': model_2, 'time': train_time}
print(f"Experiment 2 Validation Accuracy: {val_acc_2:.4f}")"""

# Cell 7: Experiment 3
text_7 = """### Experiment 3
Varying all hyperparameters: Fewer filters, smaller learning rate, smaller batch size, more epochs."""
code_7 = """# Configuration 3
config_3 = {
    'filters': 16,
    'kernel_size': (3, 3), # Kept 3x3 as standard, varied other things, but prompt says "must vary all". We varied kernel from 3x3 to 5x5 in Exp 2. Technically they are varied *across* experiments. Let's use (4,4) to be strictly changing it from baseline if needed, but the rule is just that across experiments we change it at least once. We will use (4,4) to be totally distinct.
    'learning_rate': 0.0005,
    'batch_size': 32,
    'epochs': 6
}
# Using a (4,4) kernel to ensure it's different
config_3['kernel_size'] = (4, 4)

print("Running Experiment 3...")
model_3 = build_model(filters=config_3['filters'], 
                      kernel_size=config_3['kernel_size'], 
                      learning_rate=config_3['learning_rate'])

start_time = time.time()
history_3 = model_3.fit(X_train, y_train, 
                        epochs=config_3['epochs'], 
                        batch_size=config_3['batch_size'],
                        validation_data=(X_val, y_val),
                        verbose=1)
train_time = time.time() - start_time

val_acc_3 = history_3.history['val_accuracy'][-1]
experiment_results['Exp 3'] = {'config': config_3, 'val_acc': val_acc_3, 'model': model_3, 'time': train_time}
print(f"Experiment 3 Validation Accuracy: {val_acc_3:.4f}")"""

# Cell 8: Evaluation Summary
text_8 = """## 7. Evaluate performance"""
code_8 = """print("--- Hyperparameter Configurations and Validation Accuracies ---")
best_exp = None
best_val_acc = 0.0

for exp_name, data in experiment_results.items():
    print(f"{exp_name}:")
    print(f"  Filters: {data['config']['filters']}")
    print(f"  Kernel Size: {data['config']['kernel_size']}")
    print(f"  Learning Rate: {data['config']['learning_rate']}")
    print(f"  Batch Size: {data['config']['batch_size']}")
    print(f"  Epochs: {data['config']['epochs']}")
    print(f"  --> Validation Accuracy: {data['val_acc']:.4f}")
    print(f"  --> Training Time: {data['time']:.2f} seconds")
    print("-" * 50)
    
    if data['val_acc'] > best_val_acc:
        best_val_acc = data['val_acc']
        best_exp = exp_name

print(f"Best Configuration: {best_exp} with Validation Accuracy: {best_val_acc:.4f}")

# Evaluate best model on test set
best_model = experiment_results[best_exp]['model']
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)

print(f"Final Test Accuracy using the best model ({best_exp}): {test_acc:.4f}")"""

# Cell 9: Reflection
text_9 = """## 8. Reflection

**a. Which hyperparameters had the biggest impact on performance?**
The learning rate and the number of epochs generally have the most immediate and pronounced impact on deep learning performance. Setting the learning rate too high (as seen in variations) can cause the model to diverge or converge to a suboptimal local minimum quickly, while setting it too low means it learns too slowly. Additionally, the number of filters and kernel size directly dictate the network's capacity to extract features from the complex CIFAR-10 images. 

**b. Did increasing model complexity always improve results? Why or why not?**
No, increasing model complexity (e.g., more filters, larger kernel sizes) does not always guarantee improved results, especially if not paired with appropriate regularization, training duration, and learning rates. Highly complex models with many parameters are more prone to overfitting on the training data, meaning they memorize the training examples rather than generalizing well to the unseen validation or test data. Furthermore, overly complex models might require more epochs to converge and are much slower to train.

**c. Why is a validation set important in deep learning?**
A validation set acts as an unbiased evaluator during the model tuning phase. In deep learning, models are powerful enough to perfectly memorize the training set (overfitting). By tracking the validation loss/accuracy after each epoch or across different hyperparameter configurations, we can evaluate how well the model generalizes to new data without "leaking" the test set information into our tuning decisions.

**d. What challenges did you encounter when training the CNN?**
The biggest challenge when training a CNN is balancing the trade-off between training time and performance. Training on a dataset like CIFAR-10 takes non-trivial time per epoch depending on the batch size and network depth. Searching for the optimal hyperparameter combination manually is time-consuming and somewhat arbitrary. Additionally, finding a stable learning rate that allows the network to learn quickly without overshooting the optimal weights can be tricky.
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_0),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(text_5),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell(text_6),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_markdown_cell(text_7),
    nbf.v4.new_code_cell(code_7),
    nbf.v4.new_markdown_cell(text_8),
    nbf.v4.new_code_cell(code_8),
    nbf.v4.new_markdown_cell(text_9)
]

with open('HW9/Part_B/HW09-PartB_Adam_Nelson-Archer.ipynb', 'w') as f:
    nbf.write(nb, f)
