# Single Hidden Layer Neural Network From Scratch

This project implements a **single-hidden-layer neural network** completely **from scratch** using only **NumPy**.

The main purpose of this project is to understand how a neural network works at the most fundamental level without using high-level machine learning or deep learning libraries such as TensorFlow, PyTorch, or scikit-learn.

## Project Purpose

This project was developed as a learning-focused implementation.  
The goal is not to build a highly optimized production model, but to clearly understand the internal logic of a neural network step by step.

Instead of relying on ready-made library functions, the core parts of the model were implemented manually, including:

- data generation
- forward propagation
- activation functions
- loss calculation
- backpropagation
- parameter updates with gradient descent

In this way, the connection between mathematical formulas and code becomes much more visible.

## What the Project Does

The model is trained on a simple **2D binary classification dataset** generated with NumPy.

Two classes are created as Gaussian clusters:

- class 0 is centered around `(-2, 0)`
- class 1 is centered around `(2, 0)`

The neural network learns to separate these two classes by adjusting its weights and biases during training.

## Model Architecture

This project uses a **single hidden layer MLP (Multi-Layer Perceptron)**.

- **Input layer:** 2 features
- **Hidden layer:** 8 neurons
- **Hidden activation:** ReLU
- **Output layer:** 1 neuron
- **Output activation:** Sigmoid

Loss Function:
loss = -np.mean(y*np.log(yhat) + (1-y)*np.log(1-yhat))

Backpropagation:
dz2 = (yhat - y) / N
dW2 = a1.T @ dz2
db2 = np.sum(dz2, axis=0, keepdims=True)
da1 = dz2 @ W2.T
dz1 = da1 * (z1 > 0)
dW1 = X.T @ dz1
db1 = np.sum(dz1, axis=0, keepdims=True)

Optimization:
W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2


The forward propagation equations are:

```python
z1 = X @ W1 + b1
a1 = np.maximum(0.0, z1)
z2 = a1 @ W2 + b2
yhat = sigmoid(z2)


