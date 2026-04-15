# Neural Networks

## Definition

Neural Networks are supervised learning models composed of **interconnected layers of artificial neurons that learn complex nonlinear relationships in data**.

They are inspired by the **structure of the human brain**.

---

## Problem Type

- Classification
    
- Regression
    

---

## Core Idea

A neural network learns by **adjusting weights of connections between neurons** to minimize prediction error.

Multiple layers allow the model to learn **hierarchical representations of data**.

---

## Mathematical Formulation

Neuron output:

z = w · x + b

Activation:

a = σ(z)

Where:

- w = weights
    
- x = inputs
    
- b = bias
    
- σ = activation function
    

---

## Training Process

1. Perform [[Forward Propagation]]
    
2. Compute loss using [[Loss Functions]]
    
3. Apply [[Backpropagation]]
    
4. Update weights using [[Gradient Descent]]
    

---

## Important Hyperparameters

learning_rate  
number_of_layers  
number_of_neurons  
batch_size  
epochs

---

## Advantages

- models complex nonlinear patterns
    
- highly flexible
    
- state-of-the-art performance in many domains
    

---

## Limitations

- requires large datasets
    
- computationally expensive
    
- difficult to interpret
    

---

## Applications

- computer vision
    
- natural language processing
    
- speech recognition
    
- recommendation systems
    

---

## Related Concepts

[[Activation Functions]]  
[[Backpropagation]]  
[[Gradient Descent]]  
[[Deep Learning]]