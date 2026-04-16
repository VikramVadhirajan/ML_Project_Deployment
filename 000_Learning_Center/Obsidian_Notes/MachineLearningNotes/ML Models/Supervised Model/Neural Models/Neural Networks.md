
![[Pasted image 20260416190152.png]]

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

[[Activation Functions]]:

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

(_hidden_layer_sizes=(100,)_, _activation='relu'_, _*_, _solver='adam'_, _alpha=0.0001_, _batch_size='auto'_, _learning_rate='constant'_, _learning_rate_init=0.001_, _power_t=0.5_, _max_iter=200_, _shuffle=True_, _random_state=None_, _tol=0.0001_, _verbose=False_, _warm_start=False_, _momentum=0.9_, _nesterovs_momentum=True_, _early_stopping=False_, _validation_fraction=0.1_, _beta_1=0.9_, _beta_2=0.999_, _epsilon=1e-08_, _n_iter_no_change=10_, _max_fun=15000_)


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
## Python Documentation 

https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

---
## Related Concepts

[[Activation Functions]]  
[[Backpropagation]]  
[[Gradient Descent]]  
[[Deep Learning]]