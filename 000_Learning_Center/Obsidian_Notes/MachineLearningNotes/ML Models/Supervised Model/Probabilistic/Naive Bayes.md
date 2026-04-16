
![[Pasted image 20260416190314.png]]

# Naive Bayes

## Definition

Naive Bayes is a probabilistic supervised learning algorithm based on **Bayes' theorem** with the assumption that features are independent.

It is widely used in **text classification problems**.

---

## Problem Type

- Classification
    

---

## Core Idea

The algorithm computes the probability of each class given the input features and chooses the class with the **highest posterior probability**.

---

## Mathematical Formulation

Bayes Theorem:

P(C|X) = (P(X|C) P(C)) / P(X)

Where:

C = class  
X = features

---

## Training Process

1. Estimate prior probabilities of classes
    
2. Estimate likelihood of features given class
    
3. Compute posterior probabilities
    
4. Choose class with highest probability
    

---

## Important Hyperparameters

var_smoothing → numerical stability parameter

---

## Advantages

- fast training
    
- works well for high-dimensional data
    
- performs well with small datasets
    

---

## Limitations

- independence assumption rarely holds
    
- less accurate for complex relationships
    

---

## Applications

- spam detection
    
- sentiment analysis
    
- document classification
    

---

## Related Concepts

[[Probability]]  
[[Bayes Theorem]]  
[[Classification]]