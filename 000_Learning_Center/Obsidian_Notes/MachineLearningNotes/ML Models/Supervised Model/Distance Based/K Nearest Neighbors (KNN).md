# K Nearest Neighbors (KNN)

## Definition

K Nearest Neighbors is a supervised learning algorithm that predicts labels based on the **k closest training samples in feature space**.

It is a **lazy learning algorithm**, meaning it does not learn a model during training.

---

## Problem Type

- Classification
    
- Regression
    

---

## Core Idea

The prediction for a new point is determined by the **majority label of its nearest neighbors**.

Distance metrics determine which points are closest.

---

## Mathematical Formulation

Common distance metric:

Euclidean Distance

d(x,y) = √Σ(xᵢ − yᵢ)²

---

## Training Process

1. Store the entire training dataset
    
2. Compute distance between query point and training samples
    
3. Select k nearest neighbors
    
4. Predict based on neighbor labels
    

---

## Important Hyperparameters

k → number of neighbors  
distance metric → Euclidean, Manhattan  
weights → uniform or distance-based

---

## Advantages

- simple to implement
    
- no training phase
    
- works well for small datasets
    

---

## Limitations

- slow prediction for large datasets
    
- sensitive to [[Feature Scaling]]
    
- affected by irrelevant features
    

---

## Applications

- recommendation systems
    
- pattern recognition
    
- anomaly detection
    

---

## Related Concepts

[[Feature Scaling]]  
[[Distance Metrics]]  
[[Model Evaluation]]