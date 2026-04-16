
![[Pasted image 20260416182823.png]]

# Support Vector Machines (SVM)

## Definition

Support Vector Machines are supervised learning algorithms that classify data by finding the **optimal hyperplane that maximizes the margin between classes**.

They are effective in **high-dimensional feature spaces** and can model both linear and nonlinear relationships.

---

## Problem Type

- Classification ([[Support Vector Classification]])
    
- Regression ([[Support Vector Regression]])
    

---

## Core Idea

SVM finds a decision boundary that **maximizes the margin between the closest data points of different classes**.

The points closest to the boundary are called **support vectors**.

---

## Mathematical Formulation

Hyperplane:

w · x + b = 0

Optimization objective:

Minimize

½ ||w||²

Subject to:

yᵢ(w · xᵢ + b) ≥ 1

Where:

- w = weight vector
    
- b = bias
    
- yᵢ = class label
    

---

## Training Process

1. Identify support vectors
    
2. Find hyperplane maximizing margin
    
3. Use kernel functions for nonlinear separation
    
4. Optimize objective function
    

---

## Important Hyperparameters

C → regularization parameter  
kernel → linear, polynomial, RBF  
gamma → kernel coefficient

---

## Advantages

- effective in high-dimensional spaces
    
- robust to overfitting
    
- flexible with kernel trick
    

---

## Limitations

- computationally expensive for large datasets
    
- difficult to tune hyperparameters
    

---

## Applications

- text classification
    
- image recognition
    
- bioinformatics
    

---

## Python Documentation 

https://scikit-learn.org/stable/modules/svm.html#

---


## Related Concepts

[[Kernel Trick]]  
[[Support Vector Regression]]  
[[Classification]]  
[[Model Evaluation]]