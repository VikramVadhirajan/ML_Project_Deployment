# Support Vector Regression (SVR)

## Definition

Support Vector Regression is the regression counterpart of [[Support Vector Machines]] that predicts continuous values while maintaining a **margin of tolerance around predictions**.

---

## Problem Type

- Regression
    

---

## Core Idea

Instead of minimizing prediction error directly, SVR tries to **fit a function within an epsilon margin around the true values**.

Errors outside this margin are penalized.

---

## Mathematical Formulation

Objective:

Minimize

½ ||w||² + C Σ (ξᵢ + ξᵢ*)

Where:

ξ represents slack variables.

---

## Training Process

1. Map features into higher-dimensional space
    
2. Define epsilon margin
    
3. Optimize parameters to minimize error outside margin
    

---

## Important Hyperparameters

C → regularization parameter  
epsilon → width of margin  
kernel → linear, polynomial, RBF

---

## Advantages

- handles nonlinear regression
    
- robust to outliers within epsilon margin
    

---

## Limitations

- computationally expensive
    
- sensitive to parameter selection
    

---

## Applications

- stock price prediction
    
- energy load forecasting
    

---

## Related Concepts

[[Support Vector Machines]]  
[[Regression]]  
[[Kernel Trick]]