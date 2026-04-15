# Mean Squared Error (MSE)

Mean Squared Error is a regression evaluation metric that measures the **average squared difference between predicted values and actual values**.

It is one of the most commonly used metrics for [[Regression]] problems.

---

# Formula

MSE = (1/n) Σ (yᵢ − ŷᵢ)²

Where:

- n = number of samples
    
- yᵢ = actual value
    
- ŷᵢ = predicted value
    

---

# Intuition

MSE squares the prediction errors, which means:

- large errors are penalized more heavily
    
- small errors have less impact
    

This encourages models to avoid large mistakes.

---

# Properties

- always non-negative
    
- lower values indicate better performance
    
- sensitive to outliers
    

---

# Usage

Commonly used in:

- [[Linear Regression]]
    
- [[Neural Networks]]
    
- many optimization problems
    

---

# Related Concepts

[[Model Evaluation]]  
[[Root Mean Squared Error]]  
[[Mean Absolute Error]]  
[[Regression]]