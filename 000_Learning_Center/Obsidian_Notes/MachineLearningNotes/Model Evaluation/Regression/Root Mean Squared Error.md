# Root Mean Squared Error (RMSE)

Root Mean Squared Error is the **square root of Mean Squared Error**.

It measures the average magnitude of prediction errors in the **same units as the target variable**.

---

# Formula

RMSE = √( (1/n) Σ (yᵢ − ŷᵢ)² )

---

# Intuition

Taking the square root makes RMSE easier to interpret because the result has the **same scale as the original data**.

Example:

If RMSE = 10 for house prices (in thousands), predictions are off by about **$10,000 on average**.

---

# Properties

- always non-negative
    
- sensitive to outliers
    
- interpretable in original units
    

---

# Related Concepts

[[Model Evaluation]]  
[[Mean Squared Error]]  
[[Mean Absolute Error]]  
[[Regression]]