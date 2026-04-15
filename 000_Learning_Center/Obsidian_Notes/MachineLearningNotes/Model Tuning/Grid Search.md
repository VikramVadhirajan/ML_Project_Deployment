# Grid Search

Grid Search is a hyperparameter tuning method that **tests every combination of hyperparameters in a predefined search grid**.

---

# Core Idea

Define a grid of possible hyperparameter values.

Example:

learning_rate = [0.01, 0.1]  
max_depth = [3,5,7]

Grid search evaluates all combinations.

---

# Process

1. Define hyperparameter grid
    
2. Train model for each combination
    
3. Evaluate using [[Cross Validation]]
    
4. Select best parameters
    

---

# Advantages

- exhaustive search
    
- guaranteed best combination within grid
    

---

# Limitations

- computationally expensive
    
- inefficient for large parameter spaces
    

---

# Example (Python)

```python
from sklearn.model_selection import GridSearchCV
```

---

# Related Concepts

[[Model Tuning]]  
[[Random Search]]  
[[Hyperparameters]]