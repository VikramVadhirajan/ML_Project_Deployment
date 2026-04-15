# Random Search

Random Search is a hyperparameter optimization technique that **samples random combinations of hyperparameters from a defined search space**.

---

# Core Idea

Instead of testing all combinations like [[Grid Search]], random search selects a **random subset of configurations**.

---

# Advantages

- faster than grid search
    
- explores larger search space
    
- efficient for high-dimensional tuning
    

---

# Limitations

- may miss optimal configuration
    

---

# Example (Python)

```python
from sklearn.model_selection import RandomizedSearchCV
```

---

# Related Concepts

[[Model Tuning]]  
[[Grid Search]]  
[[Hyperparameters]]