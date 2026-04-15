# Bayesian Optimization

Bayesian Optimization is an advanced hyperparameter tuning technique that **builds a probabilistic model of the objective function and selects promising hyperparameters to evaluate**.

---

# Core Idea

Instead of randomly testing configurations, Bayesian optimization **learns which hyperparameters perform well and focuses search there**.

---

# Workflow

1. Train model with initial hyperparameters
    
2. Build probabilistic model of performance
    
3. Select next promising parameters
    
4. Evaluate model
    
5. Update model and repeat
    

---

# Advantages

- efficient search
    
- requires fewer evaluations
    
- useful for expensive models
    

---

# Popular Tools

- Optuna
    
- Hyperopt
    
- Scikit-Optimize
    

---

# Limitations

- more complex implementation
    
- slower setup than grid/random search
    

---

# Related Concepts

[[Model Tuning]]  
[[Random Search]]  
[[Hyperparameters]]