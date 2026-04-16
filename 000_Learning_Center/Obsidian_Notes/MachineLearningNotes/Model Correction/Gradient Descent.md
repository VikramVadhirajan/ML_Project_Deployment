# Gradient Descent

Gradient Descent is an optimization algorithm used to **minimize a loss function** by iteratively updating model parameters.

It moves parameters in the direction of the **negative gradient** of the loss.

---

## Update Rule

θ = θ − α ∇J(θ)

Where:

- θ = model parameters
    
- α = learning rate
    
- J(θ) = cost function
    
- ∇J(θ) = gradient of the cost function
    

---

## Types of Gradient Descent

### Batch Gradient Descent

Uses the entire dataset to compute gradients.

### Stochastic Gradient Descent

Uses one data sample per update.

### Mini Batch Gradient Descent

Uses small batches of data.

This is the most commonly used approach in modern ML.

---

## Learning Rate

The learning rate controls how large the update step is.

Too large → divergence  
Too small → slow convergence