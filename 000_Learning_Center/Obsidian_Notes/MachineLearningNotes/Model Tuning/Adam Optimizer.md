# Adam Optimizer

Adam (Adaptive Moment Estimation) is an optimization algorithm used to **train neural networks efficiently**.

It combines ideas from:

- [[Momentum]]
    
- [[RMSProp]]
    

Adam adapts learning rates for each parameter during training.

---

# Motivation

Standard [[Gradient Descent]] uses a fixed learning rate.

Problems:

- slow convergence
    
- oscillations in optimization
    

Adam solves this by **adapting learning rates based on past gradients**.

---

# Key Components

Adam maintains two estimates:

First moment (mean of gradients)

mₜ = β₁ mₜ₋₁ + (1 − β₁) gₜ

Second moment (variance of gradients)

vₜ = β₂ vₜ₋₁ + (1 − β₂) gₜ²

Where:

- gₜ = gradient at time t
    
- β₁, β₂ = decay rates
    

---

# Parameter Update

Weights are updated as:

θ = θ − α m̂ / (√v̂ + ε)

Where:

- α = learning rate
    
- ε = small constant for stability
    

---

# Advantages

- fast convergence
    
- adaptive learning rates
    
- works well for large neural networks
    
- robust to noisy gradients
    

---

# Default Hyperparameters

Common default values:

learning rate = 0.001  
β₁ = 0.9  
β₂ = 0.999  
ε = 10⁻⁸

---

# Related Concepts

[[Gradient Descent]]  
[[Momentum]]  
[[RMSProp]]  
[[Neural Networks]]  
[[Backpropagation]]