# Backpropagation

Backpropagation is the algorithm used to **train neural networks by computing gradients of the loss function with respect to network parameters**.

It efficiently applies the **chain rule of calculus** to propagate errors backward through the network.

---

# Training Process

Training a neural network involves two main steps:

1. [[Forward Propagation]]
    
2. Backpropagation
    

Forward propagation computes predictions, while backpropagation computes gradients for learning.

---

# Steps of Backpropagation

1. Perform forward propagation
    
2. Compute loss using a [[Loss Functions]]
    
3. Calculate gradient of loss with respect to output
    
4. Propagate gradients backward through layers
    
5. Update weights using [[Gradient Descent]]
    

---

# Chain Rule

Backpropagation relies on the **chain rule from calculus**.

If:

L = loss  
z = weighted sum  
a = activation

Then:

∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w

This allows gradients to flow backward through the network.

---

# Gradient Flow

Backpropagation calculates gradients for:

- weights
    
- biases
    

These gradients indicate **how much each parameter contributes to prediction error**.

---

# Importance

Backpropagation makes training deep neural networks computationally feasible.

Without it, training large neural networks would be extremely inefficient.

---

# Related Concepts

[[Neural Networks]]  
[[Forward Propagation]]  
[[Gradient Descent]]  
[[Activation Functions]]  
[[Loss Functions]]