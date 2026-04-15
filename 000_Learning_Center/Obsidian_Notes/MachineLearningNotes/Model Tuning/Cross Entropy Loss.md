# Cross Entropy Loss

Cross Entropy Loss is a loss function commonly used for **classification problems in machine learning and deep learning**.

It measures the difference between:

- predicted probability distribution
    
- true probability distribution
    

---

# Binary Cross Entropy

Used in **binary classification**.

Formula:

L = − [y log(p) + (1 − y) log(1 − p)]

Where:

- y = true label
    
- p = predicted probability
    

---

# Multiclass Cross Entropy

Used with [[Softmax Function]].

Formula:

L = − Σ yᵢ log(pᵢ)

Where:

- yᵢ = true label probability
    
- pᵢ = predicted probability
    

---

# Intuition

Cross entropy penalizes predictions that assign **low probability to the correct class**.

Correct predictions produce **low loss**, while incorrect confident predictions produce **high loss**.

---

# Why It Works Well

Cross entropy works well because:

- it aligns with **maximum likelihood estimation**
    
- provides smooth gradients for optimization
    

---

# Usage

Binary classification:

Sigmoid + Binary Cross Entropy

Multiclass classification:

Softmax + Cross Entropy

---

# Related Concepts

[[Neural Networks]]  
[[Softmax Function]]  
[[Activation Functions]]  
[[Backpropagation]]  
[[Loss Functions]]