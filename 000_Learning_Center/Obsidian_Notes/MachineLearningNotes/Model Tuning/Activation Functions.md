# Activation Functions

Activation functions are mathematical functions applied to the output of a neuron in a [[Neural Networks]] model.

They introduce **nonlinearity** into the network, allowing neural networks to learn complex patterns.

Without activation functions, a neural network would behave like a **linear model**, regardless of how many layers it contains.

---

# Why Activation Functions are Needed

Neural networks compute:

z = w·x + b

If we stack multiple layers without activation functions, the entire network reduces to a **single linear transformation**.

Activation functions allow neural networks to model **nonlinear relationships**.

---

# Common Activation Functions

## Sigmoid

Formula:

σ(x) = 1 / (1 + e⁻ˣ)

Range:

0 to 1

Used for:

- binary classification output layers
    

Limitations:

- vanishing gradient problem
    

See: [[Logistic Regression]]

---

## Tanh

Formula:

tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)

Range:

-1 to 1

Advantages:

- zero-centered output
    

Limitations:

- still suffers from vanishing gradients
    

---

## ReLU (Rectified Linear Unit)

Formula:

f(x) = max(0, x)

Advantages:

- computationally efficient
    
- reduces vanishing gradient issues
    

Widely used in **deep neural networks**.

---

## Softmax

Softmax converts outputs into **probability distributions**.

Formula:

Softmax(xᵢ) = eˣⁱ / Σ eˣʲ

Used for:

- multiclass classification output layers
    

---

# Choosing Activation Functions

Hidden layers:

- ReLU (most common)
    

Output layer:

Binary classification → Sigmoid  
Multiclass classification → Softmax  
Regression → Linear activation

---

# Related Concepts

[[Neural Networks]]  
[[Backpropagation]]  
[[Cross Entropy Loss]]  
[[Deep Learning]]