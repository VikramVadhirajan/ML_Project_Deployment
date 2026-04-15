# Recurrent Neural Networks (RNN)

## Definition

Recurrent Neural Networks are neural networks designed to **process sequential data** by maintaining a memory of previous inputs.

They are commonly used for **time-series and natural language processing tasks**.

---

## Problem Type

- Classification
    
- Regression
    
- Sequence prediction
    

---

## Core Idea

RNNs process sequences step-by-step while **passing hidden states between time steps**, allowing the model to remember past information.

---

## Mathematical Formulation

Hidden state:

hₜ = σ(Wxₜ + Uhₜ₋₁ + b)

Where:

- xₜ = input at time t
    
- hₜ = hidden state
    
- W, U = weight matrices
    

---

## Training Process

1. Forward pass through sequence
    
2. Compute loss
    
3. Apply [[Backpropagation Through Time]]
    
4. Update weights
    

---

## Important Hyperparameters

hidden_size  
sequence_length  
learning_rate

---

## Advantages

- captures temporal dependencies
    
- suitable for sequential data
    

---

## Limitations

- vanishing gradient problem
    
- difficult to train long sequences
    

---

## Applications

- language modeling
    
- speech recognition
    
- time-series forecasting
    

---

## Related Concepts

[[Neural Networks]]  
[[LSTM]]  
[[GRU]]  
[[Deep Learning]]