# Loss Functions

A loss function measures the **difference between predicted values and actual values**.

The training process attempts to **minimize this loss**.

---

## Mean Squared Error (MSE)

Most common loss for regression.

MSE = (1/n) Σ (y − ŷ)²

Properties:

- Penalizes large errors
    
- Smooth and differentiable
    
- Works well with gradient descent
    

---

## Mean Absolute Error (MAE)

MAE = (1/n) Σ |y − ŷ|

Advantages:

- Less sensitive to outliers
    
- Easier interpretation
    

---

Loss functions define the **optimization objective for machine learning models**.