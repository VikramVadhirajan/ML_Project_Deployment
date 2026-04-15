# R Squared (Coefficient of Determination)

R² measures how well a regression model **explains the variance of the target variable**.

It indicates the proportion of variance in the dependent variable that is explained by the model.

---

# Formula

R² = 1 − (SS_res / SS_tot)

Where:

SS_res = Σ (y − ŷ)²  
SS_tot = Σ (y − ȳ)²

---

# Interpretation

R² = 1 → perfect prediction  
R² = 0 → model predicts the mean  
R² < 0 → model worse than baseline

Example:

R² = 0.85 means **85% of variance is explained by the model**.

---

# Limitations

R² always increases when adding features, even if they are irrelevant.

Solution:

Use [[Adjusted R Squared]].

---

# Related Concepts

[[Model Evaluation]]  
[[Regression]]  
[[Mean Squared Error]]