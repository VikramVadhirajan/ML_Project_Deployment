# Accuracy

Accuracy measures the **proportion of correct predictions made by a classification model**.

It is one of the simplest metrics for evaluating [[Classification]] models.

---

# Formula

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Where:

TP = True Positives  
TN = True Negatives  
FP = False Positives  
FN = False Negatives

---

# Interpretation

Accuracy = 0.90 means **90% of predictions are correct**.

---

# Limitations

Accuracy can be misleading for **imbalanced datasets**.

Example:

Dataset with 95% negative class → model predicting always negative gives **95% accuracy**.

---

# Related Concepts

[[Model Evaluation]]  
[[Precision]]  
[[Recall]]  
[[Confusion Matrix]]