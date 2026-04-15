# ROC Curve

The ROC Curve (Receiver Operating Characteristic Curve) is a graphical representation of the performance of a classification model at different thresholds.

It plots:

True Positive Rate vs False Positive Rate.

---

# Definitions

True Positive Rate (TPR):

TPR = TP / (TP + FN)

False Positive Rate (FPR):

FPR = FP / (FP + TN)

---

# Interpretation

A good classifier produces a curve closer to the **top-left corner** of the plot.

Random guessing produces a **diagonal line**.

---

# Usage

ROC curves are used to evaluate models across **different classification thresholds**.

---

# Related Concepts

[[Model Evaluation]]  
[[AUC]]  
[[Confusion Matrix]]