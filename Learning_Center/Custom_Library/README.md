# ml_utils

`ml_utils` is a custom Python module designed to simplify and standardize common **machine learning preprocessing tasks**.
It provides reusable utilities for data cleaning, feature engineering, outlier handling, and feature scaling so that preprocessing logic can be reused across multiple ML projects.

The main goal of this module is to:

* Reduce repetitive preprocessing code
* Maintain consistency between **training and deployment**
* Make ML workflows cleaner and modular

---

# Project Structure

```
ml_utils/
│
├── __init__.py
├── data_cleaning.py
├── feature_engineering.py
├── outliers.py
└── scalar.py
```

---

# Module Description

## data_cleaning.py

This module contains utilities for cleaning datasets before model training.

Typical functionalities include:

* Identifying numeric and categorical columns
* Detecting duplicate rows
* Removing duplicate records
* Dropping unnecessary columns
* Basic data inspection

Example usage:

```python
from ml_utils.data_cleaning import get_column_types

numeric_cols, categorical_cols = get_column_types(df)
```

---

## feature_engineering.py

This module contains functions for transforming and encoding features.

Typical functionalities include:

* Label encoding for categorical variables
* Encoding multiple categorical columns
* Saving encoder objects for reuse in deployment

Example usage:

```python
from ml_utils.feature_engineering import label_encode_multiple

df = label_encode_multiple(df, categorical_cols)
```

---

## outliers.py

This module contains utilities for detecting and handling outliers.

Typical functionalities include:

* Outlier detection using the IQR method
* Replacing outliers using clipping
* Saving outlier bounds to pickle files for reuse

Example usage:

```python
from ml_utils.outliers import replace_outliers_multi

df = replace_outliers_multi(df, numeric_cols)
```

---

## scalar.py

This module provides feature scaling utilities.

Typical functionalities include:

* Standard scaling
* Min-max scaling
* Saving scaler objects for deployment

Example usage:

```python
from ml_utils.scalar import apply_scaler

df = apply_scaler(df, numeric_cols)
```

---

# Example ML Preprocessing Workflow

A typical workflow using this module may look like the following:

```python
from ml_utils.data_cleaning import get_column_types
from ml_utils.outliers import replace_outliers_multi
from ml_utils.feature_engineering import label_encode_multiple

# Detect column types
numeric_cols, categorical_cols = get_column_types(df)

# Handle outliers
df = replace_outliers_multi(df, numeric_cols)

# Encode categorical features
df = label_encode_multiple(df, categorical_cols)
```

---

# Deployment Support

Some preprocessing functions save artifacts using **pickle** so that the same preprocessing logic can be reused during model inference.

Example artifacts generated during training:

```
encoders.pkl
outlier_bounds.pkl
scaler.pkl
imputers.pkl
```

These artifacts ensure that **training and deployment pipelines use the same preprocessing logic**.

---

# Requirements

```
pandas
numpy
scikit-learn
pickle
```

---

# Purpose of This Module

This module was created to:

* Organize preprocessing code in ML projects
* Improve code reusability
* Simplify data preparation workflows
* Ensure consistent preprocessing between training and deployment

---

# Future Improvements

Possible future enhancements include:

* Automated preprocessing pipelines
* Feature selection utilities
* Data validation tools
* Integration with ML pipelines

---

# Author

Vikram Vadhirajan
