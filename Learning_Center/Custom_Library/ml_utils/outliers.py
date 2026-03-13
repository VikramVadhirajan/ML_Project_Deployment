import pandas as pd
import pickle
import os


import pandas as pd
import pickle
import os


def detect_outliers(df, columns, file_name="outlier_bounds.pkl"):
    """
    Detect outliers using IQR method for multiple columns
    and store bounds in a single pickle file.
    """

    bounds = {}

    # # Load existing pickle if present
    # if os.path.exists(file_name):
    #     with open(file_name, "rb") as f:
    #         bounds = pickle.load(f)

    outlier_dict = {}

    for col in columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Store bounds
        bounds[col] = {
            "lower": lower,
            "upper": upper
        }

        # Detect outliers
        outliers = df[(df[col] < lower) | (df[col] > upper)]

        outlier_dict[col] = outliers

    # Save bounds to pickle
    with open(file_name, "wb") as f:
        pickle.dump(bounds, f)

    return outlier_dict





def replace_outliers(df,columns,file_name="outlier_bounds.pkl"):
    """
    Replace outliers using IQR method and store bounds
    for multiple datasets into the same pickle file.
    """

    bounds = {}

    # # Load existing pickle if available
    # if os.path.exists(file_name):
    #     with open(file_name, "rb") as f:
    #         bounds = pickle.load(f)

    for col in columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Replace outliers
        df[col] = df[col].clip(lower, upper)

        # Store bounds
        bounds[col] = {
            "lower": lower,
            "upper": upper
        }

    # Save updated pickle
    with open(file_name, "wb") as f:
        pickle.dump(bounds, f)

    return df
