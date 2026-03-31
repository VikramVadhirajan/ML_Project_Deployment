import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt 


def detect_outliers(df, columns):
    """
    Detect outliers using IQR method for multiple columns
    and store bounds in a single pickle file.
    """


    result = []
    plotcol=[]
    for col in columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Detect outliers
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()

        result.append({
            "column": col,
            "outlier_count": outlier_count
        })
        if outlier_count>=1:
            plotcol.append(col)
    if len(plotcol)>0:
        df[plotcol].boxplot(figsize=(20,7),color='r')
    plt.xticks(rotation=90)
    plt.title('With Outliers',fontsize=10)
    plt.show()
    result = pd.DataFrame(result).sort_values(by="outlier_count", ascending=False)
    return result





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
