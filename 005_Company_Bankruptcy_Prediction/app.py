import streamlit as st
import pandas as pd
import joblib



pipeline = joblib.load('model_pipeline.pkl')

outlier_bounds = joblib.load('outlier_bounds.pkl')


# Strip leading/trailing spaces from both pipeline feature names and uploaded columns
feature_names = [f.strip() for f in pipeline.feature_names_in_]
# OR hardcode: feature_names = ['col1', 'col2', ...]


st.title("ML Classifier")
st.subheader("Upload Excel file (single row)")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    # And after reading the Excel file:
    df.columns = df.columns.str.strip()
    if df.shape[0] != 1:
        st.error(f"Expected exactly 1 row, got {df.shape[0]}.")
    else:
        missing = [c for c in feature_names if c not in df.columns]
        extra = [c for c in df.columns if c not in feature_names]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if extra:
                st.warning(f"Ignoring unrecognised columns: {extra}")

            input_df = df[feature_names]  # reorder to match training order
            input_df = df[feature_names].apply(pd.to_numeric, errors='coerce')
            st.write("Input data (reordered):", input_df)
            # Apply same outlier clipping as training
            for col in input_df.columns:
                if col in outlier_bounds:
                    input_df[col] = input_df[col].clip(
                        lower=outlier_bounds[col]['lower'],
                        upper=outlier_bounds[col]['upper']
                    )

                 
            # st.write("Columns fed to model:", input_df.columns.tolist())
            # st.write("Dtypes:", input_df.dtypes)
            # st.write("Values:", input_df.values)
            # st.write("Pipeline classes:", pipeline.classes_)
            # st.write("Feature names sample:", feature_names[:3])
            # st.write("Excel columns sample:", df.columns.tolist()[:3])
            st.write("NaN count:", input_df.isnull().sum().sum())

            st.write("Pipeline steps:", pipeline.steps)
            
            # st.write("Pipeline classes:", pipeline.classes_)

            pred = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0]

            probability_of_bankruptcy = float(proba[1])

            if pred == 1:
                st.error("⚠️ Company Likely to go Bankrupt")
            else:
                st.success("✅ Company Financially Stable")

            # st.metric("Bankruptcy Probability", f"{round(probability_of_bankruptcy * 100, 2)}%")
            # st.progress(probability_of_bankruptcy)
