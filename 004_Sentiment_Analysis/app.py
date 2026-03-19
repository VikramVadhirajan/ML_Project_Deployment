import streamlit as st
from transformers import pipeline
import pandas as pd
from tqdm import tqdm

sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

st.title("Sentiment Analysis of Comments -General Purpose")

# File uploader widget
uploaded_file = st.file_uploader("Choose a file *Ensure to have a column named 'Comments'*", type=["csv", "xlsx", "json"])

# 🚨 STOP EXECUTION if no file
if uploaded_file is None:
    st.info("Please upload a file to proceed.")
    st.stop()

if uploaded_file is not None:
    # Read the file based on its type
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    st.write("File uploaded successfully!")


if uploaded_file is not None and "processed_df" not in st.session_state:
    
    df = df.dropna(subset=['Comments']).reset_index(drop=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_rows = len(df)

    for i in range(total_rows):
        text = df["Comments"][i]
        df.loc[i, 'Length of text'] = len(text)

        if len(text) < 130:
            result = sentiment_pipeline(text)[0]
            df.loc[i, 'Sentiment'] = result['label']
            df.loc[i, 'Confidence'] = round(result['score'], 2)
        else:
            chunks = [text[j:j+130] for j in range(0, len(text), 130)]
            sentiments, scores = [], []

            for chunk in chunks:
                result = sentiment_pipeline(chunk)[0]
                sentiments.append(result['label'])
                scores.append(result['score'])

            df.loc[i, 'Sentiment'] = max(set(sentiments), key=sentiments.count)
            df.loc[i, 'Confidence'] = round(sum(scores) / len(scores), 2)

        progress_bar.progress((i + 1) / total_rows)
        status_text.text(f"Processing row {i+1} of {total_rows}")

    # ✅ SAVE result
    st.session_state.processed_df = df


# --- Export Data ---

if "processed_df" in st.session_state:
    df = st.session_state.processed_df

    st.success("Analysis Completed ✅")
    # Convert dataframe to CSV for downloa
    csv = df.to_csv(index=False).encode("utf-8")
    # Download button widget
    st.download_button(
        label="Download data with sentiment analysis the result CSV now has three extra column 'Length of text', 'Sentiment', 'Confidence'",
        data=csv,
        file_name="data_With_Sentiment.csv",
        mime="text/csv",
    )