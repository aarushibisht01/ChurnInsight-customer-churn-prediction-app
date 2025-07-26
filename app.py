import streamlit as st
import pandas as pd
import joblib

model = joblib.load('churn_model.pkl')
df = pd.read_csv('data/clean_real_world_dataset.csv')

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 ChurnInsight")
st.title("Customer Churn Prediction App")

feature_columns = df.drop("Churn", axis=1).columns.tolist()

st.sidebar.header("Input Customer Details")

user_input = {}
for col in feature_columns:
    col_data = df[col]
    if col_data.nunique() <= 10:
        options = sorted(col_data.unique())
        user_input[col] = st.sidebar.selectbox(f"{col}", options)
    else:
        min_val = float(col_data.min())
        max_val = float(col_data.max())
        mean_val = float(col_data.mean())
        user_input[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)

input_df = pd.DataFrame([user_input])

st.subheader("User Input:")
st.write(input_df)

prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result:")
if prediction == 1:
    st.error(f"This customer is likely to churn with probability: {prediction_proba:.2f}")
else:
    st.success(f"This customer is likely to stay with probability: {1 - prediction_proba:.2f}")
