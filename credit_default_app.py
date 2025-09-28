import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


lgbm_model = joblib.load("lgbm_model.pkl")
selected_features = joblib.load("selected_features.pkl")

st.set_page_config(page_title = "Credit Risk Prediction", layout = "wide")

st.title("Loan Default Prediction")
st.write("Upload a CSV to see default predictions")

uploaded_file = st.file_uploader("Choose a CSV file", type = ["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    missing_features = [f for f in selected_features if f not in data.columns]
    if missing_features:
        st.error(f"Missing features requires: {missing_features}")
    else:
        X = data[selected_features]

        with st.spinner("Predicting default probabilities..."):
            data["Default Prob"] = lgbm_model.predict_proba(X)[:,1]
        
            threshold = st.slider("Set default probabiltiy threshold", 0.0, 1.0, 0.5)
            data["Default_Pred"] = (data["Default_Prob"] >= threshold).astype(int)

            st.subheader("Predictions")
            st.dataframe(data[selected_features + ["Default_Prob", "Default_Pred"]])

            csv = data.to_csv(index = False).encode()
            st.download_button(
                label = "Download Predictions at CSV",
                data = csv,
                file_name = "loan_predictions.csv",
                mime = "text/csv"
            )

            st.subheader("SHAP Explainability")
            
            explainer = shap.Explainer(lgbm_model, X)
            shap_values = explainer(X)
            
            st.write("Feature Importance (SHAP Summary)")
            plt.figure(figsize = (10,5))
            shap.summary_plot(shap_values, X, show = False)
            st.pyplot(plt.gcf())

            row_index = st.number_input(
                "Select applicant row for detailed SHAP explanation",
                min_value = 0,
                max_value = len(data) - 1,
                value = 0
            )

            st.write(f" SHAP explanation for applicant row {row_index}")
            shap.force_plot(
                explainer.expected_value,
                shap_values[row_index].values,
                X.iloc[row_index, :],
                matplotlib = True,
                show = False
            )
            st.pyplot(plt.gcf())

else:
    st.info("Upload a CSV file to begin with.")






