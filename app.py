import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Loan Approval Classifier",
    layout="centered"
)

st.title("Loan Approval Classifier")

st.subheader("Choose a Model")

AVAILABLE_MODELS = {
    "Logistic Regression": "models/LogisticRegression.pkl",
    "Random Forest": "models/RandomForest.pkl",
    "Decision Tree": "models/DecisionTree.pkl",
    "KNN": "models/KNN.pkl",
    "Naive Bayes": "models/NaiveBayes.pkl",
    "XGBoost": "models/XGBoost.pkl"
}

selected_model = st.selectbox("Select a trained model", list(AVAILABLE_MODELS.keys()))
model = joblib.load(AVAILABLE_MODELS[selected_model])

st.success(f"Model loaded: {selected_model}")

st.subheader("Upload Test Data (CSV format)")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing applicant details (must include Loan_Status column)",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded dataset:")
    st.dataframe(data.head())

    if "Loan_Status" not in data.columns:
        st.error("The dataset must include a 'Loan_Status' column.")
        st.stop()

    X_test = data.drop(columns=["Loan_Status"])
    y_true = data["Loan_Status"]

    y_pred = model.predict(X_test)

    st.subheader("Model Evaluation")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")

    col3, col4 = st.columns(2)
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    st.subheader("Detailed Classification Report")

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

else:
    st.info("Please upload a CSV file to proceed.")