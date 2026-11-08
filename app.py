import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

st.set_page_config(page_title="Spam Detector – ML Assignment 2", layout="wide")
st.title("Spam Email Classifier")   SPAM Classifier:ID 2025AG05922
st.write("Upload the test CSV, pick a model, and see how it performs.")  SPAN CLassifier Assignment_ID_2025AG05922

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "SVM": "model/svm.pkl",
}

st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))
uploaded = st.sidebar.file_uploader("Upload test_data.csv", type="csv")

if uploaded is None:
    st.info("Upload test_data.csv from the sidebar to begin.")
else:
    data = pd.read_csv(uploaded)
    X = data.drop(columns="is_spam")
    y = data["is_spam"]
    model = joblib.load(MODEL_FILES[model_name])   # saved model, no retraining
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    st.subheader(f"Results for {model_name}")
    c = st.columns(6)
    c[0].metric("Accuracy", f"{accuracy_score(y, pred):.3f}")
    c[1].metric("AUC", f"{roc_auc_score(y, prob):.3f}")
    c[2].metric("Precision", f"{precision_score(y, pred):.3f}")
    c[3].metric("Recall", f"{recall_score(y, pred):.3f}")
    c[4].metric("F1", f"{f1_score(y, pred):.3f}")
    c[5].metric("MCC", f"{matthews_corrcoef(y, pred):.3f}")

    left, right = st.columns(2)
    with left:
        st.write("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(confusion_matrix(y, pred),
                               display_labels=["Not spam", "Spam"]).plot(ax=ax, colorbar=False)
        st.pyplot(fig)
    with right:
        st.write("**Classification report**")
        st.dataframe(pd.DataFrame(classification_report(y, pred, output_dict=True)).T.round(3))
