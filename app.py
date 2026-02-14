import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
)

st.set_page_config(page_title="Student Dropout Prediction", layout="wide")
st.title("Student Dropout & Academic Success Predictor")

uploaded_file = st.file_uploader("Upload Test Data (CSV)", type="csv")

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    st.write("### Test Data Preview", test_df.head())

    model_name = st.selectbox("Select Model", ["Logistic_Regression", "Decision_Tree", "kNN", "Naive_Bayes", "Random_Forest", "XGBoost"])

    try:
        with open(f'model/{model_name}.pkl', 'rb') as f: model = pickle.load(f)
        with open('model/scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('model/encoder.pkl', 'rb') as f: le = pickle.load(f)

        if st.button("Evaluate"):
            target_col = 'Target' if 'Target' in test_df.columns else test_df.columns[-1]
            X_test = test_df.drop(columns=[target_col])
            y_test_raw = test_df[target_col]

            y_test = le.transform(y_test_raw)
            X_test_scaled = scaler.transform(X_test.fillna(X_test.median()))
            
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)
            
            st.divider()
            st.subheader(f"{model_name.replace('_', ' ')} Performance Metrics")
            
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted')
            rec = recall_score(y_test, preds, average='weighted')
            f1 = f1_score(y_test, preds, average='weighted')
            mcc = matthews_corrcoef(y_test, preds)
            auc = roc_auc_score(y_test, probs, multi_class='ovr')

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Accuracy", f"{acc:.4f}")
            m_col2.metric("AUC Score", f"{auc:.4f}")
            m_col3.metric("MCC Score", f"{mcc:.4f}")

            m_col4, m_col5, m_col6 = st.columns(3)
            m_col4.metric("Precision", f"{prec:.4f}")
            m_col5.metric("Recall", f"{rec:.4f}")
            m_col6.metric("F1 Score", f"{f1:.4f}")

            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Greens', 
                            xticklabels=le.classes_, yticklabels=le.classes_)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Classification Report")
                report_dict = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4))
                
    except Exception as e:
        st.error(f"Error: {e}")