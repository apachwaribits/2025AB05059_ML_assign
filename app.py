import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Student Dropout Prediction")
st.title("🎓 Student Dropout & Academic Success Predictor")

uploaded_file = st.file_uploader("Upload Test Data (CSV)", type="csv")

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    st.write("### Test Data Preview", test_df.head())

    model_name = st.selectbox("Select Model", ["Logistic_Regression", "Decision_Tree", "kNN", "Naive_Bayes", "Random_Forest", "XGBoost"])

    try:
        # Load Model and Transformers
        with open(f'model/{model_name}.pkl', 'rb') as f: model = pickle.load(f)
        with open('model/scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('model/encoder.pkl', 'rb') as f: le = pickle.load(f)

        if st.button("Evaluate"):
            # Prepare Features
            target_col = 'Target' if 'Target' in test_df.columns else test_df.columns[-1]
            X_test = test_df.drop(columns=[target_col])
            y_test_raw = test_df[target_col]

            # FIX: Convert string labels to numbers using the saved encoder
            y_test = le.transform(y_test_raw)
            
            # Apply Scaling
            X_test_scaled = scaler.transform(X_test.fillna(X_test.median()))
            
            # Predict
            preds = model.predict(X_test_scaled)
            
            # Display Results
            st.divider()
            st.metric("Model Accuracy", f"{accuracy_score(y_test, preds):.4f}")
            
            

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Greens', 
                            xticklabels=le.classes_, yticklabels=le.classes_)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Classification Report")
                st.text(classification_report(y_test, preds, target_names=le.classes_))
                
    except Exception as e:
        st.error(f"Error: {e}")
