# Heart Disease Prediction System

This project predicts the likelihood of heart disease using multiple machine learning models and provides an interactive interface through a Streamlit web app.

---

## Project Overview
Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve outcomes.  
This project explores different machine learning algorithms to build a predictive model and makes it accessible through a simple web interface.

---

## Features
- Implemented four machine learning models:
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Support Vector Classifier (SVC)
  - Logistic Regression
- Combined the above into a **meta-model** using Logistic Regression for the final prediction.
- Built a user-friendly web interface with **Streamlit**.
- End-to-end workflow: data preprocessing → model training → deployment.

---

## Tech Stack
- **Language:** Python  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn  
- **Interface:** Streamlit  

---

## Project Structure
```
heart-disease-prediction/
│── app.py              # Streamlit app
│── models/             # saved ML models (if applicable)
│── dataset.csv         # dataset (if shareable)
│── requirements.txt    # dependencies
│── README.md           # documentation
```

---

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open a browser and go to: `http://localhost:8501/`

---

## Dataset
The dataset includes features such as age, sex, blood pressure, cholesterol levels, and other health indicators to predict the risk of heart disease.  
kaggle heart disease dataset

---

## Future Improvements
- Hyperparameter tuning for higher accuracy  
- Feature importance visualization  
- Online deployment using Streamlit Cloud or similar platforms  

---

## Acknowledgments
- Heart Disease dataset from kaggle
- scikit-learn documentation  
- Streamlit community  
[README.md](https://github.com/user-attachments/files/22138043/README.md)
