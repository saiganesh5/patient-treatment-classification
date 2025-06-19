# 🏥 Patient Treatment Classification

This project is a Machine Learning-based solution designed to classify patients into appropriate treatment categories based on their health profile. It demonstrates how data-driven models can assist in clinical decision-making and improve personalized healthcare.

---

## 📌 Project Overview

The objective of this project is to build a predictive model that can classify the type of treatment a patient should receive using medical features such as:

- Age  
- Gender  
- Cholesterol  
- Blood Pressure  
- Body Mass Index (BMI)  

---

## 🧪 Technologies Used

- **Language**: Python  
- **IDE**: Jupyter Notebook  
- **Libraries**:
  - `pandas` – Data handling
  - `numpy` – Numerical operations
  - `matplotlib`, `seaborn` – Data visualization
  - `scikit-learn` – Machine Learning models and metrics

---

## ⚙️ Workflow

1. **Data Preprocessing**
   - Handled missing values
   - Encoded categorical variables
   - Scaled numerical features

2. **Exploratory Data Analysis (EDA)**
   - Visualized distributions and correlations
   - Checked for class imbalance

3. **Model Training**
   - Trained several classifiers:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Gaussian Naive Bayes
     - Gradient Boosting
     - Multi-Layer Perceptron (MLP)

4. **Model Ensemble**
   - Built a **Stacking Classifier** for improved generalization

5. **Model Evaluation**
   - Evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix

---

## 📊 Results

The Stacking Classifier outperformed individual models in terms of balanced accuracy and robustness. This validates the use of ensemble learning for real-world healthcare classification tasks.

---

## 📁 Project Structure

```plaintext
├── data/
│   └── patient_data.csv
├── models/
│   └── trained_models.pkl
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── README.md
└── requirements.txt
