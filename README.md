# 🩺 Diabetes Prediction & Blood Parameter Analysis

This project focuses on predicting the likelihood of diabetes in patients based on medical attributes such as glucose level, blood pressure, BMI, and more. The dataset was analyzed, cleaned, and used to train multiple machine learning models. The goal was to compare algorithms and identify which performed best for classification.

---

## 📂 Dataset
The dataset contains the following variables:

- **Pregnancies** – Number of times pregnant  
- **Glucose** – Plasma glucose concentration  
- **BloodPressure** – Diastolic blood pressure (mm Hg)  
- **SkinThickness** – Triceps skinfold thickness (mm)  
- **Insulin** – 2-Hour serum insulin (mu U/ml)  
- **BMI** – Body mass index (weight in kg/(height in m)^2)  
- **DiabetesPedigreeFunction** – Diabetes pedigree function  
- **Age** – Age of the person  
- **Outcome** – 1 for diabetic, 0 for non-diabetic  

---

## 📚 Tools & Libraries

- ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)  – Data cleaning & preprocessing
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=plotly&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-9ECAE1?style=for-the-badge&logo=seaborn&logoColor=black)  – Data visualization
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) – Model training & evaluation  

---

## 🛠️ Steps Performed

1. **Data Collection & Exploration**  
   - Loaded dataset (CSV) and explored with `.info()`, `.describe()`, statistical measures.  
   - Identified missing values and inconsistencies.  

2. **Data Preprocessing**  
   - Handled missing values.  
   - Separated data features and labels (Outcome).  
   - Standardized the dataset.  
   - Performed **train-test split** for model training.  

3. **Feature Engineering**  
   - Focused on key predictors such as **BMI, Insulin, and Glucose**.  
   - Checked feature correlations with heatmap and statistical tests.  

4. **Model Training & Evaluation**  
   Trained and compared multiple classification algorithms:
   - **SVM (Support Vector Machine)** with RBF and Polynomial kernels  
   - **Random Forest Classifier**  
   - **Gaussian Naive Bayes (NB)**  
   - **Decision Tree**  
   - **K-Nearest Neighbors (KNN)**  
   - **Logistic Regression**

   Evaluation Metrics:
   - **Accuracy Score**  
   - **Confusion Matrix**  

5. **Visualization**  
   - Correlation Matrix Heatmap  
   - Histograms of features  
   - Density plots to observe distribution  

---

## 📊 Results

- Each model was evaluated and compared based on accuracy and confusion matrices.  
- Feature importance was highlighted, especially **Glucose, BMI, and Insulin** as strong indicators.  
- Visualizations provided insights into data distribution and correlation.  

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/TasnimNishat-Dev/diabetes-prediction.git
   cd diabetes-prediction
 2. Install dependencies:
    ```bash
    pip install -r requirements.txt
 4. Run the Jupyter Notebook:
    ```bash
    jupyter notebook diabetes_prediction.ipynb
    
---

📌 Future Work

- Hyperparameter tuning for better model accuracy.
- Deployment using Flask/Streamlit for interactive predictions.
- Integration with real-time medical datasets.


---
✨ Author

Developed by [Sadia Tasnim Nishat]
- 💼 Aspiring Data Analyst | Business Analyst
- 🛠️ Skills: SQL | Python | Excel | Power BI | Retool
