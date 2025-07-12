# Autism Prediction using Machine Learning

This project focuses on building a machine learning model to predict the likelihood of Autism Spectrum Disorder (ASD) based on screening data. It was developed entirely in Google Colab and showcases a full machine learning pipeline from preprocessing to model evaluation.

---

## Project Files

| File | Description |
|------|-------------|
| `Autism_prediction.ipynb` | Google Colab notebook with full ML workflow |
| `train.csv` | Dataset used for training and testing the model |
| `README.md` | Project overview and documentation |

> *Note: This project was built in Google Colab.*

---

## Dataset

The dataset includes:
- Demographics (age, gender, ethnicity)
- Behavioral screening answers (based on AQ-10)
- Medical/family history indicators (jaundice, family ASD, etc.)

> 📦 *Dataset source: https://www.kaggle.com/datasets/shivamshinde123/autismprediction?select=test.csv*

---

## Techniques Used

- Data Cleaning and Preprocessing  
  - Missing value handling  
  - Label Encoding  
- Exploratory Data Analysis (EDA)  
- Classification Models:
  - Decision Tree  
  - Random Forest  
  - XGBoost  
- Hyperparameter Tuning with `RandomizedSearchCV`
- Model Evaluation using cross-validation scores

---

## Results

- **Best Model**: `RandomForestClassifier`
- **Cross-Validation Accuracy**: ****92.71%**

---

## How to Run

1. Open the `.ipynb` file in [Google Colab](https://colab.research.google.com)
2. Upload the `train.csv` file when prompted
3. Make sure required libraries are installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

Future Improvements
Build a Streamlit or Flask web app for real-time predictions

Add additional evaluation metrics (ROC-AUC, Precision-Recall)

Explore neural network-based approaches
