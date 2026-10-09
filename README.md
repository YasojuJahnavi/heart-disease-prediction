# Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project uses machine learning techniques to predict the likelihood of heart disease based on patient health attributes. It compares Logistic Regression and Decision Tree classification models and evaluates their performance using multiple evaluation metrics.

The project demonstrates a complete machine learning workflow, including data cleaning, exploratory data analysis (EDA), feature scaling, model training, evaluation, cross-validation, and hyperparameter tuning.

**Note:** This project is for educational purposes only and is not intended to provide medical diagnoses or replace professional medical advice.

## 🎯 Objectives

* Clean and preprocess the heart disease dataset.
* Perform exploratory data analysis and visualize patterns.
* Train Logistic Regression and Decision Tree models.
* Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
* Apply stratified cross-validation.
* Tune Decision Tree hyperparameters using GridSearchCV.
* Compare model performance and select a final model.
* Generate predictions for an example patient.

## 📂 Project Structure

```text
Heart-Disease-Prediction/
├── data/
│   └── heart.csv
├── visualizations/
├── models/
├── heart_disease_prediction_final.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `visualizations/` and `models/` folders are generated or populated by the Python script when applicable.

## 📊 Dataset

The project uses a heart disease dataset containing 13 input features and one target column.

Features include:

* `age` — Age of the patient
* `sex` — Encoded sex category
* `cp` — Chest pain type
* `trestbps` — Resting blood pressure
* `chol` — Serum cholesterol
* `fbs` — Fasting blood sugar category
* `restecg` — Resting electrocardiographic results
* `thalach` — Maximum heart rate achieved
* `exang` — Exercise-induced angina indicator
* `oldpeak` — ST depression measurement
* `slope` — Slope of the peak exercise ST segment
* `ca` — Number of major vessels
* `thal` — Thalassemia-related category

**Target variable:**

* `0` — No heart disease indicated in the dataset
* `1` — Heart disease indicated in the dataset

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

## ⚙️ Machine Learning Workflow

1. Load and inspect the dataset.
2. Check missing values and remove duplicate rows.
3. Perform EDA and generate visualizations.
4. Separate input features and target variable.
5. Split the data into training and testing sets.
6. Scale features for Logistic Regression.
7. Train Logistic Regression and Decision Tree models.
8. Evaluate model performance using classification metrics.
9. Plot confusion matrices and ROC curves.
10. Perform stratified five-fold cross-validation.
11. Tune Decision Tree hyperparameters using GridSearchCV.
12. Compare models and select the final model.
13. Generate a sample prediction and save project outputs.

## 📈 Model Evaluation

The models are evaluated using:

* **Accuracy:** Overall proportion of correct predictions.
* **Precision:** Proportion of positive predictions that are correct.
* **Recall:** Proportion of actual positive cases identified.
* **F1-score:** Harmonic mean of precision and recall.
* **ROC-AUC:** Ability to distinguish between the two target classes.

Cross-validation is used to compare model performance across multiple folds rather than relying only on a single train-test split.

## 📉 Visualizations

The project generates visualizations such as:

* Target class distribution
* Feature correlation heatmap
* Age distribution
* Heart disease distribution by sex
* Heart disease by chest pain type
* Maximum heart rate comparison
* Cholesterol comparison
* Confusion matrices
* ROC curve comparison
* Logistic Regression feature coefficient importance

The generated plots are saved as PNG files in the `visualizations/` folder.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Heart-Disease-Prediction
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Ensure `heart.csv` is available in the location expected by the Python script, such as the `data/` folder.

### 5. Run the project

```bash
python heart_disease_prediction_final.py
```

The script runs the machine learning workflow and generates the configured output files.

## 🔍 Key Learning Outcomes

* Data cleaning and preprocessing with Pandas
* Exploratory data analysis and data visualization
* Supervised classification with Scikit-learn
* Feature scaling and train-test splitting
* Classification model evaluation
* Cross-validation and hyperparameter optimization
* Saving model artifacts and visualizations
* Organizing a reproducible machine learning project

## ⚠️ Disclaimer

This is an educational machine learning project. Its predictions are not clinically validated and must not be used to diagnose, treat, or make medical decisions about any individual.

## 👩‍💻 Author

**Your Name**

Add your GitHub profile and contact information here if you want to make the project easier to explore.

## 📄 License
NO License
