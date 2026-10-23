# 🎮 Video Game Sales Analysis and Prediction using Machine Learning

## ✅ Project Status

**Core pipeline complete. Advanced models are the next step.**

- ✅ Data Understanding
- ✅ Data Cleaning
- ✅ Univariate / Bivariate / Advanced EDA
- ✅ Feature Engineering (`Sales_Category` target)
- ✅ Baseline models: Linear Regression, Logistic Regression, KNN
- ✅ Decision Tree (regression + classification, tuned with GridSearchCV)
- ✅ Final Prediction System (`predict.py` + saved model in `Models/`)
- ⏳ Random Forest / XGBoost, time-based split, Streamlit app

### Results (held-out 20% test set)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline | 0.488 | 0.219 |
| Logistic Regression | 0.571 | 0.460 |
| KNN (k=15) | 0.571 | 0.496 |
| Decision Tree (tuned) | 0.564 | 0.484 |

Sales regression (log scale R²): Linear Regression 0.22, Decision Tree 0.18. Features are metadata only (platform, genre, year, publisher), which limits accuracy.

### Predict a new game

```
python predict.py --platform PS2 --genre Action --publisher "Electronic Arts" --year 2005
```

---

# 📌 Project Overview

This project analyzes historical video game sales data to discover trends in game sales across different platforms, genres, publishers, and release years. The project also aims to build machine learning models that can predict global video game sales and classify games into sales categories such as Hit, Average, or Flop based on historical patterns.

---

# 🎯 Objectives

- Understand the dataset
- Clean and preprocess the data
- Perform Exploratory Data Analysis (EDA)
- Engineer useful features
- Build Regression Models
- Build Classification Models
- Compare Machine Learning Models
- Develop a Prediction System
- Extract meaningful business insights

---

# 📊 Dataset Information

The dataset contains historical video game sales information including:

- Name
- Platform
- Year
- Genre
- Publisher
- North America Sales
- Europe Sales
- Japan Sales
- Other Region Sales
- Global Sales

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- Jupyter Notebook

---

# 📂 Project Structure

```
video-game-sales-analysis-and-prediction/

│
├── Data/
│
├── Images/
│
├── Models/
│
├── Notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Univariate_EDA.ipynb
│   ├── 04_Bivariate_EDA.ipynb
│   ├── 05_Advanced_EDA.ipynb
│   ├── 06_Feature_Engineering.ipynb
│   ├── 07_Prepare Data for ML and Implementation.ipynb
│   ├── 08_Decision_Tree_Models.ipynb
│   └── 09_Final_Prediction_System.ipynb
├── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🔄 Project Workflow

```
Data Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
Regression Model
        ↓
Classification Model
        ↓
Model Comparison
        ↓
Prediction System
```

---

# 📈 Current Progress

Completed

- Data Understanding, Data Cleaning
- Univariate, Bivariate and Advanced EDA
- Feature Engineering (Sales_Category: Flop / Average / Hit)
- Baseline models (Linear Regression, Logistic Regression, KNN)
- Decision Tree regression and classification (tuned with GridSearchCV)
- Final Prediction System (`predict.py` and saved model in `Models/`)

Upcoming

- Random Forest / XGBoost and a full model comparison
- Time-based train/test split and cross-validated evaluation
- Interactive app (Streamlit)

---

# 🚀 Future Improvements

- Add tree ensembles (Random Forest, XGBoost) and compare with the baselines
- Add features such as publisher track record (computed on training data only)
- Use sklearn Pipelines so preprocessing and model are saved together
- Build an interactive Streamlit app on top of `predict.py`

---

# 📄 License

This project is created for educational purposes and placement preparation.
