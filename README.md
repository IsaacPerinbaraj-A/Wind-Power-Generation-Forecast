#  Wind Turbine SCADA Project – Week 1  

## 📌 Overview  
Forecasting wind power generation using SCADA dataset to predict **active power output (kW)** from wind speed, wind direction, and theoretical power curve.  

## 📂 Dataset  
- **Source:** [Kaggle – Wind Turbine SCADA Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)  
- **Features:**  
  - Date/Time  
  - Wind Speed  
  - Wind Direction  
  - Theoretical Power Curve  
  - LV Active Power  

## 🛠️ Work Done (Week 1)  
- Imported libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`)  
- Loaded dataset from CSV  
- Cleaned data (datetime conversion, handled negative values)  
- Basic EDA:  
  - Average power per hour of day (bar chart)  
  - Wind Speed vs Power (scatter plot)  

# Wind Turbine SCADA Project – Week 2

## 📌 Overview  
Predicting **active power output (kW)** using **Machine Learning models** (Linear Regression & XGBoost) from SCADA data. The goal is to compare model performance and improve prediction accuracy.

## 🛠️ Work Done (Week 2)  
- Imported libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`  
- Preprocessed dataset (datetime conversion, feature extraction)  
- Split dataset into training (80%) and testing (20%)  
- Scaled features for XGBoost  
- Built **Linear Regression** model:  
  - Trained, predicted, and evaluated using MAE, RMSE, R²  
  - Plotted actual vs predicted power output  
- Built **XGBoost Regressor**:  
  - Trained, predicted, and evaluated  
  - Plotted actual vs predicted power output  
- Compared model performance  

## 📊 Model Performance  

| Model             | MAE     | RMSE    | R² Score |
|-------------------|---------|---------|----------|
| Linear Regression | 196.68  | 407.66  | 0.9026   |
| XGBoost           | 84.12   | 200.74  | 0.9764   |

> **Observation:** XGBoost captures nonlinear relationships better, while Linear Regression provides a simple linear baseline.

## 🎯 Next Steps  
- Test **Random Forest, SVR, and MLP models**  
- Explore additional features (temperature, air density)  
- Develop a **forecasting dashboard** for real-time predictions  

