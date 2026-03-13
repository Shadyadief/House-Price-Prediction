# 🏠 House Price Prediction using Machine Learning

A complete end-to-end machine learning project to predict house prices based on property features such as area, number of rooms, neighborhood, and more.

---

## 📌 Project Overview

This project helps real estate companies and investors estimate property values accurately using data-driven models. It demonstrates the full ML pipeline: from data analysis to model deployment.

**Goal:** Build and compare multiple ML regression models to predict house prices with high accuracy.

---

## 🗂️ Project Structure

```
house_price_prediction/
│
├── data/
│   └── house_prices.csv          # Dataset (1000 houses)
│
├── src/
│   ├── generate_data.py          # Dataset generation script
│   ├── train.py                  # Full ML pipeline (EDA → Training → Evaluation)
│   └── predict.py                # Predict price for a new house
│
├── models/
│   ├── linear_regression.pkl     # Saved Linear Regression model
│   ├── decision_tree.pkl         # Saved Decision Tree model
│   ├── random_forest.pkl         # Saved Random Forest model
│   ├── scaler.pkl                # StandardScaler
│   └── label_encoder.pkl        # LabelEncoder for neighborhood
│
├── outputs/
│   ├── eda_plots.png             # EDA visualizations
│   ├── correlation_heatmap.png   # Feature correlation heatmap
│   ├── model_comparison.png      # MAE / RMSE / R² comparison
│   ├── actual_vs_predicted.png   # Scatter plots per model
│   └── feature_importance.png    # Random Forest feature importance
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `area_sqft` | Total area of the house in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `floors` | Number of floors |
| `garage_spaces` | Number of garage spaces |
| `year_built` | Year the house was built |
| `house_age` | Age of the house (engineered) |
| `has_pool` | Pool availability (0/1) |
| `has_garden` | Garden availability (0/1) |
| `distance_to_center_km` | Distance to city center in km |
| `neighborhood` | Neighborhood category |
| `price` | Target variable — house price in USD |

---

## ⚙️ Methodology

```
1. Load & Explore Data (EDA)
       ↓
2. Data Cleaning (handle missing values)
       ↓
3. Feature Engineering (new meaningful features)
       ↓
4. Train/Test Split (80% / 20%)
       ↓
5. Train 3 Models
       ↓
6. Evaluate with MAE, MSE, RMSE, R²
       ↓
7. Select Best Model → Save
```

---

## 🤖 Models Used

| Model | Description |
|---|---|
| Linear Regression | Baseline model — fast, interpretable |
| Decision Tree Regressor | Non-linear, handles complex patterns |
| Random Forest Regressor | Ensemble of trees — best performance |

---

## 📈 Results

| Model | MAE | RMSE | R² Score | CV R² |
|---|---|---|---|---|
| Linear Regression | $74,570 | $90,042 | 0.8264 | 0.8377 |
| Decision Tree | $42,265 | $55,271 | 0.9346 | 0.9379 |
| **Random Forest** | **$28,806** | **$35,498** | **0.9730** | **0.9713** |

🏆 **Best Model: Random Forest** — R² Score of **0.9730**

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset
```bash
python src/generate_data.py
```

### 4. Train Models
```bash
python src/train.py
```

### 5. Predict a New House Price
```bash
python src/predict.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

---

## 🔑 Key Findings

- **Area (sqft)** and **Neighborhood** are the strongest price predictors
- **Waterfront** and **Downtown** neighborhoods command the highest premiums
- Houses within **5 km of city center** are priced significantly higher
- **Random Forest** outperforms other models with 97.3% variance explained (R²)
- Feature engineering (luxury_score, area_per_room) improved model performance

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-red)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> Built as part of the **Digilians AI & Machine Learning Track** 🚀
