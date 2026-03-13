"""
House Price Prediction - Main Pipeline
======================================
Models: Linear Regression, Decision Tree, Random Forest
Metrics: MAE, MSE, R² Score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'house_prices.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
print("=" * 55)
print("   HOUSE PRICE PREDICTION — ML PIPELINE")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
print(f"\n📂 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ──────────────────────────────────────────────
# 2. EDA PLOTS
# ──────────────────────────────────────────────
print("\n📊 Generating EDA visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis — House Prices', fontsize=16, fontweight='bold', y=1.01)

# Price distribution
axes[0, 0].hist(df['price'] / 1000, bins=40, color='#4472C4', edgecolor='white', alpha=0.85)
axes[0, 0].set_title('Price Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Price ($ thousands)')
axes[0, 0].set_ylabel('Frequency')

# Area vs Price
axes[0, 1].scatter(df['area_sqft'], df['price'] / 1000, alpha=0.4, color='#ED7D31', s=15)
axes[0, 1].set_title('Area vs Price', fontweight='bold')
axes[0, 1].set_xlabel('Area (sqft)')
axes[0, 1].set_ylabel('Price ($ thousands)')

# Avg price by neighborhood
neigh_avg = df.groupby('neighborhood')['price'].mean().sort_values(ascending=False) / 1000
axes[0, 2].bar(neigh_avg.index, neigh_avg.values, color='#70AD47', edgecolor='white')
axes[0, 2].set_title('Avg Price by Neighborhood', fontweight='bold')
axes[0, 2].set_xlabel('Neighborhood')
axes[0, 2].set_ylabel('Avg Price ($ thousands)')
axes[0, 2].tick_params(axis='x', rotation=30)

# Bedrooms vs Price boxplot
df.boxplot(column='price', by='bedrooms', ax=axes[1, 0],
           boxprops=dict(color='#4472C4'),
           medianprops=dict(color='red', linewidth=2))
axes[1, 0].set_title('Price by Bedrooms', fontweight='bold')
axes[1, 0].set_xlabel('Bedrooms')
axes[1, 0].set_ylabel('Price ($)')
plt.sca(axes[1, 0])
plt.title('Price by Bedrooms', fontweight='bold')

# Distance vs Price
axes[1, 1].scatter(df['distance_to_center_km'], df['price'] / 1000,
                   alpha=0.4, color='#9E480E', s=15)
axes[1, 1].set_title('Distance to Center vs Price', fontweight='bold')
axes[1, 1].set_xlabel('Distance (km)')
axes[1, 1].set_ylabel('Price ($ thousands)')

# House Age vs Price
axes[1, 2].scatter(df['house_age'], df['price'] / 1000,
                   alpha=0.4, color='#7030A0', s=15)
axes[1, 2].set_title('House Age vs Price', fontweight='bold')
axes[1, 2].set_xlabel('Age (years)')
axes[1, 2].set_ylabel('Price ($ thousands)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_plots.png'), bbox_inches='tight')
plt.close()
print("   ✅ EDA plots saved.")

# ──────────────────────────────────────────────
# 3. CORRELATION HEATMAP
# ──────────────────────────────────────────────
numeric_df = df.select_dtypes(include=np.number)
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, ax=ax, linewidths=0.5,
            annot_kws={"size": 9})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), bbox_inches='tight')
plt.close()
print("   ✅ Correlation heatmap saved.")

# ──────────────────────────────────────────────
# 4. DATA CLEANING
# ──────────────────────────────────────────────
print("\n🧹 Data Cleaning...")
print(f"   Missing values before:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
df['garage_spaces'] = df['garage_spaces'].fillna(df['garage_spaces'].median())

print(f"   Missing values after: {df.isnull().sum().sum()} ✅")

# ──────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ──────────────────────────────────────────────
print("\n⚙️  Feature Engineering...")

# Encode neighborhood
le = LabelEncoder()
df['neighborhood_encoded'] = le.fit_transform(df['neighborhood'])

# New features
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['area_per_room'] = df['area_sqft'] / df['total_rooms']
df['is_new_house'] = (df['house_age'] <= 10).astype(int)
df['luxury_score'] = df['has_pool'] + df['has_garden'] + (df['garage_spaces'] > 1).astype(int)

print("   ✅ New features: total_rooms, area_per_room, is_new_house, luxury_score")

# ──────────────────────────────────────────────
# 6. PREPARE FEATURES
# ──────────────────────────────────────────────
features = [
    'area_sqft', 'bedrooms', 'bathrooms', 'floors', 'garage_spaces',
    'house_age', 'has_pool', 'has_garden', 'distance_to_center_km',
    'neighborhood_encoded', 'total_rooms', 'area_per_room',
    'is_new_house', 'luxury_score'
]

X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📐 Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

# ──────────────────────────────────────────────
# 7. TRAIN MODELS
# ──────────────────────────────────────────────
print("\n🤖 Training Models...")

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
}

results = {}

for name, model in models.items():
    use_scaled = (name == 'Linear Regression')
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring='r2')

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std(),
    }

    print(f"\n   📌 {name}")
    print(f"      MAE  : ${mae:,.0f}")
    print(f"      RMSE : ${rmse:,.0f}")
    print(f"      R²   : {r2:.4f}")
    print(f"      CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    joblib.dump(model, os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.pkl"))

# ──────────────────────────────────────────────
# 8. MODEL COMPARISON PLOT
# ──────────────────────────────────────────────
print("\n📈 Generating model comparison plots...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

model_names = list(results.keys())
colors = ['#4472C4', '#ED7D31', '#70AD47']

# MAE
mae_vals = [results[m]['MAE'] / 1000 for m in model_names]
axes[0].bar(model_names, mae_vals, color=colors, edgecolor='white', width=0.5)
axes[0].set_title('MAE (lower is better)', fontweight='bold')
axes[0].set_ylabel('MAE ($ thousands)')
for i, v in enumerate(mae_vals):
    axes[0].text(i, v + 0.5, f'${v:.1f}k', ha='center', fontsize=10, fontweight='bold')

# RMSE
rmse_vals = [results[m]['RMSE'] / 1000 for m in model_names]
axes[1].bar(model_names, rmse_vals, color=colors, edgecolor='white', width=0.5)
axes[1].set_title('RMSE (lower is better)', fontweight='bold')
axes[1].set_ylabel('RMSE ($ thousands)')
for i, v in enumerate(rmse_vals):
    axes[1].text(i, v + 0.5, f'${v:.1f}k', ha='center', fontsize=10, fontweight='bold')

# R2
r2_vals = [results[m]['R2'] for m in model_names]
axes[2].bar(model_names, r2_vals, color=colors, edgecolor='white', width=0.5)
axes[2].set_title('R² Score (higher is better)', fontweight='bold')
axes[2].set_ylabel('R² Score')
axes[2].set_ylim(0, 1.1)
for i, v in enumerate(r2_vals):
    axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

for ax in axes:
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), bbox_inches='tight')
plt.close()
print("   ✅ Model comparison plot saved.")

# ──────────────────────────────────────────────
# 9. ACTUAL VS PREDICTED PLOTS
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Actual vs Predicted Prices', fontsize=14, fontweight='bold')

for ax, (name, res), color in zip(axes, results.items(), colors):
    ax.scatter(y_test / 1000, res['y_pred'] / 1000,
               alpha=0.4, color=color, s=20, label='Predictions')
    min_val = min(y_test.min(), res['y_pred'].min()) / 1000
    max_val = max(y_test.max(), res['y_pred'].max()) / 1000
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Perfect fit')
    ax.set_title(f'{name}\nR² = {res["R2"]:.3f}', fontweight='bold')
    ax.set_xlabel('Actual Price ($ thousands)')
    ax.set_ylabel('Predicted Price ($ thousands)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png'), bbox_inches='tight')
plt.close()
print("   ✅ Actual vs Predicted plots saved.")

# ──────────────────────────────────────────────
# 10. FEATURE IMPORTANCE (Random Forest)
# ──────────────────────────────────────────────
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
importances.plot(kind='barh', ax=ax, color='#4472C4', edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), bbox_inches='tight')
plt.close()
print("   ✅ Feature importance plot saved.")

# ──────────────────────────────────────────────
# 11. SUMMARY TABLE
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("   FINAL RESULTS SUMMARY")
print("=" * 55)
summary = pd.DataFrame({
    'Model': model_names,
    'MAE ($)': [f"${results[m]['MAE']:,.0f}" for m in model_names],
    'RMSE ($)': [f"${results[m]['RMSE']:,.0f}" for m in model_names],
    'R² Score': [f"{results[m]['R2']:.4f}" for m in model_names],
    'CV R² (mean)': [f"{results[m]['CV_R2_mean']:.4f}" for m in model_names],
})
print(summary.to_string(index=False))

best_model = max(results, key=lambda m: results[m]['R2'])
print(f"\n🏆 Best Model: {best_model} (R² = {results[best_model]['R2']:.4f})")

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

print("\n✅ All models saved to /models/")
print("✅ All plots saved to /outputs/")
print("\nPipeline complete! 🎉")
