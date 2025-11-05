# task2_predict_range.py
# CODTECH - Task 2: Predict Electric Range (simple ML)
# BY Varshini G S

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# ---- 1. Load dataset ----
csv_path = os.path.join(os.getcwd(), "Electric_Vehicle_Population_Data.csv")
print("Loading:", csv_path)
df = pd.read_csv(csv_path)

# ---- 2. Quick data check ----
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

# ---- 3. Select features and target ----
# We pick a few simple numeric columns that likely influence electric range
cols_needed = ['Model Year', 'Base MSRP', 'Legislative District', 'Electric Range']
df = df[cols_needed].copy()

# ---- 4. Clean data ----
df = df.dropna()
# Optionally remove impossible values (e.g., Electric Range <= 0)
df = df[df['Electric Range'] > 0]

print("\nAfter cleaning, rows:", len(df))
print(df.describe().T[['count','mean','std','min','50%','max']])

# ---- 5. Prepare X and y ----
X = df[['Model Year', 'Base MSRP', 'Legislative District']]
y = df['Electric Range']

# ---- 6. Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining rows:", len(X_train), "Testing rows:", len(X_test))

# ---- 7. Train model ----
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel coefficients:", model.coef_)
print("Model intercept:", model.intercept_)

# ---- 8. Predict and evaluate ----
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation:")
print("Mean Squared Error (MSE):", round(mse, 3))
print("Root MSE:", round(mse**0.5, 3))
print("R^2 score:", round(r2, 3))

# ---- 9. Save a simple results CSV (actual vs predicted) ----
results = pd.DataFrame({
    'actual_range': y_test.values,
    'predicted_range': y_pred
})
out_csv = os.path.join(os.getcwd(), "output", "task2_predictions.csv")
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
results.to_csv(out_csv, index=False)
print("Saved predictions to:", out_csv)

# ---- 10. Plot Actual vs Predicted ----
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect line
plt.xlabel("Actual Electric Range")
plt.ylabel("Predicted Electric Range")
plt.title("Actual vs Predicted Electric Range")
plt.tight_layout()
fig_path = os.path.join(os.getcwd(), "output", "task2_actual_vs_pred.png")
plt.savefig(fig_path)
print("Saved plot to:", fig_path)
plt.show()
