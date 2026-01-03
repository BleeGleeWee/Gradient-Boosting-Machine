import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from gbm_model import GradientBoostingRegressorScratch

print("Loading Boston Housing Dataset from source...")
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

print(f"Training GBM from scratch (n_estimators=200, lr=0.1)...")
model = GradientBoostingRegressorScratch(n_estimators=200, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print("-" * 30)
print("PERFORMANCE REPORT")
print("-" * 30)
print(f"Train RMSE: {rmse_train:.4f}")
print(f"Test RMSE : {rmse_test:.4f}")
print(f"Test R^2  : {r2:.4f}")
print("-" * 30)


plt.figure(figsize=(10, 5))
plt.plot(model.loss_history, label='Training MSE', color='blue')
plt.title("Gradient Boosting Training Loss (MSE)")
plt.xlabel("Boosting Iterations")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("1_Training_Loss_Curve.png")
print("Saved 1_Training_Loss_Curve.png")

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_test, alpha=0.6, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', linewidth=2)
plt.title(f"Actual vs Predicted (Test Set)\nRMSE: {rmse_test:.2f}")
plt.xlabel("Actual Housing Price")
plt.ylabel("Predicted Price")
plt.grid(True, alpha=0.3)
plt.savefig("2_Actual_vs_Predicted.png")
print("Saved 2_Actual_vs_Predicted.png")

residuals = y_test - y_pred_test
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=20, color='purple', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--')
plt.title("Residuals Distribution (Errors)")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.savefig("3_Residuals_Distribution.png")
print("Saved 3_Residuals_Distribution.png")

print("Generating Learning Rate Comparison (Plot 4)...")
lrs = [0.01, 0.1, 0.5]
plt.figure(figsize=(10, 5))
for lr in lrs:
    temp_model = GradientBoostingRegressorScratch(n_estimators=100, learning_rate=lr, max_depth=3)
    temp_model.fit(X_train, y_train)
    plt.plot(temp_model.loss_history, label=f'LR={lr}')

plt.title("Effect of Learning Rate on Convergence")
plt.xlabel("Iterations ")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("4_LR_Comparison.png")
print("Saved 4_LR_Comparison.png")
