# Gradient Boosting Regressor (From Scratch)

---

## 📌 Project Overview
This project implements a **Gradient Boosting Machine (GBM)** for regression tasks completely from first principles using Python and NumPy. It avoids using high-level boosting libraries (like XGBoost or LightGBM) to demonstrate a deep understanding of the underlying algorithms.

The model is trained and evaluated on the **Boston Housing Dataset**, using Decision Trees as weak learners to minimize Mean Squared Error (MSE) via gradient descent in function space.

## 🚀 Key Features
* **Custom Implementation:** Core boosting logic (`fit`, `predict`) implemented manually in the `GradientBoostingRegressorScratch` class.
* **Hyperparameter Tuning:** Supports configuration of:
    * `n_estimators` (Number of boosting stages)
    * `learning_rate` (Shrinkage parameter to prevent overfitting)
    * `max_depth` (Complexity of individual weak learners)
* **Loss Function:** Optimization based on **Squared Error Loss** ($L = \frac{1}{2}(y - \hat{y})^2$).
* **Robust Data Pipeline:** Handles the deprecated Boston Housing dataset by fetching directly from the CMU StatLib repository.

## 🛠️ Technologies Used
* **Language:** Python 3.x
* **Core Logic:** NumPy (Matrix operations), Pandas (Data handling)
* **Base Learner:** Scikit-learn (`DecisionTreeRegressor` used *only* as the weak learner)
* **Visualization:** Matplotlib (Training curves and residual plots)

## 📂 Project Structure
```text
Gradient-Boosting-Machine/
├── gbm_model.py          # Core class library containing the GBM algorithm
├── train_eval.py         # Script to load data, train model, and generate plots
├── README.md             # Project documentation
├── 1_Training_Loss_Curve.png  # (Generated) Loss minimization visualization
├── 2_Actual_vs_Predicted.png  # (Generated) Prediction scatter plot
├── 3_Residuals_Distribution.png # (Generated) Error distribution analysis
└── 4_LR_Comparison.png   # (Generated) Hyperparameter impact analysis

```
---

## ⚙️ Installation & Usage

### 1. Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn

```

### 2. Running the Training & Evaluation

Execute the main script to train the model and generate performance reports:

```bash
python train_eval.py

```

### 3. Using the Model in Your Code

You can import the class and use it just like a Scikit-learn estimator:

```python
from gbm_model import GradientBoostingRegressorScratch

# Initialize
model = GradientBoostingRegressorScratch(
    n_estimators=200, 
    learning_rate=0.1, 
    max_depth=3
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

```
---

## 📊 Performance Results
On the held-out test set (20% split), the model achieves excellent convergence:

* **Test RMSE:** 2.4525 (Root Mean Squared Error)
* **Test R²:** 0.9180 (Coefficient of Determination)
* **Train RMSE:** 0.8274

### Visualization Checkpoints

The script automatically generates the following insights:

1. **Training Loss Curve:** Verifies that the MSE decreases with each boosting iteration.
2. **Residual Analysis:** Confirms errors are normally distributed (validating regression assumptions).
3. **Learning Rate Comparison:** Demonstrates the trade-off between convergence speed and stability.

---
