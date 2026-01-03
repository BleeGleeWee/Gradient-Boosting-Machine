import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressorScratch:
    """
    Gradient Boosting Regressor implemented from scratch.
    Loss Function: Mean Squared Error (MSE)
    Gradient: Negative Residuals (y - y_pred)
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.loss_history = []
        self.initial_prediction = None

    def fit(self, X, y):
        # 1. Initialize with the mean of the target (F0)
        self.initial_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction)
        
        for i in range(self.n_estimators):
            # 2. Compute Pseudo-Residuals (Negative Gradient of MSE)
            # Loss = 0.5 * (y - pred)^2  => dLoss/dPred = -(y - pred)
            # Negative Gradient = (y - pred)
            residuals = y - y_pred
            
            # 3. Train a weak learner (Decision Tree) on the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # 4. Update Predictions
            # F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            
            # 5. Track MSE Loss for plotting
            mse = np.mean((y - y_pred) ** 2)
            self.loss_history.append(mse)
            
        return self

    def predict(self, X):
        # Start with initial mean
        y_pred = np.full(X.shape[0], self.initial_prediction)
        
        # Add contributions from all trees scaled by learning rate
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred