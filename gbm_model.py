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
        self.initial_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction)
        
        for i in range(self.n_estimators):
            residuals = y - y_pred
            
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)
            self.trees.append(tree)

            update = tree.predict(X)
            y_pred += self.learning_rate * update
            
            mse = np.mean((y - y_pred) ** 2)
            self.loss_history.append(mse)
            
        return self

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction)
        
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred
