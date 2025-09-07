# src/models/double_ml.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np

class ContinuousDoubleML:
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.effect_ = None
        self.residuals_ = []

    def fit(self, X, T, Y):
        np.random.seed(self.random_state)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        psi_n = np.zeros(X.shape[0])

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Estimate T ~ X
            m_hat = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            m_hat.fit(X_train, T_train)
            v_hat = T_test - m_hat.predict(X_test)

            # Estimate Y ~ X
            g_hat = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            g_hat.fit(X_train, Y_train)
            epsilon_hat = Y_test - g_hat.predict(X_test)

            # Orthogonal score
            psi_n[test_idx] = (epsilon_hat * v_hat) / (v_hat ** 2 + 1e-8)

        self.effect_ = np.mean(psi_n)
        self.residuals_ = psi_n
        return self.effect_