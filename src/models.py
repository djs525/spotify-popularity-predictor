"""
models.py
---------
Model definitions and training helpers for the Spotify popularity prediction project.
Covers: Linear Regression, Random Forest, LightGBM (with Optuna tuning), Neural Network.
"""

import numpy as np
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Linear Regression (baseline)
# ---------------------------------------------------------------------------

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 2. Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(X_train, y_train, n_estimators=200, max_depth=15,
                        random_state=42, n_jobs=-1):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 3. LightGBM
# ---------------------------------------------------------------------------

def train_lightgbm(X_train, y_train, params: dict = None, random_state: int = 42):
    """Train with fixed params (used after tuning)."""
    default_params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def tune_lightgbm(X_train, y_train, n_trials: int = 50, cv: int = 5,
                  random_state: int = 42):
    """
    Use Optuna to find the best LightGBM hyperparameters via cross-validation.
    Returns the best params dict and the Optuna study object.
    """
    from sklearn.model_selection import cross_val_score

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        return -scores.mean()  # Optuna minimises

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest CV RMSE: {study.best_value:.4f}")
    print(f"Best params:  {study.best_params}")
    return study.best_params, study


# ---------------------------------------------------------------------------
# 4. Neural Network (PyTorch MLP)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_neural_network(X_train, y_train, X_val=None, y_val=None,
                          hidden_dims=(256, 128, 64), dropout=0.3,
                          lr=1e-3, batch_size=512, max_epochs=100,
                          patience=10, device=None):
    """
    Train an MLP with early stopping on validation RMSE.
    If X_val/y_val are not provided, uses a 10% split from training data.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Val split if not provided
    if X_val is None:
        split = int(len(X_train) * 0.9)
        X_val, y_val = X_train[split:], y_train[split:]
        X_train, y_train = X_train[:split], y_train[:split]

    def to_tensors(X, y):
        return (torch.tensor(X, dtype=torch.float32).to(device),
                torch.tensor(y, dtype=torch.float32).to(device))

    X_tr, y_tr = to_tensors(X_train, y_train)
    X_vl, y_vl = to_tensors(X_val, y_val)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = MLP(X_train.shape[1], hidden_dims, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_rmse": [], "val_rmse": []}

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_rmse = (train_loss / len(X_tr)) ** 0.5

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl)
            val_rmse = ((criterion(val_pred, y_vl)).item()) ** 0.5

        scheduler.step(val_rmse)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train RMSE {train_rmse:.3f} | val RMSE {val_rmse:.3f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history


def predict_nn(model, X, device=None, batch_size=1024):
    """Run inference on a numpy array, return numpy array."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# 5. Save / load helpers
# ---------------------------------------------------------------------------

def save_model(model, name: str, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), f"{model_dir}/{name}.pt")
    else:
        joblib.dump(model, f"{model_dir}/{name}.pkl")
    print(f"  Saved {model_dir}/{name}")


def load_sklearn_model(name: str, model_dir: str = "models"):
    return joblib.load(f"{model_dir}/{name}.pkl")