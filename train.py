import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import joblib
import os

from sklearn.ensemble import RandomForestRegressor

import json
from datetime import datetime


BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# make sure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. load data
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# 2. split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. create v0.1 pipeline (StandardScaler + LinearRegression)
pipeline_v01 = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

# 4. train v0.1
print("Training v0.1 (LinearRegression)...")
pipeline_v01.fit(X_train, y_train)

# 5. evaluate v0.1
y_pred = pipeline_v01.predict(X_test)
rmse_v01 = mean_squared_error(y_test, y_pred)**0.5 # corresponding to MLOps 1
print(f"v0.1 (LinearRegression) 's  RMSE: {rmse_v01}")

# 6. save v0.1 model (using absolute path and correct filename 'model_v01.joblib')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'model_v01.joblib')

joblib.dump(pipeline_v01, MODEL_SAVE_PATH)
print(f"v0.1 model has been saved to: {MODEL_SAVE_PATH}")


#v2
print("\n--- Training v0.2 (RandomForest) ---")

# 7. create v0.2 pipeline (StandardScaler + RandomForest)
# This is an Ensemble model (corresponding to MLOps 2 courseware)
pipeline_v02 = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(random_state=42, n_estimators=100)
)

# 8. train v0.2
pipeline_v02.fit(X_train, y_train)

# 9. evaluate v0.2
y_pred_v02 = pipeline_v02.predict(X_test)
rmse_v02 = mean_squared_error(y_test, y_pred_v02)**0.5
print(f"v0.2 (RandomForest) 's RMSE: {rmse_v02}")

# 10. save v0.2 model (using absolute path and correct filename 'model_v02.joblib')
MODEL_V02_SAVE_PATH = os.path.join(MODELS_DIR, 'model_v02.joblib')
joblib.dump(pipeline_v02, MODEL_V02_SAVE_PATH)
print(f"v0.2 model has been saved to: {MODEL_V02_SAVE_PATH}")

print("\n--- Training Summary ---")
print(f"v0.1 (LinearRegression) RMSE: {rmse_v01}")
print(f"v0.2 (RandomForest) RMSE: {rmse_v02}")

metrics = {
    "v01": {
        "model": "LinearRegression",
        "rmse": float(rmse_v01)
    },
    "v02": {
        "model": "RandomForestRegressor", 
        "rmse": float(rmse_v02)
    },
    "timestamp": datetime.now().isoformat(),
    "dataset": "sklearn.diabetes"
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to metrics.json")
