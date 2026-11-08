import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

os.makedirs("models", exist_ok=True)

df = pd.read_csv("dataset.csv").dropna()

cat = ["state", "district", "place", "location", "area"]
num = ["square_feet", "bedrooms", "bathrooms"]
X = df[cat + num]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def scaled_prep():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])


# ---------- 1. Regression models ----------
# Random Forest: the model the website uses (step names must stay "prep" and "rf")
rf_model = Pipeline([
    ("prep", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
        remainder="passthrough",
    )),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
])

# Linear Regression: the simple regression from theory
linear_model = Pipeline([
    ("prep", scaled_prep()),
    ("lr", LinearRegression()),
])

# Neural network: the Deep Learning example (prices are scaled so it learns well)
nn_model = TransformedTargetRegressor(
    regressor=Pipeline([
        ("prep", scaled_prep()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=42)),
    ]),
    transformer=StandardScaler(),
)

models = {
    "Linear Regression": linear_model,
    "Random Forest": rf_model,
    "Neural Network (Deep Learning)": nn_model,
}

# ---------- 2. Train and check each model ----------
scores = []
predictions = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    predictions[name] = pred
    scores.append({
        "model": name,
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 0),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 0),
    })

print("\n=== Model check (on test data the model never saw) ===")
for s in scores:
    print(f"{s['model']:32s} R2 = {s['r2']:.3f}   MAE = Rs {s['mae']:,.0f}   RMSE = Rs {s['rmse']:,.0f}")

with open("models/metrics.json", "w") as f:
    json.dump(scores, f, indent=2)

# Actual vs predicted (for the chart in the website)
pd.DataFrame({
    "actual": y_test.values,
    "predicted": predictions["Random Forest"].round(0),
}).to_csv("models/test_results.csv", index=False)

# ---------- 3. Save the model used by the website ----------
joblib.dump(rf_model, "models/house_model.pkl")
print("\nModel saved in models/house_model.pkl")

# ---------- 4. K-means: group houses into segments ----------
feat = df[["price", "square_feet"]]
feat_scaled = StandardScaler().fit_transform(feat)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(feat_scaled)

seg = df[["house_id", "price"]].copy()
seg["cluster"] = km.labels_
order = seg.groupby("cluster")["price"].mean().sort_values().index.tolist()
names = {order[0]: "Budget", order[1]: "Mid-range", order[2]: "Luxury"}
seg["segment"] = seg["cluster"].map(names)
seg[["house_id", "segment"]].to_csv("models/segments.csv", index=False)
print("K-means segments saved:", seg["segment"].value_counts().to_dict())