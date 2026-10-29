import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from features import load_url_dataset, url_features

df = load_url_dataset("data/urls.csv")
if len(df) > 100_000:                       # big Kaggle file -> use a sample so training is quick
    df = df.sample(100_000, random_state=42)

X = pd.DataFrame([url_features(u) for u in df["url"]])
# Kaggle URLs usually have no http/https prefix, so this column would be useless for training.
# (It is still used to show the "No HTTPS" risk indicator in the app.)
X = X.drop(columns=["has_https"])
y = df["label"].values
names = list(X.columns)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Limited depth / leaf size keeps the saved model small (needed for free hosting)
model = RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=10,
                               random_state=42, n_jobs=-1, class_weight="balanced")
model.fit(X_tr, y_tr)
pred = model.predict(X_te)

print(classification_report(y_te, pred, target_names=["safe", "phishing"]))
print("Confusion matrix:\n", confusion_matrix(y_te, pred))

rep = classification_report(y_te, pred, output_dict=True)
metrics = {"accuracy": round(accuracy_score(y_te, pred), 4),
           "precision": round(rep["1"]["precision"], 4), "recall": round(rep["1"]["recall"], 4),
           "samples": int(len(df))}
json.dump(metrics, open("models/metrics_url.json", "w"))
joblib.dump({"model": model, "features": names}, "models/url_model.pkl", compress=3)
print("Saved models/url_model.pkl", metrics)