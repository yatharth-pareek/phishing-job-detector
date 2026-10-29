import json

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from features import clean_text, job_flag_vector, load_job_dataset

df = load_job_dataset("data/jobs.csv")
cleaned = df["text"].apply(clean_text)
flags = np.vstack(df["text"].apply(job_flag_vector).values)
y = df["label"].values

idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_tr = hstack([tfidf.fit_transform(cleaned.iloc[idx_tr]), csr_matrix(flags[idx_tr])]).tocsr()
X_te = hstack([tfidf.transform(cleaned.iloc[idx_te]), csr_matrix(flags[idx_te])]).tocsr()

# class_weight="balanced" because fake jobs are much fewer than real ones
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_tr, y[idx_tr])
pred = model.predict(X_te)

print(classification_report(y[idx_te], pred, target_names=["genuine", "fake"]))
print("Confusion matrix:\n", confusion_matrix(y[idx_te], pred))

rep = classification_report(y[idx_te], pred, output_dict=True)
metrics = {"accuracy": round(accuracy_score(y[idx_te], pred), 4),
           "precision": round(rep["1"]["precision"], 4), "recall": round(rep["1"]["recall"], 4),
           "samples": int(len(df))}
json.dump(metrics, open("models/metrics_job.json", "w"))
joblib.dump({"tfidf": tfidf, "model": model}, "models/job_model.pkl")
print("Saved models/job_model.pkl", metrics)
