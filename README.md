# Phishing Website & Fake Job Detection System

Tech: Python, Pandas, Scikit-learn, NLP (TF-IDF), FastAPI (+ simple HTML page for the UI)

## Run (4 commands)
```
pip install -r requirements.txt
python generate_data.py     # demo data (skip if you use Kaggle files)
python train_url.py
python train_job.py
python -m uvicorn main:app --reload
```
Open http://localhost:8000  (API docs: http://localhost:8000/docs)

## Use real Kaggle data (recommended for resume/interview)
1. "Phishing Site URLs" (Kaggle) -> save as `data/urls.csv`  (columns URL, Label = good/bad)
2. "Fake Job Postings" (Kaggle, EMSCAD) -> save as `data/jobs.csv` (title, description, requirements, fraudulent)
3. Run `python train_url.py` and `python train_job.py` again. Done.

## API
- POST /predict-url  {"url": "..."}   -> result, risk_score, risk_level, indicators
- POST /predict-job  {"text": "..."}  -> result, risk_score, risk_level, indicators
- GET  /metrics      -> accuracy / precision / recall of both models
