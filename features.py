"""Feature engineering: URL features, job-text cleaning and red-flag checks."""
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ URL part
SUSPICIOUS_WORDS = ["login", "verify", "secure", "account", "update", "bank",
                    "confirm", "password", "signin", "wallet", "bonus", "free"]
SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly"]
BAD_TLDS = (".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".gq", ".click", ".work", ".info")
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _parse(url: str):
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    try:
        p = urlparse(url)
    except ValueError:
        # Some real-world URLs are malformed (e.g. a stray '[' or ']'): clean and retry
        try:
            p = urlparse(re.sub(r"[\[\]]", "", url))
        except ValueError:
            p = urlparse("http://invalid")
    return p, (p.hostname or "")


def url_features(url: str) -> dict:
    raw_full = url.strip().lower()
    # remove "http://" / "https://" so URLs with and without a scheme look the same to the model
    raw = re.sub(r"^[a-z][a-z0-9+.-]*://", "", raw_full)
    p, host = _parse(raw_full)
    path_q = (p.path or "") + ("?" + p.query if p.query else "")
    digits = sum(c.isdigit() for c in raw)
    return {
        "url_length": len(raw),
        "host_length": len(host),
        "path_length": len(path_q),
        "dot_count": raw.count("."),
        "hyphen_count": host.count("-"),
        "at_symbol": int("@" in raw),
        "digit_ratio": digits / max(len(raw), 1),
        "has_ip": int(bool(IP_RE.match(host))),
        "subdomain_count": max(host.count(".") - 1, 0),
        "has_https": int(raw_full.startswith("https")),
        "suspicious_words": sum(w in raw for w in SUSPICIOUS_WORDS),
        "is_shortener": int(any(host == s for s in SHORTENERS)),
        "bad_tld": int(host.endswith(BAD_TLDS)),
        "query_params": raw.count("="),
        "double_slash_in_path": int("//" in (p.path or "")),
        "has_port": int(p.port is not None) if _safe_port(p) else 0,
    }


def _safe_port(p):
    try:
        p.port
        return True
    except ValueError:
        return False


def url_indicators(url: str, f: dict) -> list:
    out = []
    if f["has_ip"]:
        out.append("An IP address is used instead of a domain name")
    if f["at_symbol"]:
        out.append("URL contains '@' (can be used to hide the real website)")
    if f["is_shortener"]:
        out.append("Shortened link: the real destination is hidden")
    if f["bad_tld"]:
        out.append("Domain ending is often seen in spam/phishing sites")
    if f["suspicious_words"] >= 2:
        out.append("URL contains suspicious words (login, verify, secure, update...)")
    if f["subdomain_count"] >= 3:
        out.append("Too many subdomains")
    if f["hyphen_count"] >= 2:
        out.append("Domain contains multiple hyphens (-)")
    if f["url_length"] > 75:
        out.append("URL is unusually long")
    if not f["has_https"]:
        out.append("No HTTPS (connection is not secure)")
    return out


def load_url_dataset(path: str) -> pd.DataFrame:
    """Accepts our own csv or the Kaggle 'Phishing Site URLs' csv (URL, Label=good/bad)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    url_col = "url" if "url" in df.columns else df.columns[0]
    lab_col = "label" if "label" in df.columns else df.columns[-1]
    lab = df[lab_col].astype(str).str.lower().str.strip()
    df["label"] = lab.isin(["bad", "phishing", "1", "malicious", "defacement", "malware"]).astype(int)
    df["url"] = df[url_col].astype(str)
    return df[["url", "label"]].dropna().drop_duplicates(subset="url")


# ------------------------------------------------------------------ Job part
RED_FLAGS = [
    ("fee", r"(registration|processing|security|training|joining|application|refundable)\s+(fee|fees|charge|charges|deposit|amount)|pay\s+(to|for)\s+(apply|joining|registration)",
     "Asks for money or a registration fee"),
    ("chat_app", r"whatsapp|telegram|wa\.me",
     "Hiring is done through WhatsApp/Telegram"),
    ("free_email", r"@(gmail|yahoo|outlook|hotmail|rediffmail)\.com",
     "Uses a free email (Gmail/Yahoo) instead of a company email"),
    ("easy_money", r"(earn|income|salary)[^.]{0,40}(per|a)\s+(day|week)|easy money|guaranteed (income|salary|job)|no experience",
     "Unrealistic claims like 'no experience, high earnings'"),
    ("urgency", r"urgent|immediate joining|limited seats|apply (now|today)|hurry",
     "Pressure to act fast (urgent / limited seats)"),
    ("wfh_data_entry", r"data entry|typing job|copy paste|form filling|work from home",
     "Common scam pattern (data entry / work from home)"),
]
_FLAG_RES = [(k, re.compile(p, re.I), msg) for k, p, msg in RED_FLAGS]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def job_flag_vector(raw_text: str) -> np.ndarray:
    return np.array([1.0 if rx.search(raw_text) else 0.0 for _, rx, _ in _FLAG_RES])


def job_indicators(raw_text: str) -> list:
    return [msg for _, rx, msg in _FLAG_RES if rx.search(raw_text)]


def load_job_dataset(path: str) -> pd.DataFrame:
    """Accepts our csv or Kaggle 'Fake Job Postings' (title, description, requirements, fraudulent)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    parts = [c for c in ["title", "company_profile", "description", "requirements", "benefits"] if c in df.columns]
    df["text"] = df[parts].fillna("").astype(str).agg(" ".join, axis=1)
    df["label"] = df["fraudulent"].astype(int)
    return df[["text", "label"]]