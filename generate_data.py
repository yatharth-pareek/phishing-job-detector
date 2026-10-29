"""Creates small demo datasets so the project runs without downloading anything.
For real accuracy, replace data/urls.csv and data/jobs.csv with the Kaggle files (see README)."""
import random
import string

import pandas as pd

random.seed(42)
BRANDS = ["google", "amazon", "paypal", "facebook", "microsoft", "apple", "netflix", "flipkart",
          "paytm", "sbi", "hdfcbank", "icici", "instagram", "linkedin", "naukri", "wikipedia",
          "github", "youtube", "irctc", "swiggy"]
PATHS = ["", "about", "products/item", "help/contact", "blog/post", "news/today", "careers",
         "search", "docs/guide", "login", "account/settings"]
OK_TLDS = ["com", "in", "org", "net", "co.in"]
BAD_TLDS = ["xyz", "top", "tk", "ml", "click", "work", "info", "gq"]


def rs(n=6):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def legit_url():
    sub = random.choice(["www.", "", "www.", "mail.", "support.", "blog."])
    q = random.choice(["", "", "?id=" + str(random.randint(1, 999)), "?q=" + rs(5)])
    scheme = "https" if random.random() < 0.9 else "http"
    return f"{scheme}://{sub}{random.choice(BRANDS)}.{random.choice(OK_TLDS)}/{random.choice(PATHS)}{q}"


def phish_url():
    b, w = random.choice(BRANDS), random.choice(["login", "verify", "secure", "update", "account", "confirm", "signin"])
    k = random.randint(0, 5)
    if k == 0:
        ip = ".".join(str(random.randint(1, 255)) for _ in range(4))
        return f"http://{ip}/{w}/{b}.php?id={rs(8)}"
    if k == 1:
        return f"http://{w}-{b}-{rs(4)}.{random.choice(BAD_TLDS)}/{w}?session={rs(10)}"
    if k == 2:
        return f"http://{b}.com.{w}-{rs(5)}.{random.choice(BAD_TLDS)}/{w}/index.html"
    if k == 3:
        return f"http://{random.choice(['bit.ly', 'tinyurl.com', 'cutt.ly'])}/{rs(7)}"
    if k == 4:
        return f"http://{b}.com@{rs(8)}.{random.choice(BAD_TLDS)}/{w}"
    return f"https://{b}-{w}.{rs(6)}.{random.choice(BAD_TLDS)}/{w}/{b}/confirm?acc={rs(6)}"


# ---------------------------------------------------------------- jobs
TITLES = ["Software Developer", "Data Analyst", "Accountant", "Marketing Executive", "HR Associate",
          "Customer Support Executive", "Python Developer", "Content Writer", "Sales Manager", "Graphic Designer"]
CITIES = ["Bengaluru", "Pune", "Hyderabad", "Delhi", "Mumbai", "Indore", "Chennai"]
REAL_DESC = [
    "We are looking for a {t} to join our team in {c}. You will work with cross-functional teams to deliver quality projects.",
    "Our company is hiring a {t} at our {c} office. Responsibilities include planning, reporting and collaborating with stakeholders.",
    "Join us as a {t}. You will own tasks end to end, support the team and help improve our processes.",
]
REAL_REQ = [
    "Bachelor's degree in a relevant field. 2+ years of experience. Good communication skills.",
    "Strong knowledge of the domain, ability to work in a team and meet deadlines. Graduate in any stream.",
    "Proficiency in relevant tools, analytical thinking and attention to detail. Interview process has 3 rounds.",
]
REAL_NOISE = ["Apply now through our careers page.", "Immediate joining preferred.", "Hybrid work from home option available.", ""]
FAKE_BASE = ["Hiring {t}. Work from home, simple online work.", "Urgent requirement for {t} in {c}.", "Data entry and typing job, anyone can do."]
FAKE_PIECES = [
    "Earn Rs {n} per week from home. No experience needed.",
    "Pay a registration fee of Rs {f} to confirm your seat.",
    "Contact us on WhatsApp {p} for immediate joining.",
    "Send your resume to hr.{r}@gmail.com.",
    "Limited seats, apply now, hurry up!",
    "Refundable security deposit required before training.",
    "Guaranteed income, easy money, no interview.",
]


def real_job():
    t, c = random.choice(TITLES), random.choice(CITIES)
    return {"title": t, "description": random.choice(REAL_DESC).format(t=t, c=c) + " " + random.choice(REAL_NOISE),
            "requirements": random.choice(REAL_REQ), "fraudulent": 0}


def fake_job():
    t, c = random.choice(TITLES), random.choice(CITIES)
    pieces = random.sample(FAKE_PIECES, random.randint(1, 4))
    txt = " ".join(p.format(n=random.choice([20000, 40000, 60000]), f=random.choice([500, 1500, 2500]),
                            p="9" + "".join(random.choices(string.digits, k=9)), r=rs(5)) for p in pieces)
    return {"title": t + random.choice([" - Work From Home", "", " (Urgent)"]),
            "description": random.choice(FAKE_BASE).format(t=t, c=c) + " " + txt,
            "requirements": random.choice(["Anyone can apply. 10th pass.", "No experience needed.", random.choice(REAL_REQ)]),
            "fraudulent": 1}


if __name__ == "__main__":
    urls = [(legit_url(), 0) for _ in range(1500)] + [(phish_url(), 1) for _ in range(1500)]
    random.shuffle(urls)
    pd.DataFrame(urls, columns=["url", "label"]).to_csv("data/urls.csv", index=False)
    jobs = [real_job() for _ in range(1700)] + [fake_job() for _ in range(300)]
    random.shuffle(jobs)
    pd.DataFrame(jobs).to_csv("data/jobs.csv", index=False)
    print("Demo data created: data/urls.csv, data/jobs.csv")
