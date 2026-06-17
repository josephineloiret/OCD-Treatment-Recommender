"""
Train an OCD screening text classifier on real Reddit mental-health posts.

Each CSV in data/ is one condition (named <condition>.csv, e.g. ocd.csv).
We combine each post's title + body and train a TF-IDF + Logistic Regression
model to predict which condition a post most resembles. The model is the
"screening" stage; treatment guidance is handled separately in the web app.

Outputs (in models/ and reports/):
  - models/pipeline.joblib      the trained text classifier
  - models/labels.joblib        ordered class labels
  - reports/metrics.txt         accuracy / cross-val / per-class report
  - reports/confusion_matrix.png
  - reports/top_words.txt       most predictive words per condition
"""

import csv
import glob
import os
import re

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

DATA_DIR = "data"
MODEL_DIR = "models"
REPORT_DIR = "reports"
MAX_PER_CLASS = 6000  # cap per class so no condition dominates / keeps training fast
MIN_PER_CLASS = 500   # skip a condition if it has fewer usable posts than this
RANDOM_STATE = 42

REMOVED = {"[removed]", "[deleted]", "", "nan"}

csv.field_size_limit(10_000_000)


def read_csv_safe(path):
    """Read a CSV that may be truncated mid-record (from a range download).

    Falls back to the stdlib csv reader, collecting every complete row and
    silently dropping a partial final record instead of erroring out.
    """
    try:
        return pd.read_csv(path)
    except Exception:
        rows = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                for row in reader:
                    rows.append(row)
            except csv.Error:
                pass  # truncated tail of a range-downloaded file
        if len(rows) < 2:
            return pd.DataFrame()
        header, body = rows[0], rows[1:]
        width = len(header)
        body = [r for r in body if len(r) == width]
        return pd.DataFrame(body, columns=header)


def clean_text(title, body):
    parts = []
    for v in (title, body):
        if isinstance(v, str) and v.strip().lower() not in REMOVED:
            parts.append(v.strip())
    text = " ".join(parts)
    text = re.sub(r"http\S+", " ", text)        # strip urls
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data():
    frames = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}/. Download the data first.")
    for path in files:
        label = os.path.splitext(os.path.basename(path))[0].lower()
        df = read_csv_safe(path)
        if df.empty:
            print(f"  {label:<12} skipped (could not read)")
            continue
        title = df["title"] if "title" in df else ""
        body = df["body"] if "body" in df else ""
        texts = [clean_text(t, b) for t, b in zip(
            title if isinstance(title, pd.Series) else [title] * len(df),
            body if isinstance(body, pd.Series) else [body] * len(df),
        )]
        sub = pd.DataFrame({"text": texts, "label": label})
        sub = sub[sub["text"].str.len() >= 20]          # drop near-empty posts
        sub = sub.drop_duplicates(subset="text")
        if len(sub) < MIN_PER_CLASS:
            print(f"  {label:<12} {len(sub):>6} usable posts  -> SKIPPED (< {MIN_PER_CLASS})")
            continue
        if len(sub) > MAX_PER_CLASS:
            sub = sub.sample(MAX_PER_CLASS, random_state=RANDOM_STATE)
        print(f"  {label:<12} {len(sub):>6} usable posts  (from {path})")
        frames.append(sub)
    data = pd.concat(frames, ignore_index=True)
    return data


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading data...")
    data = load_data()
    print(f"\nTotal posts: {len(data)} across {data['label'].nunique()} conditions")
    print(data["label"].value_counts().to_string())

    X = data["text"].values
    y = data["label"].values
    labels = sorted(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_features=30000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=3.0,
            class_weight="balanced",
        )),
    ])

    print("\nCross-validating (5-fold) on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")

    print("Fitting final model...")
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    test_acc = accuracy_score(y_test, pred)
    test_f1 = f1_score(y_test, pred, average="macro")
    baseline = pd.Series(y_train).value_counts(normalize=True).max()
    report = classification_report(y_test, pred, digits=3)
    cm = confusion_matrix(y_test, pred, labels=labels)

    # ---- write metrics report ----
    lines = []
    lines.append("OCD SCREENING CLASSIFIER - EVALUATION REPORT")
    lines.append("=" * 52)
    lines.append(f"Conditions ({len(labels)}): {', '.join(labels)}")
    lines.append(f"Train posts: {len(X_train)}   Test posts: {len(X_test)}")
    lines.append("")
    lines.append(f"Majority-class baseline accuracy : {baseline:.3f}")
    lines.append(f"5-fold CV accuracy  : {cv_acc.mean():.3f} +/- {cv_acc.std():.3f}")
    lines.append(f"5-fold CV macro-F1  : {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")
    lines.append(f"Held-out test accuracy : {test_acc:.3f}")
    lines.append(f"Held-out test macro-F1 : {test_f1:.3f}")
    lines.append("")
    lines.append("Per-class report (held-out test set):")
    lines.append(report)
    report_text = "\n".join(lines)
    with open(os.path.join(REPORT_DIR, "metrics.txt"), "w") as f:
        f.write(report_text)
    print("\n" + report_text)

    # ---- confusion matrix figure ----
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"Confusion Matrix (test acc = {test_acc:.1%})")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150)

    # ---- top predictive words per class ----
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = np.array(vec.get_feature_names_out())
    with open(os.path.join(REPORT_DIR, "top_words.txt"), "w") as f:
        f.write("Most predictive terms per condition\n")
        f.write("=" * 40 + "\n")
        classes = clf.classes_
        coefs = clf.coef_
        if len(classes) == 2:
            coefs = np.vstack([-coefs[0], coefs[0]])
        for i, cls in enumerate(classes):
            top = feature_names[np.argsort(coefs[i])[-20:][::-1]]
            f.write(f"\n{cls}:\n  " + ", ".join(top) + "\n")
    print(f"\nTop predictive words written to {REPORT_DIR}/top_words.txt")

    joblib.dump(pipeline, os.path.join(MODEL_DIR, "pipeline.joblib"))
    joblib.dump(labels, os.path.join(MODEL_DIR, "labels.joblib"))
    print(f"\nSaved model to {MODEL_DIR}/pipeline.joblib")


if __name__ == "__main__":
    main()
