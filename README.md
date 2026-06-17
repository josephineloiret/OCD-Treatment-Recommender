# OCD Screening Assistant

A machine-learning web app that reads a free-text description of someone's
experience and estimates which mental-health condition it most resembles
(**OCD, depression, ADHD, or PTSD**), then surfaces **evidence-based, guideline-level
treatment information** for OCD. It is built on **real Reddit mental-health posts**
and is intended as a research/educational demo of an end-to-end ML pipeline:
data → text classifier → screening decision → guidance → deployed web app.

> This is **not** a diagnostic tool. Screening is text-similarity, can be wrong,
> and any real mental-health concern or treatment decision must be handled by a
> licensed clinician.

## Results (held-out test set)

Trained on ~20k real posts across 4 conditions (TF-IDF + Logistic Regression):

| Metric | Score |
|---|---|
| Majority-class baseline | 0.296 |
| 5-fold CV accuracy | 0.836 ± 0.006 |
| **Held-out test accuracy** | **0.839** |
| Held-out test macro-F1 | 0.824 |
| **OCD class** | precision **0.90**, recall **0.86**, F1 **0.88** |

Full report and confusion matrix are in `reports/`. The model also yields highly
interpretable signals — e.g. the top OCD terms are *intrusive, compulsions, erp,
obsessions, rituals, reassurance*.

## How to run

1. Create an environment and install dependencies (Python 3.10–3.12 recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. (Optional) Retrain from scratch:
   ```bash
   python download_data.py   # pulls real Reddit data into data/
   python train.py           # writes models/ and reports/
   ```
   A pre-trained model is already included in `models/`, so you can skip this.

3. Run the web app:
   ```bash
   python app.py
   ```
   Open http://localhost:5001 and paste a description to screen.

### Optional GPT guidance layer

By default the app shows built-in, evidence-based OCD guidance (no API key needed).
To have GPT phrase the guidance for the specific input instead, create `key.env`
with `OPENAI_API_KEY=your_key` and `pip install openai`. The app auto-detects it.

## Project files

- **`download_data.py`** – downloads the real Reddit dataset (Hugging Face, public domain).
- **`train.py`** – cleans text, trains the TF-IDF + Logistic Regression classifier,
  and writes evaluation metrics, a confusion matrix, and top predictive words to `reports/`.
- **`app.py`** – Flask web app: text in → predicted condition + probabilities →
  evidence-based (or GPT-phrased) guidance, with a clinician-approval disclaimer.
- **`models/`** – the saved trained pipeline.
- **`reports/`** – metrics, confusion matrix, and interpretability output.

## Data

Reddit Mental Health posts (`solomonk/reddit_mental_health_posts`, Hugging Face),
distributed under the Public Domain Dedication and License; reuse subject to
Reddit API terms.
