"""
OCD Screening Assistant - web app.

Two honest stages:
  1. ML screening: a TF-IDF + Logistic Regression model (trained on real Reddit
     mental-health posts) reads a free-text description and estimates which
     condition it most resembles, with calibrated-ish probabilities.
  2. Guidance: evidence-based, guideline-level information about OCD treatment
     (SSRIs as first-line pharmacotherapy, ERP therapy). If an OpenAI API key is
     present it is phrased by GPT for the specific text; otherwise a built-in
     evidence-based summary is shown. Either way it is clearly decision-support
     that must be reviewed by a licensed clinician.

This is a research/educational demo and is NOT a diagnostic tool.
"""

import os

import joblib
import numpy as np
from flask import Flask, render_template_string, request

MODEL_DIR = "models"
pipeline = joblib.load(os.path.join(MODEL_DIR, "pipeline.joblib"))
labels = joblib.load(os.path.join(MODEL_DIR, "labels.joblib"))

# If the model's top probability is below this, we report "uncertain" instead of
# forcing a guess - the screening analogue of novelty/anomaly detection.
CONFIDENCE_THRESHOLD = 0.45

# Optional LLM guidance layer ------------------------------------------------
# The provider is configurable so the live demo can run on a FREE,
# OpenAI-compatible API (e.g. Groq or Google Gemini) instead of paid OpenAI.
# Set these env vars (e.g. in the Render dashboard):
#   LLM_API_KEY   - your key (falls back to OPENAI_API_KEY)
#   LLM_BASE_URL  - e.g. https://api.groq.com/openai/v1   (omit for OpenAI)
#   LLM_MODEL     - e.g. llama-3.3-70b-versatile (Groq) or gpt-4o-mini (OpenAI)
client = None
try:
    from dotenv import load_dotenv
    load_dotenv("key.env")
except Exception:
    pass

LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")  # None -> default OpenAI endpoint
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

if LLM_API_KEY:
    try:
        from openai import OpenAI
        kwargs = {"api_key": LLM_API_KEY}
        if LLM_BASE_URL:
            kwargs["base_url"] = LLM_BASE_URL
        client = OpenAI(**kwargs)
    except Exception:
        client = None

CONDITION_FULL = {
    "ocd": "OCD (Obsessive-Compulsive Disorder)",
    "depression": "Depression (Major Depressive Disorder)",
    "adhd": "ADHD (Attention-Deficit/Hyperactivity Disorder)",
    "ptsd": "PTSD (Post-Traumatic Stress Disorder)",
}

GUIDANCE_INSTRUCTIONS = (
    "You are a clinical information assistant for a research/education demo. You receive a "
    "free-text description and the mental-health condition that a screening model predicted "
    "the text most resembles. Write a brief, neutral, evidence-based overview of how THAT "
    "specific condition is typically treated, in three clearly labelled sections:\n"
    "1) First-line medications - the guideline first-line medication class(es) with a few "
    "representative example drugs.\n"
    "2) Other medication options - common second-line, adjunct/augmentation, or alternative "
    "medications.\n"
    "3) Therapies - the main evidence-based psychotherapies, named specifically (e.g. ERP, "
    "CBT, ACT, CPT, EMDR, prolonged exposure, behavioral activation, IPT, behavioral parent "
    "training), as appropriate to the condition.\n\n"
    "Rules: name medication CLASSES and representative examples only - do NOT give doses or "
    "tell the person what to take. Be accurate to current clinical guidelines for the "
    "predicted condition. Briefly tie your reasoning to cues in the description when relevant. "
    "Write plainly and do NOT use em dashes or en dashes; use short sentences or commas. "
    "Keep it concise (about 9-13 lines total). Finish with one sentence stating this is "
    "general educational information, not a diagnosis, and must be reviewed by a licensed "
    "clinician."
)

# Built-in, guideline-level guidance used only as a fallback when the OpenAI
# library/key is unavailable (the live GPT generation is the primary path).
CONDITION_GUIDANCE = {
    "ocd": (
        "OCD - evidence-based treatment overview\n\n"
        "First-line medications: SSRIs (e.g. fluoxetine, sertraline, fluvoxamine, "
        "escitalopram) - often at higher doses and with longer trials than in depression.\n\n"
        "Other medication options: clomipramine (a tricyclic) for SSRI non-responders; in "
        "treatment-resistant cases, augmentation with a low-dose atypical antipsychotic "
        "(e.g. aripiprazole, risperidone).\n\n"
        "Therapies: Exposure and Response Prevention (ERP) - the gold-standard CBT for OCD; "
        "broader Cognitive Behavioral Therapy; Acceptance and Commitment Therapy (ACT) as an "
        "alternative or adjunct."
    ),
    "depression": (
        "Depression - evidence-based treatment overview\n\n"
        "First-line medications: SSRIs (e.g. sertraline, escitalopram) and SNRIs "
        "(e.g. venlafaxine, duloxetine).\n\n"
        "Other medication options: atypical antidepressants (bupropion, mirtazapine); older "
        "agents such as tricyclics and MAOIs; augmentation strategies; esketamine or ECT for "
        "treatment-resistant depression.\n\n"
        "Therapies: Cognitive Behavioral Therapy (CBT), Behavioral Activation, and "
        "Interpersonal Therapy (IPT); mindfulness-based cognitive therapy for relapse "
        "prevention."
    ),
    "adhd": (
        "ADHD - evidence-based treatment overview\n\n"
        "First-line medications: stimulants - methylphenidate-class (e.g. Ritalin, Concerta) "
        "and amphetamine-class (e.g. Adderall, Vyvanse).\n\n"
        "Other medication options: non-stimulants such as atomoxetine, guanfacine, and "
        "clonidine; bupropion is sometimes used off-label.\n\n"
        "Therapies: behavioral therapy and CBT adapted for adult ADHD; skills/coaching for "
        "organization and time management; psychoeducation; behavioral parent training for "
        "children."
    ),
    "ptsd": (
        "PTSD - evidence-based treatment overview\n\n"
        "First-line medications: SSRIs (sertraline and paroxetine are FDA-approved for PTSD) "
        "and the SNRI venlafaxine.\n\n"
        "Other medication options: prazosin for trauma-related nightmares; atypical "
        "antipsychotics are sometimes used as adjuncts in specific cases.\n\n"
        "Therapies (often preferred over medication): trauma-focused CBT, Prolonged Exposure "
        "(PE), Cognitive Processing Therapy (CPT), and Eye Movement Desensitization and "
        "Reprocessing (EMDR)."
    ),
}

GENERIC_GUIDANCE = (
    "Evidence-based care usually combines medication and psychotherapy, tailored to the "
    "individual. Common medication classes include SSRIs and SNRIs with condition-specific "
    "alternatives, alongside therapies such as CBT."
)

DISCLAIMER_LINE = (
    "\n\nThis is general educational information, not a diagnosis, and must be reviewed and "
    "approved by a licensed clinician before any treatment decision."
)

UNCERTAIN_GUIDANCE = (
    "The model is not confident enough about this text to map it to a single condition "
    "(its top probability is below the decision threshold). Rather than force a guess, it "
    "abstains - the same principle used in fault detection, where a low-confidence or "
    "out-of-distribution reading should be flagged for review instead of acted on. Try a "
    "longer or more specific description." + DISCLAIMER_LINE
)


def _humanize(s: str) -> str:
    """Strip em/en dashes and the AI-tell ' - ' spacing so copy reads naturally."""
    return (
        s.replace(" \u2014 ", ", ").replace("\u2014", ", ")
        .replace(" \u2013 ", ", ").replace("\u2013", ", ")
        .replace(" - ", ", ")
    )


def _static_guidance(top_label: str) -> str:
    return CONDITION_GUIDANCE.get(top_label, GENERIC_GUIDANCE) + DISCLAIMER_LINE


def get_guidance(top_label: str, text: str) -> str:
    if client is None:
        return _static_guidance(top_label)
    condition = CONDITION_FULL.get(top_label, top_label)
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": GUIDANCE_INSTRUCTIONS},
                {"role": "user", "content": f"Predicted condition: {condition}\n\nDescription:\n{text}"},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return _static_guidance(top_label)


def screen(text: str):
    proba = pipeline.predict_proba([text])[0]
    classes = list(pipeline.named_steps["clf"].classes_)
    ranked = sorted(zip(classes, proba), key=lambda t: t[1], reverse=True)
    return ranked


def explain(text: str, top_label: str, k: int = 8):
    """Return the input words that most pushed the model toward `top_label`.

    For a linear model over TF-IDF features, each word's contribution to a class
    score is simply tfidf_value * class_coefficient - so we can read off exactly
    which terms drove the prediction (the text analogue of feature attribution on
    sensor signals).
    """
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    classes = list(clf.classes_)
    X = vec.transform([text]).tocoo()
    coef = clf.coef_
    if coef.shape[0] == 1:  # binary case: row applies to the positive class
        row = coef[0] if top_label == classes[1] else -coef[0]
    else:
        row = coef[classes.index(top_label)]
    feats = vec.get_feature_names_out()
    contribs = [(feats[i], v * row[i]) for i, v in zip(X.col, X.data)]
    contribs.sort(key=lambda t: t[1], reverse=True)
    return [w for w, c in contribs if c > 0][:k]


app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCD Screening Tool</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&family=Spline+Sans+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#eceff3; --surface:#ffffff; --ink:#13171c; --muted:#5c6571; --faint:#929aa6;
    --line:#dce1e7; --track:#e7eaef; --rest:#b9c1cc;
    --accent:#1f5cff; --accent-ink:#0c3bd6; --warn:#c2410c; --live:#22c55e;
    --topbar:#13171c;
    --mono:'Spline Sans Mono',ui-monospace,Menlo,monospace;
    --sans:'Archivo',system-ui,sans-serif;
  }
  * { box-sizing:border-box; }
  html { -webkit-font-smoothing:antialiased; }
  body { margin:0; min-height:100vh; color:var(--ink); font-family:var(--sans); background:var(--bg); }

  .topbar { position:sticky; top:0; z-index:5; background:var(--topbar); color:#fff; }
  .topbar__in { max-width:880px; margin:0 auto; padding:0 24px; height:58px;
                display:flex; align-items:center; justify-content:space-between; gap:16px; }
  .brand { display:flex; align-items:center; gap:11px; }
  .brand__bar { width:3px; height:22px; background:var(--accent); border-radius:2px; }
  .brand__name { font-weight:700; font-size:1.05rem; letter-spacing:-.01em; }
  .brand__tag { font-family:var(--mono); font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; color:#7f8895; }
  .topmeta { display:flex; align-items:center; gap:20px; }
  .topmeta span { font-family:var(--mono); font-size:.64rem; letter-spacing:.09em; text-transform:uppercase; color:#7f8895; position:relative; }
  .topmeta span + span::before { content:""; position:absolute; left:-10px; top:50%; transform:translateY(-50%);
                                 width:1px; height:12px; background:#30363f; }
  @media (max-width:600px){ .brand__tag, .topmeta { display:none; } }

  .wrap { max-width:880px; margin:0 auto; padding:28px 24px 72px; }
  .intro { color:var(--muted); font-size:.98rem; line-height:1.5; max-width:64ch; margin:4px 0 24px; }

  .card { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:22px 22px 24px;
          margin-bottom:16px; box-shadow:0 1px 2px rgba(16,23,33,.05); }
  .card__head { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; }
  .card__title { font-weight:700; font-size:1.04rem; letter-spacing:-.01em; }
  .tag { font-family:var(--mono); font-size:.6rem; letter-spacing:.16em; text-transform:uppercase; color:var(--faint); }

  textarea { width:100%; min-height:140px; resize:vertical; background:#fbfcfe; color:var(--ink);
             border:1px solid var(--line); border-radius:9px; padding:14px; font-family:var(--sans);
             font-size:.97rem; line-height:1.5; outline:none; transition:border-color .15s, box-shadow .15s; }
  textarea:focus { border-color:var(--accent); box-shadow:0 0 0 4px rgba(31,92,255,.13); }
  textarea::placeholder { color:#a3acb8; }
  .actions { display:flex; align-items:center; justify-content:space-between; gap:14px; margin-top:14px; }
  .hint { font-family:var(--mono); font-size:.68rem; color:var(--faint); }
  button { font-family:var(--sans); font-weight:600; font-size:.92rem; color:#fff; background:var(--accent);
           border:none; border-radius:9px; padding:11px 22px; cursor:pointer; transition:background .15s, transform .1s; }
  button:hover { background:var(--accent-ink); }
  button:active { transform:translateY(1px); }

  .verdict { display:flex; align-items:flex-end; justify-content:space-between; gap:16px; flex-wrap:wrap; }
  .verdict__name { font-weight:800; font-size:2.15rem; letter-spacing:-.02em; line-height:1; }
  .verdict__name.warn { color:var(--warn); }
  .verdict__meta { font-family:var(--mono); font-size:.76rem; color:var(--muted); text-align:right; line-height:1.5; }
  .verdict__pct { font-family:var(--sans); font-weight:700; font-size:1.5rem; color:var(--ink); letter-spacing:-.01em; }
  .note { color:var(--muted); font-size:.9rem; line-height:1.5; margin:12px 0 0; }

  .gauge-wrap { position:relative; margin:24px 0 26px; }
  .gauge { height:12px; background:var(--track); border-radius:7px; overflow:hidden; }
  .gauge__fill { height:100%; width:0; background:var(--accent); border-radius:7px;
                 animation:grow 1s cubic-bezier(.2,.8,.2,1) forwards; }
  .gauge__fill.warn { background:var(--warn); }
  .threshold { position:absolute; top:-3px; height:18px; width:2px; background:var(--ink); opacity:.5; }
  .threshold__lab { position:absolute; top:-17px; transform:translateX(-50%); font-family:var(--mono);
                    font-size:.57rem; letter-spacing:.04em; color:var(--muted); white-space:nowrap; }
  .ticks { display:flex; justify-content:space-between; margin-top:7px; font-family:var(--mono); font-size:.57rem; color:var(--faint); }

  .subtle { font-family:var(--mono); font-size:.62rem; letter-spacing:.14em; text-transform:uppercase; color:var(--faint); margin:0 0 10px; }
  .bd-row { display:grid; grid-template-columns:92px 1fr 44px; align-items:center; gap:12px; margin:9px 0; }
  .bd-label { font-family:var(--mono); font-size:.73rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
  .bd-track { height:6px; background:var(--track); border-radius:6px; overflow:hidden; }
  .bd-fill { display:block; height:100%; width:0; background:var(--rest); border-radius:6px;
             animation:grow 1s cubic-bezier(.2,.8,.2,1) forwards; }
  .bd-fill.top { background:var(--accent); }
  .bd-pct { font-family:var(--mono); font-size:.74rem; text-align:right; color:var(--ink); }
  @keyframes grow { to { width:var(--w); } }

  .chips { display:flex; flex-wrap:wrap; gap:8px; }
  .chip { font-family:var(--mono); font-size:.78rem; color:var(--ink); background:#f2f6fd;
          border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:5px; padding:6px 11px; }

  .guidance { white-space:pre-wrap; line-height:1.6; color:#2b323b; font-size:.95rem; }

  .foot { font-family:var(--mono); color:var(--faint); font-size:.67rem; line-height:1.6; margin:6px 2px 0; }
  @media (max-width:560px){ .verdict__name{font-size:1.8rem} .bd-row{grid-template-columns:68px 1fr 40px} }
</style>
</head>
<body>
<header class="topbar">
  <div class="topbar__in">
    <div class="brand">
      <span class="brand__bar"></span>
      <span class="brand__name">OCD Screening Tool</span>
      <span class="brand__tag">/ clinical nlp screening</span>
    </div>
    <nav class="topmeta">
      <span>TF-IDF + LogReg</span>
      <span>4 conditions</span>
      <span>threshold {{ '%.0f'|format(threshold*100) }}%</span>
    </nav>
  </div>
</header>

<main class="wrap">
  <p class="intro">Screens a patient's intake text, estimates the most likely condition, shows the words
    behind it, and abstains when unsure. You review and decide.</p>

  <section class="card">
    <div class="card__head"><div class="card__title">Patient intake</div><div class="tag">Input</div></div>
    <form method="post">
      <textarea name="text" placeholder="Type or paste the patient's own words about how they've been feeling.">{{ text or '' }}</textarea>
      <div class="actions">
        <span class="hint">min 10 chars / de-identified</span>
        <button type="submit">Run screening</button>
      </div>
    </form>
  </section>

  {% if ranked %}
  <section class="card">
    <div class="card__head"><div class="card__title">Result</div><div class="tag">Assessment</div></div>
    {% if uncertain %}
    <div class="verdict">
      <div class="verdict__name warn">Uncertain</div>
      <div class="verdict__meta">top {{ top_label|upper }}<br>{{ '%.0f'|format(ranked[0][1]*100) }}%</div>
    </div>
    <div class="note">Top score is below the {{ '%.0f'|format(threshold*100) }}% threshold, so the tool holds back instead of guessing.</div>
    {% else %}
    <div class="verdict">
      <div class="verdict__name">{{ top_label|upper }}</div>
      <div class="verdict__meta"><span class="verdict__pct">{{ '%.0f'|format(ranked[0][1]*100) }}%</span><br>confidence</div>
    </div>
    {% endif %}

    <div class="gauge-wrap">
      <div class="gauge"><span class="gauge__fill {{ 'warn' if uncertain else '' }}" style="--w: {{ ranked[0][1]*100 }}%"></span></div>
      <div class="threshold" style="left: {{ threshold*100 }}%"></div>
      <div class="threshold__lab" style="left: {{ threshold*100 }}%">threshold {{ '%.0f'|format(threshold*100) }}%</div>
      <div class="ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
    </div>

    <div class="subtle">Probability by condition</div>
    {% for label, p in ranked %}
    <div class="bd-row">
      <span class="bd-label">{{ label }}</span>
      <span class="bd-track"><span class="bd-fill {{ 'top' if loop.first and not uncertain else '' }}" style="--w: {{ p*100 }}%"></span></span>
      <span class="bd-pct">{{ '%.0f'|format(p*100) }}%</span>
    </div>
    {% endfor %}
  </section>

  {% if signals %}
  <section class="card">
    <div class="card__head"><div class="card__title">Why this result</div><div class="tag">Signals</div></div>
    <div class="note">Words that pushed the model toward {{ top_label|upper }} (TF-IDF &times; weight).</div>
    <div class="chips" style="margin-top:12px">
      {% for w in signals %}<span class="chip">{{ w }}</span>{% endfor %}
    </div>
  </section>
  {% endif %}

  <section class="card">
    <div class="card__head"><div class="card__title">Treatment options</div><div class="tag">For review</div></div>
    <div class="guidance">{{ guidance }}</div>
  </section>
  {% endif %}

  <p class="foot">Educational demo. Screening can be wrong and is not a diagnosis. A licensed clinician
    reviews every output. Do not enter identifying information.</p>
</main>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"ranked": None, "text": "", "gpt_on": client is not None, "threshold": CONFIDENCE_THRESHOLD}
    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        ctx["text"] = text
        if len(text) >= 10:
            ranked = screen(text)
            top_label, top_p = ranked[0]
            uncertain = top_p < CONFIDENCE_THRESHOLD
            ctx.update({
                "ranked": ranked,
                "top_label": top_label,
                "uncertain": uncertain,
                "threshold": CONFIDENCE_THRESHOLD,
                "signals": [] if uncertain else explain(text, top_label),
                "guidance": _humanize(UNCERTAIN_GUIDANCE if uncertain else get_guidance(top_label, text)),
            })
    return render_template_string(PAGE, **ctx)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
