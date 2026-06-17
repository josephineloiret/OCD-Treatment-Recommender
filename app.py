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
<title>OCD Screening Console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --ink:#e7efe9; --muted:#7e9389; --faint:#54655d;
    --bg:#070b09; --panel:#0d1411; --panel-2:#101a15;
    --line:#1d2a23; --accent:#37e0a8; --accent-dim:#1f6e54;
    --warn:#f5c451; --ocd:#37e0a8;
    --mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;
    --sans:'IBM Plex Sans',system-ui,sans-serif;
    --serif:'Fraunces',Georgia,serif;
  }
  * { box-sizing:border-box; }
  html { -webkit-font-smoothing:antialiased; }
  body {
    margin:0; min-height:100vh; color:var(--ink); font-family:var(--sans);
    background:
      radial-gradient(1100px 520px at 82% -12%, rgba(55,224,168,.12), transparent 60%),
      radial-gradient(900px 500px at -10% 110%, rgba(55,224,168,.06), transparent 55%),
      var(--bg);
  }
  body::before {
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image:radial-gradient(circle at 1px 1px, rgba(231,239,233,.035) 1px, transparent 0);
    background-size:24px 24px; mask-image:linear-gradient(#000,transparent 90%);
  }
  .wrap { position:relative; z-index:1; max-width:780px; margin:0 auto; padding:54px 22px 80px; }

  .kicker { font-family:var(--mono); font-size:.72rem; letter-spacing:.28em; text-transform:uppercase;
            color:var(--accent); display:flex; align-items:center; gap:10px; margin-bottom:14px; }
  .kicker::before { content:""; width:26px; height:1px; background:var(--accent); opacity:.7; }
  h1 { font-family:var(--serif); font-weight:600; font-size:3.1rem; line-height:1.02;
       letter-spacing:-.02em; margin:0 0 14px; }
  h1 em { font-style:italic; color:var(--accent); }
  .sub { color:var(--muted); max-width:60ch; font-size:1rem; line-height:1.55; margin-bottom:8px; }
  .status { font-family:var(--mono); font-size:.74rem; color:var(--faint); margin:18px 0 30px;
            display:flex; align-items:center; gap:8px; }
  .dot { width:7px; height:7px; border-radius:50%; background:var(--accent); box-shadow:0 0 10px var(--accent); }
  .dot.off { background:var(--faint); box-shadow:none; }

  .panel { position:relative; background:linear-gradient(180deg,var(--panel-2),var(--panel));
           border:1px solid var(--line); border-radius:4px; padding:24px; margin-bottom:18px; }
  .panel__tag { position:absolute; top:-9px; left:18px; background:var(--bg); padding:0 9px;
                font-family:var(--mono); font-size:.66rem; letter-spacing:.2em; text-transform:uppercase; color:var(--faint); }

  textarea { width:100%; min-height:142px; resize:vertical; background:#060a08; color:var(--ink);
             border:1px solid var(--line); border-radius:3px; padding:14px; font-family:var(--mono);
             font-size:.92rem; line-height:1.5; outline:none; transition:border-color .2s, box-shadow .2s; }
  textarea:focus { border-color:var(--accent-dim); box-shadow:0 0 0 3px rgba(55,224,168,.10); }
  textarea::placeholder { color:var(--faint); }
  .row { display:flex; align-items:center; justify-content:space-between; margin-top:14px; gap:14px; }
  .hint { font-family:var(--mono); font-size:.7rem; color:var(--faint); }
  button { font-family:var(--mono); font-weight:600; letter-spacing:.06em; text-transform:uppercase;
           font-size:.8rem; color:#05140d; background:var(--accent); border:none; cursor:pointer;
           padding:13px 22px; border-radius:3px; transition:transform .12s, box-shadow .2s; }
  button:hover { box-shadow:0 6px 22px rgba(55,224,168,.28); transform:translateY(-1px); }
  button:active { transform:translateY(0); }

  .verdict-top { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:4px; }
  .verdict-label { font-family:var(--serif); font-size:2.1rem; font-weight:600; letter-spacing:-.01em; }
  .verdict-label.ocd { color:var(--ocd); }
  .verdict-label.warn { color:var(--warn); }
  .verdict-num { font-family:var(--mono); font-size:1.05rem; color:var(--muted); }
  .note { color:var(--muted); font-size:.9rem; line-height:1.5; margin:6px 0 18px; }

  .bar-row { display:grid; grid-template-columns:96px 1fr 48px; align-items:center; gap:14px; margin:11px 0; }
  .bar-label { font-family:var(--mono); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
  .bar { height:10px; background:#060a08; border:1px solid var(--line); border-radius:99px; overflow:hidden; }
  .bar__fill { display:block; height:100%; width:0; border-radius:99px;
               background:linear-gradient(90deg,var(--accent-dim),var(--faint));
               animation:grow 1.1s cubic-bezier(.2,.85,.2,1) forwards; }
  .bar__fill.ocd { background:linear-gradient(90deg,var(--accent-dim),var(--accent)); box-shadow:0 0 12px rgba(55,224,168,.35); }
  .bar-pct { font-family:var(--mono); font-size:.82rem; text-align:right; color:var(--ink); }
  @keyframes grow { to { width:var(--w); } }

  .chips { display:flex; flex-wrap:wrap; gap:9px; margin-top:6px; }
  .chip { font-family:var(--mono); font-size:.8rem; color:var(--muted); background:#060a08;
          border:1px solid var(--line); border-radius:3px; padding:6px 11px; }
  .chip::before { content:"› "; color:var(--accent-dim); }
  .chip.ocd { color:var(--ocd); border-color:var(--accent-dim); }
  .chip.ocd::before { color:var(--accent); }

  .guidance { font-family:var(--sans); white-space:pre-wrap; line-height:1.62; color:#cdd9d1; font-size:.95rem; }
  .disclaimer { color:var(--faint); font-size:.78rem; line-height:1.5; margin-top:26px;
                border-top:1px solid var(--line); padding-top:16px; font-family:var(--mono); }

  .reveal { opacity:0; transform:translateY(16px); animation:rise .65s cubic-bezier(.2,.8,.2,1) forwards; }
  .d1{animation-delay:.04s}.d2{animation-delay:.12s}.d3{animation-delay:.2s}.d4{animation-delay:.28s}
  @keyframes rise { to { opacity:1; transform:none; } }
  @media (max-width:560px){ h1{font-size:2.3rem} .bar-row{grid-template-columns:74px 1fr 42px} }
</style>
</head>
<body>
<div class="wrap">
  <div class="kicker reveal">NLP Diagnostic &middot; TF-IDF + Logistic Regression</div>
  <h1 class="reveal d1">OCD Screening <em>Console</em></h1>
  <p class="sub reveal d2">Describe how someone is feeling or behaving. A classifier trained on
    real Reddit mental-health posts estimates which condition the language most resembles, shows
    the words driving that call, and surfaces evidence-based guidance.</p>
  <div class="status reveal d2">
    <span class="dot {{ '' if gpt_on else 'off' }}"></span>
    {{ 'LIVE AI GUIDANCE' if gpt_on else 'BUILT-IN GUIDANCE' }} &middot; RESEARCH DEMO, NOT A DIAGNOSIS
  </div>

  <div class="panel reveal d3">
    <span class="panel__tag">Input</span>
    <form method="post">
      <textarea name="text" placeholder="e.g. I keep checking the stove over and over and can't stop intrusive thoughts that something bad will happen if I don't...">{{ text or '' }}</textarea>
      <div class="row">
        <span class="hint">free text &middot; min 10 chars</span>
        <button type="submit">Run screening &rarr;</button>
      </div>
    </form>
  </div>

  {% if ranked %}
  <div class="panel reveal d1">
    <span class="panel__tag">Prediction</span>
    {% if uncertain %}
    <div class="verdict-top">
      <span class="verdict-label warn">Uncertain</span>
      <span class="verdict-num">top: {{ top_label|upper }} @ {{ '%.0f'|format(ranked[0][1]*100) }}%</span>
    </div>
    <div class="note">Below the {{ '%.0f'|format(threshold*100) }}% confidence threshold &mdash; the model
      abstains instead of forcing a label (the screening analogue of flagging an out-of-distribution reading).</div>
    {% else %}
    <div class="verdict-top">
      <span class="verdict-label {{ 'ocd' if top_label=='ocd' else '' }}">{{ top_label|upper }}</span>
      <span class="verdict-num">{{ '%.1f'|format(ranked[0][1]*100) }}% confidence</span>
    </div>
    <div class="note">Probability across all screened conditions:</div>
    {% endif %}
    {% for label, p in ranked %}
    <div class="bar-row">
      <span class="bar-label">{{ label }}</span>
      <span class="bar"><span class="bar__fill {{ 'ocd' if label=='ocd' else '' }}" style="--w: {{ p*100 }}%"></span></span>
      <span class="bar-pct">{{ '%.0f'|format(p*100) }}%</span>
    </div>
    {% endfor %}
  </div>

  {% if signals %}
  <div class="panel reveal d2">
    <span class="panel__tag">Explanation</span>
    <div class="note">Terms that most pushed the model toward <b>{{ top_label|upper }}</b>
      (TF-IDF &times; model weight) &mdash; feature attribution:</div>
    <div class="chips">
      {% for w in signals %}<span class="chip {{ 'ocd' if top_label=='ocd' else '' }}">{{ w }}</span>{% endfor %}
    </div>
  </div>
  {% endif %}

  <div class="panel reveal d3">
    <span class="panel__tag">Guidance</span>
    <div class="guidance">{{ guidance }}</div>
  </div>
  {% endif %}

  <div class="disclaimer">
    Research/educational demo. This performs text-similarity screening, not medical diagnosis,
    and can be wrong. Any mental-health concern or treatment decision must be handled by a
    licensed clinician.
  </div>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"ranked": None, "text": "", "gpt_on": client is not None}
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
                "guidance": UNCERTAIN_GUIDANCE if uncertain else get_guidance(top_label, text),
            })
    return render_template_string(PAGE, **ctx)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
