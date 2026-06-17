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
<title>OCD Screening Assistant</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --accent:#38bdf8; --ocd:#34d399; --muted:#94a3b8; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         background:linear-gradient(160deg,#0f172a,#1e293b); color:#e2e8f0; min-height:100vh; }
  .wrap { max-width: 820px; margin: 0 auto; padding: 32px 20px 60px; }
  h1 { font-size: 1.7rem; margin-bottom: 4px; }
  .sub { color: var(--muted); margin-bottom: 24px; font-size: .95rem; }
  .card { background: var(--card); border:1px solid #334155; border-radius:14px;
          padding:22px; margin-bottom:20px; box-shadow:0 8px 30px rgba(0,0,0,.25); }
  textarea { width:100%; min-height:150px; background:#0b1220; color:#e2e8f0;
             border:1px solid #334155; border-radius:10px; padding:12px; font-size:1rem; resize:vertical; }
  button { margin-top:14px; background:var(--accent); color:#04263a; border:none;
           padding:12px 22px; font-size:1rem; font-weight:700; border-radius:10px; cursor:pointer; }
  button:hover { filter:brightness(1.08); }
  .bar-row { display:flex; align-items:center; gap:12px; margin:8px 0; }
  .bar-label { width:110px; text-transform:capitalize; font-size:.92rem; }
  .bar-track { flex:1; background:#0b1220; border-radius:8px; overflow:hidden; height:22px; border:1px solid #334155; }
  .bar-fill { height:100%; background:var(--accent); }
  .bar-fill.ocd { background:var(--ocd); }
  .bar-pct { width:54px; text-align:right; font-variant-numeric:tabular-nums; font-size:.9rem; }
  .verdict { font-size:1.15rem; font-weight:700; margin-bottom:6px; }
  .verdict.ocd { color:var(--ocd); }
  .verdict.uncertain { color:#fbbf24; }
  .note { color:var(--muted); font-size:.9rem; margin-bottom:10px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:4px; }
  .chip { background:#0b1220; border:1px solid var(--accent); color:#bfdbfe;
          padding:5px 11px; border-radius:999px; font-size:.85rem; }
  .chip.ocd { border-color:var(--ocd); color:#a7f3d0; }
  .guidance { white-space:pre-wrap; line-height:1.5; color:#cbd5e1; }
  .disclaimer { color:var(--muted); font-size:.82rem; margin-top:18px; border-top:1px solid #334155; padding-top:12px; }
  .pill { display:inline-block; font-size:.72rem; background:#334155; color:#cbd5e1;
          padding:3px 9px; border-radius:999px; margin-left:8px; vertical-align:middle; }
</style>
</head>
<body>
<div class="wrap">
  <h1>OCD Screening Assistant
    <span class="pill">{{ 'live AI guidance' if gpt_on else 'built-in guidance' }}</span>
  </h1>
  <div class="sub">Type how someone is feeling/behaving. A model trained on real
    Reddit mental-health posts estimates which condition the text most resembles,
    then shows evidence-based guidance. Research demo &mdash; not a diagnosis.</div>

  <div class="card">
    <form method="post">
      <textarea name="text" placeholder="e.g. I keep checking the stove over and over and can't stop intrusive thoughts that something bad will happen if I don't...">{{ text or '' }}</textarea>
      <button type="submit">Run screening</button>
    </form>
  </div>

  {% if ranked %}
  <div class="card">
    {% if uncertain %}
    <div class="verdict uncertain">Uncertain &middot; not enough signal</div>
    <div class="note">Top guess is {{ top_label|upper }} at only {{ '%.0f'|format(ranked[0][1]*100) }}%,
      below the {{ '%.0f'|format(threshold*100) }}% confidence threshold &mdash; so the model abstains
      instead of forcing a label.</div>
    {% else %}
    <div class="verdict {{ 'ocd' if top_label=='ocd' else '' }}">
      Top match: {{ top_label|upper }} &middot; {{ '%.0f'|format(ranked[0][1]*100) }}% confidence
    </div>
    {% endif %}
    {% for label, p in ranked %}
    <div class="bar-row">
      <div class="bar-label">{{ label }}</div>
      <div class="bar-track"><div class="bar-fill {{ 'ocd' if label=='ocd' else '' }}" style="width: {{ p*100 }}%"></div></div>
      <div class="bar-pct">{{ '%.0f'|format(p*100) }}%</div>
    </div>
    {% endfor %}
  </div>

  {% if signals %}
  <div class="card">
    <div class="verdict">Why this prediction</div>
    <div class="note">Words in the text that most pushed the model toward {{ top_label|upper }}
      (TF-IDF &times; model weight):</div>
    <div class="chips">
      {% for w in signals %}<span class="chip {{ 'ocd' if top_label=='ocd' else '' }}">{{ w }}</span>{% endfor %}
    </div>
  </div>
  {% endif %}

  <div class="card">
    <div class="verdict">Guidance</div>
    <div class="guidance">{{ guidance }}</div>
  </div>
  {% endif %}

  <div class="disclaimer">
    This tool is for research and educational purposes only. It performs text
    similarity screening, not medical diagnosis, and can be wrong. Any mental
    health concern or treatment decision must be handled by a licensed clinician.
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
