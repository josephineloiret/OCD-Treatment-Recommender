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
<title>OCD Screening / Clinical Decision Support</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400;1,6..72,500&display=swap" rel="stylesheet">
<style>
  :root {
    --paper:#f3f0e7; --ink:#1b1d1a; --muted:#6b6859; --faint:#9c9889;
    --rule:#d9d4c5; --accent:#0f6b54; --warn:#9a6312; --field:#fffdf7;
    --mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
    --sans:'Hanken Grotesk',system-ui,sans-serif;
    --serif:'Newsreader',Georgia,serif;
  }
  * { box-sizing:border-box; }
  html { -webkit-font-smoothing:antialiased; }
  body { margin:0; min-height:100vh; color:var(--ink); font-family:var(--sans); background:var(--paper); }
  body::before {
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.55; mix-blend-mode:multiply;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E");
  }
  .sheet { position:relative; z-index:1; max-width:760px; margin:0 auto; padding:0 26px 90px; }

  .mast { padding:48px 0 0; }
  .meta { font-family:var(--mono); font-size:.7rem; letter-spacing:.22em; text-transform:uppercase;
          color:var(--accent); display:flex; justify-content:space-between; gap:12px;
          border-bottom:1px solid var(--ink); padding-bottom:11px; }
  .meta span:last-child { color:var(--faint); }
  h1 { font-family:var(--serif); font-weight:400; font-size:3.4rem; line-height:1.03;
       letter-spacing:-.015em; margin:28px 0 16px; }
  h1 em { font-style:italic; color:var(--accent); }
  .lede { font-family:var(--serif); font-size:1.18rem; line-height:1.5; color:#3a3a31;
          max-width:55ch; margin:0 0 20px; }
  .statusline { font-family:var(--mono); font-size:.71rem; letter-spacing:.05em; color:var(--muted);
                display:flex; align-items:center; gap:9px; padding:13px 0 0; border-top:1px solid var(--rule); }
  .tick { width:8px; height:8px; border-radius:50%; background:var(--accent); }
  .tick.off { background:var(--faint); }

  .block { display:grid; grid-template-columns:46px 1fr; gap:20px; padding:30px 0;
           border-top:1px solid var(--rule); }
  .block__no { font-family:var(--mono); font-size:.8rem; color:var(--accent); padding-top:4px; }
  .field-label { font-family:var(--mono); font-size:.71rem; letter-spacing:.16em; text-transform:uppercase;
                 color:var(--muted); margin-bottom:14px; }

  textarea { width:100%; min-height:150px; resize:vertical; background:var(--field); color:var(--ink);
             border:1px solid var(--rule); border-radius:2px; padding:15px; font-family:var(--mono);
             font-size:.9rem; line-height:1.55; outline:none; transition:border-color .2s, box-shadow .2s; }
  textarea:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(15,107,84,.12); }
  textarea::placeholder { color:var(--faint); }
  .actions { display:flex; align-items:center; justify-content:space-between; margin-top:14px; gap:14px; }
  .hint { font-family:var(--mono); font-size:.67rem; color:var(--faint); }
  button { font-family:var(--mono); font-weight:500; letter-spacing:.08em; text-transform:uppercase;
           font-size:.73rem; color:var(--paper); background:var(--ink); border:none; cursor:pointer;
           padding:13px 20px; border-radius:2px; transition:background .2s, transform .12s; }
  button:hover { background:var(--accent); transform:translateY(-1px); }
  button:active { transform:translateY(0); }

  .verdict-top { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:4px; }
  .verdict-label { font-family:var(--serif); font-size:2.3rem; font-weight:400; line-height:1; }
  .verdict-label.ocd { color:var(--accent); font-style:italic; }
  .verdict-label.warn { color:var(--warn); font-style:italic; }
  .verdict-num { font-family:var(--mono); font-size:.92rem; color:var(--muted); }
  .note { color:var(--muted); font-size:.9rem; line-height:1.5; margin:8px 0 18px; }

  .bar-row { display:grid; grid-template-columns:92px 1fr 46px; align-items:center; gap:14px; margin:10px 0; }
  .bar-label { font-family:var(--mono); font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
  .bar { height:7px; background:#e7e2d3; border-radius:99px; overflow:hidden; }
  .bar__fill { display:block; height:100%; width:0; border-radius:99px; background:#b9b3a0;
               animation:grow 1.1s cubic-bezier(.2,.85,.2,1) forwards; }
  .bar__fill.ocd { background:var(--accent); }
  .bar-pct { font-family:var(--mono); font-size:.8rem; text-align:right; color:var(--ink); }
  @keyframes grow { to { width:var(--w); } }

  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:4px; }
  .chip { font-family:var(--mono); font-size:.78rem; color:var(--muted); background:var(--field);
          border:1px solid var(--rule); border-radius:2px; padding:6px 10px; }
  .chip.ocd { color:var(--accent); border-color:var(--accent); }

  .guidance { font-family:var(--sans); white-space:pre-wrap; line-height:1.65; color:#33332c; font-size:.95rem; }

  .colophon { font-family:var(--mono); color:var(--faint); font-size:.71rem; line-height:1.6;
              border-top:1px solid var(--ink); padding-top:16px; margin-top:8px; }

  .reveal { opacity:0; transform:translateY(14px); animation:rise .6s cubic-bezier(.2,.8,.2,1) forwards; }
  .d1{animation-delay:.05s}.d2{animation-delay:.12s}.d3{animation-delay:.19s}.d4{animation-delay:.26s}
  @keyframes rise { to { opacity:1; transform:none; } }
  @media (max-width:560px){ h1{font-size:2.4rem} .block{grid-template-columns:1fr; gap:10px} .block__no{padding-top:0} .bar-row{grid-template-columns:70px 1fr 40px} }
</style>
</head>
<body>
<div class="sheet">
  <header class="mast">
    <div class="meta reveal"><span>Clinical Decision Support</span><span>OCD Screening / NLP Triage</span></div>
    <h1 class="reveal d1">Screening &amp; <em>Triage</em></h1>
    <p class="lede reveal d2">Reads a patient's intake text, flags the most likely condition, and shows why. You review and decide.</p>
    <div class="statusline reveal d2">
      <span class="tick {{ '' if gpt_on else 'off' }}"></span>
      {{ 'LIVE AI GUIDANCE' if gpt_on else 'BUILT-IN GUIDANCE' }} &middot; CLINICIAN USE &middot; NOT A DIAGNOSIS
    </div>
  </header>

  <section class="block reveal d2">
    <div class="block__no">01</div>
    <div>
      <div class="field-label">Patient intake text</div>
      <form method="post">
        <textarea name="text" placeholder="Type or paste the patient's own words about how they've been feeling.">{{ text or '' }}</textarea>
        <div class="actions">
          <span class="hint">min 10 chars &middot; de-identified</span>
          <button type="submit">Run screening &rarr;</button>
        </div>
      </form>
    </div>
  </section>

  {% if ranked %}
  <section class="block reveal d1">
    <div class="block__no">02</div>
    <div>
      <div class="field-label">Assessment</div>
      {% if uncertain %}
      <div class="verdict-top">
        <span class="verdict-label warn">Uncertain</span>
        <span class="verdict-num">top: {{ top_label|upper }} @ {{ '%.0f'|format(ranked[0][1]*100) }}%</span>
      </div>
      <div class="note">Below the {{ '%.0f'|format(threshold*100) }}% threshold, so the model holds back instead of guessing.</div>
      {% else %}
      <div class="verdict-top">
        <span class="verdict-label {{ 'ocd' if top_label=='ocd' else '' }}">{{ top_label|upper }}</span>
        <span class="verdict-num">{{ '%.1f'|format(ranked[0][1]*100) }}% confidence</span>
      </div>
      <div class="note">Probability by condition:</div>
      {% endif %}
      {% for label, p in ranked %}
      <div class="bar-row">
        <span class="bar-label">{{ label }}</span>
        <span class="bar"><span class="bar__fill {{ 'ocd' if label=='ocd' else '' }}" style="--w: {{ p*100 }}%"></span></span>
        <span class="bar-pct">{{ '%.0f'|format(p*100) }}%</span>
      </div>
      {% endfor %}
    </div>
  </section>

  {% if signals %}
  <section class="block reveal d2">
    <div class="block__no">03</div>
    <div>
      <div class="field-label">Contributing language</div>
      <div class="note">Words that pushed the model toward <b>{{ top_label|upper }}</b> (TF-IDF &times; weight):</div>
      <div class="chips">
        {% for w in signals %}<span class="chip {{ 'ocd' if top_label=='ocd' else '' }}">{{ w }}</span>{% endfor %}
      </div>
    </div>
  </section>
  {% endif %}

  <section class="block reveal d3">
    <div class="block__no">04</div>
    <div>
      <div class="field-label">Treatment options for review</div>
      <div class="guidance">{{ guidance }}</div>
    </div>
  </section>
  {% endif %}

  <div class="colophon">
    Educational demo. Screening can be wrong and is not a diagnosis. A licensed clinician reviews
    every output. Do not enter identifying information.
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
                "guidance": _humanize(UNCERTAIN_GUIDANCE if uncertain else get_guidance(top_label, text)),
            })
    return render_template_string(PAGE, **ctx)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
