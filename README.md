# Adaptive AI-Driven Post-Quantum File Encryption Framework

Academic prototype. Upload a file → an AI agent assesses its risk → the
framework picks a post-quantum security profile and encrypts it with
hybrid ML-KEM + AES-256-GCM.

## What changed in this version

The risk-assessment step used to be a **RandomForest classifier** trained
on a synthetic, hand-written policy formula (a rule-based "rigets"
scoring system in disguise). It's now a **Groq-hosted LLM risk agent**:

- The file's text and automated evidence (PII regex hits, confidential
  keyword hits, email domains, etc.) are sent to a Groq model.
- The model reasons about the content and returns a structured JSON
  verdict: risk class, confidence, per-factor scores, key concerns, and a
  short written justification — shown in the UI.
- The old RandomForest model is **kept only as an offline fallback**: if
  `GROQ_API_KEY` isn't set, the `groq` package isn't installed, or the API
  call fails, the pipeline automatically falls back to it and clearly
  labels the result as a fallback (never silently swaps a real Groq
  verdict for the heuristic one).

## 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configure your Groq API key

Get a key from https://console.groq.com/keys, then either:

**Option A — .env file (recommended):**
```
# .env, next to main.py
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile   # optional, this is the default
```

**Option B — environment variable:**
```bash
export GROQ_API_KEY=gsk_your_key_here      # macOS/Linux
setx GROQ_API_KEY "gsk_your_key_here"      # Windows
```

Other good Groq models to try (set `GROQ_MODEL`): `openai/gpt-oss-120b`
(stronger reasoning, still fast) or `llama-3.1-8b-instant` (cheapest/fastest).

## 3. Train the offline fallback model (optional but recommended)

```bash
python main.py train
```

This is only used when Groq is unavailable — but train it once so the
app never fully blocks if your key/network has an issue.

## 4. Run the web UI

```bash
python main.py serve --port 5000
```

Open http://localhost:5000. Upload a file, pick an engine (Auto / Groq
only / Heuristic only), and click **Analyze & Encrypt**. You'll see the
risk class, the AI's confidence, its written reasoning, key concerns, the
selected security profile, and encryption metrics — then a download link
for the `.pqc` file. Use the decrypt panel to recover and verify it.

## 5. CLI usage

```bash
# Risk assessment only
python main.py analyze --file mydoc.pdf --engine auto

# Full pipeline: analyze + encrypt
python main.py encrypt --file mydoc.pdf --engine groq

# Decrypt
python main.py decrypt --file encrypted/mydoc.pdf.pqc

# Benchmark KEM/AES performance across file sizes
python main.py benchmark

# Security self-tests (roundtrip, tamper rejection, wrong key, etc.)
python main.py selftest
```

`--engine` accepts `auto` (default: Groq, falls back to heuristic),
`groq` (force Groq, errors if unavailable), or `heuristic` (never calls
the network).

## Notes / honest limitations (worth saying in a viva)

- Post-quantum KEM is `kyber-py`, a pure-Python FIPS-203 ML-KEM
  implementation — not the native `liboqs` build, which needs a C
  toolchain this environment doesn't have. Swap in `oqs.KeyEncapsulation`
  later if you build liboqs; nothing else in the pipeline needs to change.
- Key storage under `keys/` is plaintext on disk — fine for a prototype,
  not for production (would need an HSM/KMS in a real deployment).
- The Groq risk agent is a genuine reasoning step, not a rule engine —
  but it's still working from regex/keyword evidence plus the visible
  text, so treat "threat_level" and "access_risk" as informed estimates,
  not verified organizational facts, exactly as the original heuristic
  scores were.
