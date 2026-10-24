# Adaptive AI-Driven Post-Quantum File Encryption Framework

Academic prototype. Upload a file → an AI agent assesses its risk → the
framework picks a security profile from the risk class and encrypts the
file with it (ChaCha20-Poly1305 / AES-256-GCM, with conventional or ML-KEM
key protection — see the table below).

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

## Key expiry + crypto-shredding, and key-committing containers (container v3)

Two encryption-stage features that follow the risk class chosen by the agent.

**1. Risk-driven key expiry / crypto-shredding.** Each tier has a key lifetime
(default policy; shorter as risk rises, following NIST SP 800-57's crypto-period guidance):

| Risk | Default key lifetime |
|------|----------------------|
| LOW | no expiry |
| MEDIUM | 90 days |
| HIGH | 30 days |
| CRITICAL | 7 days |

When the lifetime ends, the per-file key material is overwritten and deleted
(`keys/shred/<id>.key`, or `keys/conventional/<id>.key` for LOW). For MEDIUM, HIGH
and CRITICAL the file key is derived from the master key / ML-KEM secret **and** that
per-file shred key, so the file stays unreadable even though the master key and ML-KEM
private key still exist. The expiry is authenticated in the header (it cannot be
extended), enforced by the local ledger `keys/expiry_ledger.json` (not by the
container's own claims), and the web server runs a background sweeper.

```
python main.py encrypt --file x.txt --ttl-seconds 30   # override (0 = never expire)
python main.py inspect --file encrypted/x.txt.pqc      # header + time remaining
python main.py sweep                                   # destroy all expired keys now
python main.py shred   --file encrypted/x.txt.pqc      # destroy this file's key now
```

**2. Key commitment.** AES-GCM and ChaCha20-Poly1305 are not key-committing: one
ciphertext can be crafted to authenticate under two different keys. Every v3 container
therefore carries a 256-bit HKDF-based commitment to the file key, checked in constant
time before decryption. `python main.py selftest` actually builds such a two-key AES-GCM
ciphertext, shows plain GCM accepts it, and shows the commitment separates the keys.

Limitations: shredding relies on overwrite+delete of a local file (not guaranteed on
SSD / journaling / copy-on-write storage); any backup of `keys/` made before expiry
defeats it; the lifetime check uses the local clock (setting the clock back does not
revive a file that was already shredded, but a never-swept file would look unexpired);
v3 containers store no plaintext hash so nothing about the content survives shredding.
Containers made by older versions (v1, v2) still decrypt and have no expiry.

## Risk-scaled self-destruct (attempt budget)

Guarded tiers need a passphrase, and the number of **wrong** passphrases a file tolerates
shrinks as risk rises. When the budget is used up the file's key is crypto-shredded
immediately (same mechanism as key expiry), even if the right passphrase is entered later.

| Risk | Wrong-passphrase budget |
|------|-------------------------|
| LOW | unlimited (no passphrase) |
| MEDIUM | 20 |
| HIGH | 10 |
| CRITICAL | 3 |

* If you do not supply a passphrase, a random 80-bit one is generated and shown **once** (never stored).
* The passphrase is run through scrypt and mixed into the file key together with the per-file shred key.
* A correct passphrase resets the counter. A missing passphrase is not counted.
* The budget is authenticated in the header and enforced from the local ledger, so editing the
  container cannot raise it, and a forged container cannot spend another file's budget.

```
python main.py encrypt --file x.txt --max-attempts 3 --passphrase "my secret"   # 0 = unlimited
python main.py decrypt --file encrypted/x.txt.pqc                                # prompts for passphrase
python main.py inspect --file encrypted/x.txt.pqc                                # attempts left
```

Honest limitation: the counter is enforced by this program and its key store (like a phone's
secure element), not by the mathematics. Someone who copies both `keys/` and the `.pqc` file can
guess offline without the counter - only scrypt and the passphrase length slow them down - and
can reset the ledger. A real deployment would put the counter inside an HSM / TPM.
Guarded tiers also pay one scrypt call (~50 ms) per encrypt and decrypt.

## Notes / honest limitations (worth saying in a viva)

- Post-quantum KEM is `kyber-py`, a pure-Python FIPS-203 ML-KEM
  implementation — not the native `liboqs` build, which needs a C
  toolchain this environment doesn't have. Swap in `oqs.KeyEncapsulation`
  later if you build liboqs; nothing else in the pipeline needs to change.
- Key storage under `keys/` (ML-KEM keys, master key, per-file keys) is plaintext on disk — fine for a prototype,
  not for production (would need an HSM/KMS in a real deployment).
- The Groq risk agent is a genuine reasoning step, not a rule engine —
  but it's still working from regex/keyword evidence plus the visible
  text, so treat "threat_level" and "access_risk" as informed estimates,
  not verified organizational facts, exactly as the original heuristic
  scores were.


## Risk class -> security profile

| Risk | Cipher | Key protection | Security level |
|------|--------|----------------|----------------|
| LOW | ChaCha20-Poly1305 | Conventional key | Basic |
| MEDIUM | AES-256-GCM | Conventional key management | Strong |
| HIGH | AES-256-GCM | ML-KEM-768 (PQC) | Very Strong |
| CRITICAL | AES-256-GCM | ML-KEM-1024 (PQC) | Maximum |

How each key mode works:

- **Conventional key (LOW)** — a fresh random 256-bit key per file, stored
  in `keys/conventional/<key_id>.key`. The `.pqc` container only holds the
  `key_id`.
- **Conventional key management (MEDIUM)** — envelope encryption: a fresh
  random file key per file, wrapped (AES key wrap, RFC 3394) with a master
  key in `keys/master_kek.bin`. The container holds the wrapped key.
- **ML-KEM (HIGH / CRITICAL)** — the file key is derived from an ML-KEM
  (FIPS 203) shared secret via HKDF-SHA256. No key is stored; the container
  holds the KEM ciphertext.

Every container authenticates its own header (cipher, key mode, KEM) as AEAD
associated data, so changing a header field (e.g. downgrading HIGH to a
conventional key) makes decryption fail. Containers from the earlier
ML-KEM-only version (v1) still decrypt.

**Back up `keys/`.** Without it, LOW / MEDIUM / HIGH / CRITICAL files can no
longer be decrypted. It is git-ignored on purpose — never commit it.