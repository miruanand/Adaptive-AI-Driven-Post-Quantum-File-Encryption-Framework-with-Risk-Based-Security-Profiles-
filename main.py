#!/usr/bin/env python3
"""
Adaptive AI-Driven Post-Quantum File Encryption Framework
with Risk-Based Security Profiles
===========================================================

ACADEMIC / RESEARCH PROTOTYPE — NOT a production security product.

This is a SINGLE-FILE consolidation of the full pipeline described in the
project brief:

    FILE -> Feature Extraction -> AI Risk Assessment -> Risk Score/Class
         -> Security Profile Selection -> ChaCha20-Poly1305 / AES-256-GCM
            with conventional keys (LOW/MEDIUM) or ML-KEM keys (HIGH/CRITICAL)
         -> Encrypted File -> Decryption -> SHA-256 Integrity Check
         -> Benchmarking / Results

--------------------------------------------------------------------------
HONEST DISCLOSURE ABOUT POST-QUANTUM CRYPTOGRAPHY IN THIS BUILD
--------------------------------------------------------------------------
The brief asked for the Open Quantum Safe `liboqs` Python bindings
(`liboqs-python` / `oqs`). That package is only a *wrapper*: it needs the
`liboqs` C library compiled on the host machine (cmake + a C toolchain +
the liboqs source tree). That native build step is not available in this
sandboxed execution environment, and the PyPI package literally named
`oqs` is an unrelated project (a toy scripting-language interpreter), so
installing it would have silently given you the wrong library.

Rather than fabricate ML-KEM support or silently fall back to RSA/ECC
(which the brief explicitly forbids), this build uses **kyber-py**
(https://pypi.org/project/kyber-py/), a pure-Python implementation of
**ML-KEM exactly as standardized in FIPS 203** (NIST PQC standard), with
no native build step. It was verified in this environment against the
expected FIPS 203 key/ciphertext sizes for all three parameter sets:

    ML-KEM-512  : ek=800B,  dk=1632B, ct=768B,  shared secret=32B
    ML-KEM-768  : ek=1184B, dk=2400B, ct=1088B, shared secret=32B
    ML-KEM-1024 : ek=1568B, dk=3168B, ct=1568B, shared secret=32B

This is real, standards-compliant post-quantum key encapsulation — just a
different (pure-Python, not liboqs-backed) implementation of the same
algorithm. It will be noticeably slower than a native liboqs build,
which is worth noting in a viva if asked about performance. If you later
run this on a machine where you *can* build liboqs, replace the small
`KEM` adapter class below with calls into `oqs.KeyEncapsulation` — the
rest of the pipeline (AES-GCM, HKDF, container format, Flask UI, AI
model) does not need to change.

--------------------------------------------------------------------------
WHAT IS AND ISN'T INCLUDED IN THIS SINGLE FILE
--------------------------------------------------------------------------
Included and actually executed/tested in this environment:
  - Feature extraction (txt/csv/json/md/log/xml/pdf/docx)
  - Synthetic-policy-labelled dataset generation + RandomForest risk model
  - Risk-based security profile selection
  - Profile-driven encryption/decryption:
      LOW      ChaCha20-Poly1305 + conventional key
      MEDIUM   AES-256-GCM       + conventional key management (wrapped keys)
      HIGH     AES-256-GCM       + ML-KEM-768  (PQC key establishment)
      CRITICAL AES-256-GCM       + ML-KEM-1024 (PQC key establishment)
  - Risk-driven KEY EXPIRY + CRYPTO-SHREDDING: each tier has a key lifetime
    (LOW none, MEDIUM 90 d, HIGH 30 d, CRITICAL 7 d by default). When it
    ends the per-file key material is destroyed, so the file can never be
    decrypted again - even by someone holding the master / ML-KEM keys.
  - RISK-SCALED SELF-DESTRUCT: guarded tiers need a passphrase and tolerate only
    a few wrong ones (MEDIUM 20, HIGH 10, CRITICAL 3); when the budget is used
    up the file's key is crypto-shredded immediately.
  - KEY-COMMITTING container (v3): a 256-bit key commitment is verified before
    the AEAD runs, so one ciphertext can never be opened by two different keys
    (AES-GCM / ChaCha20-Poly1305 alone do not guarantee this).
  - SHA-256 integrity verification
  - A benchmark command (CSV/JSON output, mean+std over repetitions)
  - A self-test command covering the core security validation cases
  - A minimal Flask web dashboard (upload -> analyze -> encrypt -> decrypt)

NOT included in this single file (would meaningfully bloat one file):
  - The matplotlib 6-graph experiment suite (experiments.py) with adaptive
    vs. fixed-baseline plots
  - A separate pytest test tree (the equivalent checks are in `selftest`)
  - Automatic PDF-report table generation
  If you want those as a second file, ask and they can be generated
  separately so this file doesn't balloon past a few thousand lines.

Run `python adaptive_pqc_framework.py --help` for all commands.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hmac
import json
import math
import os
import secrets
import re  # noqa: F811 (kept here; used throughout feature extraction below)
import statistics
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads GROQ_API_KEY / GROQ_MODEL from a .env file next to this script, if present
except ImportError:
    pass  # python-dotenv is optional; env vars set another way still work fine

# --------------------------------------------------------------------------
# 0. PATHS / CONSTANTS
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
KEYS_DIR = BASE_DIR / "keys"
UPLOADS_DIR = BASE_DIR / "uploads"
ENCRYPTED_DIR = BASE_DIR / "encrypted"
DECRYPTED_DIR = BASE_DIR / "decrypted"
RESULTS_DIR = BASE_DIR / "results"

for _d in (MODELS_DIR, KEYS_DIR, UPLOADS_DIR, ENCRYPTED_DIR, DECRYPTED_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "risk_model.joblib"
RANDOM_SEED = 42
CONTAINER_VERSION = 3   # v3 adds key commitment + key expiry; v1 and v2 containers are still readable

RISK_CLASSES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
FEATURE_NAMES = [
    "sensitivity", "threat_level", "access_risk", "external_sharing",
    "personal_data", "financial_data", "confidential_keywords",
    "file_size_mb_log",
]

# --------------------------------------------------------------------------
# 1. POST-QUANTUM KEM ADAPTER  (see disclosure above)
# --------------------------------------------------------------------------

try:
    from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
    _KEM_BACKEND_AVAILABLE = True
    _KEM_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - exercised only if dependency missing
    _KEM_BACKEND_AVAILABLE = False
    _KEM_IMPORT_ERROR = _e

_KEM_TABLE = {
    "ML-KEM-512": ML_KEM_512 if _KEM_BACKEND_AVAILABLE else None,
    "ML-KEM-768": ML_KEM_768 if _KEM_BACKEND_AVAILABLE else None,
    "ML-KEM-1024": ML_KEM_1024 if _KEM_BACKEND_AVAILABLE else None,
}


class KEMUnavailableError(RuntimeError):
    pass


class UnsupportedKEMError(ValueError):
    pass


def _get_kem(algo: str):
    if not _KEM_BACKEND_AVAILABLE:
        raise KEMUnavailableError(
            "Post-quantum KEM backend 'kyber-py' is not installed.\n"
            "Install it with:  pip install kyber-py\n"
            f"Original import error: {_KEM_IMPORT_ERROR}\n"
            "This project deliberately does NOT fall back to RSA/ECC."
        )
    if algo not in _KEM_TABLE:
        raise UnsupportedKEMError(f"Unsupported KEM algorithm requested: {algo!r}")
    return _KEM_TABLE[algo]


def kem_keygen(algo: str):
    kem = _get_kem(algo)
    ek, dk = kem.keygen()
    return ek, dk


def kem_encaps(algo: str, ek: bytes):
    kem = _get_kem(algo)
    shared_secret, ciphertext = kem.encaps(ek)
    return shared_secret, ciphertext


def kem_decaps(algo: str, dk: bytes, ciphertext: bytes):
    kem = _get_kem(algo)
    return kem.decaps(dk, ciphertext)


# --------------------------------------------------------------------------
# 2. KEY MANAGEMENT
# --------------------------------------------------------------------------

class KeyStoreError(RuntimeError):
    """A required key is missing or malformed in the local key store."""


class KeyShreddedError(KeyStoreError):
    """The file's key material has expired or been destroyed (crypto-shredded)."""


class PassphraseRequiredError(KeyStoreError):
    """The file is guarded by an attempt budget and needs a passphrase."""


class WrongPassphraseError(KeyStoreError):
    """Wrong passphrase; `remaining` attempts are left before the key self-destructs."""
    def __init__(self, msg: str, remaining: int):
        super().__init__(msg)
        self.remaining = remaining


class KeyManager:
    """
    Academic-prototype key storage: raw key bytes on the local filesystem.

    Three kinds of key material live under keys/:
      * ML-KEM key pairs  (keys/ML-KEM-768_ek.bin, _dk.bin, ...)   -> HIGH / CRITICAL
      * a master key      (keys/master_kek.bin)                    -> MEDIUM
        (wraps a fresh random file key for every file: envelope encryption)
      * per-file keys     (keys/conventional/<key_id>.key)         -> LOW
        (a fresh random 256-bit key per file, referenced by key_id)

    SECURITY CAVEAT (documented, not hidden): this is NOT suitable for
    production. A real deployment needs secure key storage (HSM / KMS),
    access control, key rotation, and authenticated key retrieval.
    Losing the keys/ folder means LOW / MEDIUM / HIGH / CRITICAL files can
    no longer be decrypted - back it up.
    """

    # key_id values come from the container file, which is untrusted input
    # (users upload .pqc files to /decrypt). Only accept exactly what we
    # generate, so a crafted key_id can never point outside keys/conventional.
    _KEY_ID_RE = re.compile(r"[0-9a-f]{32}")

    def __init__(self, keys_dir: Path = KEYS_DIR):
        self.keys_dir = keys_dir
        self.keys_dir.mkdir(parents=True, exist_ok=True)

    # ---- ML-KEM key pairs (HIGH / CRITICAL) ---------------------------
    def _paths(self, algo: str):
        safe = algo.replace("/", "_")
        return (self.keys_dir / f"{safe}_ek.bin", self.keys_dir / f"{safe}_dk.bin")

    def ensure_keypair(self, algo: str):
        ek_path, dk_path = self._paths(algo)
        if ek_path.exists() and dk_path.exists():
            return
        ek, dk = kem_keygen(algo)
        ek_path.write_bytes(ek)
        dk_path.write_bytes(dk)

    def load_ek(self, algo: str) -> bytes:
        ek_path, _ = self._paths(algo)
        if not ek_path.exists():
            self.ensure_keypair(algo)
        return ek_path.read_bytes()

    def load_dk(self, algo: str) -> bytes:
        _, dk_path = self._paths(algo)
        if not dk_path.exists():
            self.ensure_keypair(algo)
        return dk_path.read_bytes()

    # ---- conventional keys (LOW / MEDIUM) -----------------------------
    @staticmethod
    def _write_secret(path: Path, data: bytes):
        path.write_bytes(data)
        try:
            os.chmod(path, 0o600)  # best effort; mostly a no-op on Windows
        except OSError:
            pass

    def master_kek(self, create: bool = True) -> bytes:
        """
        The master key-encryption key used by the MEDIUM profile.
        Created on first use when encrypting. When decrypting, pass
        create=False so a missing master key is reported instead of being
        silently replaced by a brand-new (useless) one.
        """
        path = self.keys_dir / "master_kek.bin"
        if not path.exists():
            if not create:
                raise KeyStoreError(
                    "Master key (keys/master_kek.bin) not found - this file was "
                    "encrypted with a different keys/ folder or the folder was deleted."
                )
            self._write_secret(path, os.urandom(32))
        kek = path.read_bytes()
        if len(kek) != 32:
            raise KeyStoreError("keys/master_kek.bin is corrupt (expected 32 bytes).")
        return kek

    def new_conventional_key(self) -> tuple:
        """Create + store a fresh random 256-bit key for one file (LOW profile)."""
        key_id = uuid.uuid4().hex
        key = os.urandom(32)
        conv_dir = self.keys_dir / "conventional"
        conv_dir.mkdir(parents=True, exist_ok=True)
        self._write_secret(conv_dir / f"{key_id}.key", key)
        return key_id, key

    def load_conventional_key(self, key_id) -> bytes:
        if not isinstance(key_id, str) or not self._KEY_ID_RE.fullmatch(key_id):
            raise KeyStoreError("Invalid key_id in container.")
        path = self.keys_dir / "conventional" / f"{key_id}.key"
        if not path.exists():
            self._raise_missing(key_id)
        key = path.read_bytes()
        if len(key) != 32:
            raise KeyStoreError(f"Stored key {key_id} is corrupt (expected 32 bytes).")
        return key

    # ---- per-file shred keys + expiry ledger (key expiry / crypto-shredding) --
    #
    # A "shred key" is a random 256-bit secret kept ONLY in keys/shred/<id>.key.
    # For MEDIUM / HIGH / CRITICAL files with an expiry, the file key is derived
    # from (master key | ML-KEM secret) AND the shred key. Destroying the shred
    # key makes the file undecryptable even for someone who still holds the
    # master key or the ML-KEM private key. For LOW files the per-file
    # conventional key itself is the thing that gets destroyed.
    #
    # keys/expiry_ledger.json records id -> expiry / status so that a sweep can
    # destroy keys on time and decrypt can say "shredded" rather than "missing".

    _LEDGER_LOCK = threading.RLock()

    @property
    def _ledger_path(self) -> Path:
        return self.keys_dir / "expiry_ledger.json"

    def _read_ledger(self) -> dict:
        try:
            data = json.loads(self._ledger_path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_ledger(self, ledger: dict):
        tmp = self._ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger, indent=2))
        os.replace(tmp, self._ledger_path)

    def ledger(self) -> dict:
        with self._LEDGER_LOCK:
            return self._read_ledger()

    def new_shred_key(self) -> tuple:
        shred_id = uuid.uuid4().hex
        key = os.urandom(32)
        d = self.keys_dir / "shred"
        d.mkdir(parents=True, exist_ok=True)
        self._write_secret(d / f"{shred_id}.key", key)
        return shred_id, key

    def load_shred_key(self, shred_id) -> bytes:
        if not isinstance(shred_id, str) or not self._KEY_ID_RE.fullmatch(shred_id):
            raise KeyStoreError("Invalid shred_id in container.")
        path = self.keys_dir / "shred" / f"{shred_id}.key"
        if not path.exists():
            self._raise_missing(shred_id)
        key = path.read_bytes()
        if len(key) != 32:
            raise KeyStoreError(f"Stored shred key {shred_id} is corrupt (expected 32 bytes).")
        return key

    def _raise_missing(self, key_id: str):
        entry = self.ledger().get(key_id)
        if entry and entry.get("status") == "SHREDDED":
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.get("shredded_at", 0)))
            raise KeyShreddedError(
                f"The key for this file was crypto-shredded at {when} "
                f"({entry.get('reason', 'expired')}). The data is permanently unrecoverable."
            )
        raise KeyStoreError(
            f"No stored key for id {key_id} - this file was encrypted with "
            "a different keys/ folder or the folder was deleted."
        )

    # fields that must never be shown through the web API
    _PRIVATE_LEDGER_FIELDS = ("pw_verifier", "commitment")

    def public_ledger(self) -> dict:
        return {k: {f: v for f, v in e.items() if f not in self._PRIVATE_LEDGER_FIELDS}
                for k, e in self.ledger().items()}

    def register_expiry(self, key_id: str, created_at: float, expires_at: Optional[float],
                        label: str, profile_name: str, key_mode: str,
                        max_attempts: Optional[int] = None,
                        pw_verifier: Optional[bytes] = None,
                        commitment: Optional[bytes] = None):
        with self._LEDGER_LOCK:
            ledger = self._read_ledger()
            entry = {
                "label": label, "profile": profile_name, "key_mode": key_mode,
                "created_at": created_at, "expires_at": expires_at, "status": "ACTIVE",
            }
            if max_attempts:
                entry.update({
                    "max_attempts": int(max_attempts), "failed_attempts": 0,
                    "pw_verifier": _b64e(pw_verifier), "commitment": _b64e(commitment),
                })
            ledger[key_id] = entry
            self._write_ledger(ledger)

    def record_failed_attempt(self, key_id: str, now: Optional[float] = None) -> tuple:
        """Count one wrong passphrase. Returns (attempts_left, key_destroyed).
        When the budget is used up the key material is crypto-shredded."""
        with self._LEDGER_LOCK:
            ledger = self._read_ledger()
            e = ledger[key_id]
            e["failed_attempts"] = int(e.get("failed_attempts", 0)) + 1
            left = int(e["max_attempts"]) - e["failed_attempts"]
            self._write_ledger(ledger)
            if left <= 0:
                self.shred(key_id, reason="attempt budget exhausted", now=now)
                return 0, True
            return left, False

    def reset_attempts(self, key_id: str):
        with self._LEDGER_LOCK:
            ledger = self._read_ledger()
            e = ledger.get(key_id)
            if e and e.get("failed_attempts"):
                e["failed_attempts"] = 0
                self._write_ledger(ledger)

    @staticmethod
    def _overwrite_and_delete(path: Path) -> bool:
        """Best-effort secure delete: overwrite with random bytes, fsync, unlink.
        NOTE: on SSDs / journaling or copy-on-write filesystems, overwriting
        is not a guarantee - the real protection is that the key was never
        stored anywhere else (see README caveats)."""
        if not path.exists():
            return False
        try:
            size = path.stat().st_size
            with open(path, "r+b") as f:
                f.write(os.urandom(max(size, 32)))
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass
        path.unlink()
        return True

    def shred(self, key_id: str, reason: str = "manual", now: Optional[float] = None) -> bool:
        """Destroy the key material for `key_id` (idempotent). Returns True if
        something was destroyed."""
        if not isinstance(key_id, str) or not self._KEY_ID_RE.fullmatch(key_id):
            raise KeyStoreError("Invalid key id.")
        destroyed = False
        with self._LEDGER_LOCK:
            destroyed |= self._overwrite_and_delete(self.keys_dir / "shred" / f"{key_id}.key")
            destroyed |= self._overwrite_and_delete(self.keys_dir / "conventional" / f"{key_id}.key")
            ledger = self._read_ledger()
            entry = ledger.get(key_id)
            if entry is not None and entry.get("status") != "SHREDDED":
                entry.update({"status": "SHREDDED", "shredded_at": now if now is not None else time.time(),
                              "reason": reason})
                self._write_ledger(ledger)
        return destroyed

    def sweep_expired(self, now: Optional[float] = None) -> list:
        """Destroy every key whose expiry has passed. Returns the list of ids."""
        now = now if now is not None else time.time()
        done = []
        with self._LEDGER_LOCK:
            for kid, e in list(self._read_ledger().items()):
                exp = e.get("expires_at")
                if e.get("status") == "ACTIVE" and exp is not None and exp <= now:
                    self.shred(kid, reason="expired", now=now)
                    done.append(kid)
        return done

    def ensure_all(self):
        for algo in _KEM_TABLE:
            self.ensure_keypair(algo)
        self.master_kek()


# --------------------------------------------------------------------------
# 3. FEATURE EXTRACTION  —  FULLY CONTENT-DERIVED (no manual sliders)
# --------------------------------------------------------------------------
#
# DESIGN NOTE / HONEST LIMITATION (read this before trusting the numbers):
#
# All 8 risk features are now estimated from the file's own content and
# metadata. Two of the features (personal_data, financial_data via PII
# regex, confidential_keywords) rest on fairly direct evidence — an
# Aadhaar-shaped number or a credit-card-shaped number really is present
# in the text. The other four (sensitivity, threat_level, access_risk,
# external_sharing) are *proxies*, not measurements: a file's bytes cannot
# tell you who actually has access to it, who is actually targeting it, or
# what your organization's policy classifies it as. What content CAN give
# is textual evidence that correlates with those things — explicit
# markers ("TOP SECRET"), threat-related vocabulary, embedded email
# domains, distribution/sharing language. That is what the heuristics
# below use. Treat these four as informed estimates for demoing the
# adaptive architecture, not as ground truth — say this explicitly in your
# viva if asked how "threat_level" is measured.
#
# If you want to override any single value (e.g. because you *do* know
# the organizational context), every CLI/HTTP entry point still accepts
# optional manual overrides — see the `overrides` parameter — but nothing
# is required beyond the file itself.

INDICATOR_KEYWORDS = [
    "confidential", "secret", "private", "restricted", "password",
    "salary", "employee", "medical", "patient", "aadhaar", "passport",
    "bank", "account number", "credit card", "financial", "invoice",
]
PERSONAL_DATA_KEYWORDS = [
    "aadhaar", "passport", "patient", "medical", "employee", "salary",
    "home address", "residential address", "mailing address",
    "postal address", "date of birth", "ssn",
]
FINANCIAL_DATA_KEYWORDS = [
    "bank", "account number", "credit card", "financial", "invoice",
    "salary", "iban", "swift",
]
SENSITIVITY_MARKER_KEYWORDS = [
    "top secret", "strictly confidential", "classified", "highly sensitive",
    "not for distribution", "attorney-client privileged",
]
THREAT_KEYWORDS = [
    "breach", "data breach", "leak", "leaked", "hacked", "hack", "attack",
    "malware", "phishing", "exploit", "vulnerability", "ransomware",
    "litigation", "lawsuit", "whistleblower", "insider threat", "subpoena",
]
ACCESS_RISK_HIGH_KEYWORDS = [
    "shared drive", "public folder", "all-staff", "company-wide", "everyone",
    "open access", "unrestricted access", "published internally",
    "shared with all",
]
ACCESS_RISK_LOW_KEYWORDS = [
    "restricted access", "need-to-know", "do not distribute",
    "access limited", "authorized personnel only", "internal use only",
]
EXTERNAL_SHARING_KEYWORDS = [
    "external partner", "client copy", "vendor", "third party",
    "forwarded to", "shared externally", "external distribution",
    "cc:", "bcc:", "attached for review by",
]
PUBLIC_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "protonmail.com", "aol.com",
}

EMAIL_RE = re.compile(r"[\w.\-+]+@([\w\-]+\.[\w.\-]+)")
URL_RE = re.compile(r"https?://[^\s)>\]]+")
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\d{10}\b")

TEXT_EXTENSIONS = {".txt", ".csv", ".json", ".md", ".log", ".xml"}


# Cap how many times a single keyword can count toward a score. Without
# this, one ordinary word that happens to be topical for a document (e.g.
# an academic paper on cryptographic "attacks" saying "attack" a dozen
# times) can single-handedly saturate a whole feature at 100, even though
# repeating the same word 12 times is not 12x the risk signal of saying
# it once. Counting *how many different* indicator terms appear, each
# capped, better approximates "how many distinct risk signals are
# present" than raw occurrence counts do.
_MAX_HITS_PER_KEYWORD = 3

_KEYWORD_RE_CACHE: dict = {}


def _keyword_hits(text_lower: str, keywords: list) -> int:
    """
    Counts keyword occurrences using whole-word/whole-phrase matching
    (not raw substring matching), with each individual keyword's
    contribution capped.

    Word-boundary matching matters: a plain substring check on "address"
    also matches "addresses" ("this paper addresses the problem of..."),
    "IP address", "email address", etc. — none of which indicate an
    actual postal address is present. Substring checks on "attack" also
    match "attacker"/"attacks", which is fine semantically but, combined
    with no cap, let a single recurring term (common in any
    cybersecurity-themed document) alone drive a score to its maximum.
    """
    total = 0
    for kw in keywords:
        pattern = _KEYWORD_RE_CACHE.get(kw)
        if pattern is None:
            pattern = re.compile(r"\b" + re.escape(kw) + r"\b")
            _KEYWORD_RE_CACHE[kw] = pattern
        hits = len(pattern.findall(text_lower))
        total += min(hits, _MAX_HITS_PER_KEYWORD)
    return total


def _read_text_generic(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def _read_docx_text(path: Path) -> str:
    try:
        import docx
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return _read_text_generic(path)
    if ext == ".pdf":
        return _read_pdf_text(path)
    if ext == ".docx":
        return _read_docx_text(path)
    # Unknown/binary type: no textual indicator extraction is attempted.
    return ""


@dataclass
class AutoFeatures:
    """All 8 model features, derived entirely from the file — no manual input."""
    sensitivity: float
    threat_level: float
    access_risk: float
    external_sharing: float
    personal_data: float
    financial_data: float
    confidential_keywords: float
    file_size_mb_log: float
    # Diagnostic evidence, shown in the UI/CLI so the numbers are explainable
    # rather than a black box — useful for a viva walkthrough.
    matched_indicators: list
    evidence: dict


def extract_auto_features(path: Path, raw_text: Optional[str] = None) -> AutoFeatures:
    """
    Derives ALL risk features from the file's content and metadata alone.

    See the design-note comment above INDICATOR_KEYWORDS for the honest
    limitation: personal_data/financial_data/confidential_keywords rest on
    fairly direct textual/PII-pattern evidence, while sensitivity,
    threat_level, access_risk and external_sharing are heuristic proxies
    built from that same evidence — not verified organizational facts.

    `raw_text` can be passed in by a caller that already extracted the
    file's text (e.g. the Groq risk assessor, which sends the same text to
    the LLM) so the file is not parsed twice.
    """
    if raw_text is None:
        raw_text = extract_text(path)
    text = raw_text.lower()

    matched_confidential = sorted({kw for kw in INDICATOR_KEYWORDS if kw in text})
    confidential_score = min(100.0, len(matched_confidential) * 15.0)

    emails = EMAIL_RE.findall(raw_text)
    domains = {d.lower() for d in emails}
    urls = URL_RE.findall(raw_text)
    aadhaar_hits = len(AADHAAR_RE.findall(raw_text))
    pan_hits = len(PAN_RE.findall(raw_text))
    cc_hits = len(CREDIT_CARD_RE.findall(raw_text))
    phone_hits = len(PHONE_RE.findall(raw_text))

    personal_kw_hits = _keyword_hits(text, PERSONAL_DATA_KEYWORDS)
    financial_kw_hits = _keyword_hits(text, FINANCIAL_DATA_KEYWORDS)

    personal_data = min(100.0, personal_kw_hits * 12.0 + aadhaar_hits * 40.0
                         + pan_hits * 40.0 + phone_hits * 8.0)
    financial_data = min(100.0, financial_kw_hits * 12.0 + cc_hits * 40.0)

    size_mb = path.stat().st_size / (1024 * 1024)
    size_log = math.log(1 + size_mb)

    # --- heuristic proxies for context that content can only hint at ---
    marker_hits = _keyword_hits(text, SENSITIVITY_MARKER_KEYWORDS)
    sensitivity = min(100.0,
                       0.4 * confidential_score + 0.3 * personal_data
                       + 0.3 * financial_data + marker_hits * 20.0)

    threat_hits = _keyword_hits(text, THREAT_KEYWORDS)
    threat_level = min(100.0, threat_hits * 20.0)

    access_high_hits = _keyword_hits(text, ACCESS_RISK_HIGH_KEYWORDS)
    access_low_hits = _keyword_hits(text, ACCESS_RISK_LOW_KEYWORDS)
    domain_signal = min(60.0, len(domains) * 15.0)
    access_risk = min(100.0, max(0.0,
                       domain_signal + access_high_hits * 20.0 - access_low_hits * 20.0))

    public_domain_hits = len(domains & PUBLIC_EMAIL_DOMAINS)
    external_kw_hits = _keyword_hits(text, EXTERNAL_SHARING_KEYWORDS)
    external_sharing = min(100.0,
                            len(domains) * 20.0 + public_domain_hits * 30.0
                            + external_kw_hits * 15.0 + (10.0 if urls else 0.0))

    evidence = {
        "emails_found": sorted(set(re.findall(r"[\w.\-+]+@[\w.\-]+", raw_text))),
        "domains_found": sorted(domains),
        "urls_found": urls[:10],
        "aadhaar_pattern_hits": aadhaar_hits,
        "pan_pattern_hits": pan_hits,
        "credit_card_pattern_hits": cc_hits,
        "phone_pattern_hits": phone_hits,
        "threat_keyword_hits": threat_hits,
        "sensitivity_marker_hits": marker_hits,
        "access_high_keyword_hits": access_high_hits,
        "access_low_keyword_hits": access_low_hits,
        "external_keyword_hits": external_kw_hits,
    }

    return AutoFeatures(
        sensitivity=sensitivity,
        threat_level=threat_level,
        access_risk=access_risk,
        external_sharing=external_sharing,
        personal_data=personal_data,
        financial_data=financial_data,
        confidential_keywords=confidential_score,
        file_size_mb_log=size_log,
        matched_indicators=matched_confidential,
        evidence=evidence,
    )


def build_feature_vector(auto: AutoFeatures, overrides: Optional[dict] = None) -> np.ndarray:
    """
    Builds the 8-value model input from fully automatic features.

    `overrides` is optional and only for power users who want to manually
    correct a specific value (e.g. they know access_risk is actually higher
    than the file suggests). Keys: sensitivity, threat_level, access_risk,
    external_sharing, personal_data, financial_data, confidential_keywords,
    file_size_mb_log. Nothing needs to be passed — the file alone is enough.
    """
    values = {
        "sensitivity": auto.sensitivity,
        "threat_level": auto.threat_level,
        "access_risk": auto.access_risk,
        "external_sharing": auto.external_sharing,
        "personal_data": auto.personal_data,
        "financial_data": auto.financial_data,
        "confidential_keywords": auto.confidential_keywords,
        "file_size_mb_log": auto.file_size_mb_log,
    }
    if overrides:
        for k, v in overrides.items():
            if k in values and v is not None:
                values[k] = float(v)
    return np.array([[values[name] for name in FEATURE_NAMES]], dtype=float)


# --------------------------------------------------------------------------
# 4. AI RISK MODEL (synthetic-policy-labelled RandomForest)
# --------------------------------------------------------------------------

def _policy_score(row: np.ndarray) -> float:
    (sensitivity, threat_level, access_risk, external_sharing,
     personal_data, financial_data, confidential_keywords, file_size_mb_log) = row
    return (
        0.28 * sensitivity + 0.22 * threat_level + 0.15 * access_risk
        + 0.08 * external_sharing + 0.10 * personal_data + 0.08 * financial_data
        + 0.07 * confidential_keywords + 0.02 * min(file_size_mb_log, 100)
    )


def _score_to_class(score: float) -> str:
    if score < 30:
        return "LOW"
    if score < 60:
        return "MEDIUM"
    if score < 80:
        return "HIGH"
    return "CRITICAL"


def generate_synthetic_dataset(n_samples: int = 8000, seed: int = RANDOM_SEED):
    """
    Generates a synthetic dataset labelled by a TRANSPARENT WEIGHTED POLICY
    (documented in the module docstring / README), because a large,
    manually labelled real-world dataset is not yet available.

    This dataset validates the END-TO-END ADAPTIVE ARCHITECTURE. It does
    NOT prove real-world AI risk-assessment accuracy — the labels are
    generated by the same formula the "AI" is asked to reconstruct, so
    high accuracy here only shows the classifier can learn the policy
    boundary, not that the policy itself matches real-world risk.
    """
    rng = np.random.default_rng(seed)
    sensitivity = rng.uniform(0, 100, n_samples)
    threat_level = rng.uniform(0, 100, n_samples)
    access_risk = rng.uniform(0, 100, n_samples)
    external_sharing = rng.choice([0.0, 100.0], size=n_samples, p=[0.6, 0.4])
    personal_data = rng.choice([0.0, 100.0], size=n_samples, p=[0.55, 0.45])
    financial_data = rng.choice([0.0, 100.0], size=n_samples, p=[0.6, 0.4])
    confidential_keywords = rng.uniform(0, 100, n_samples)
    file_size_mb = rng.exponential(scale=5.0, size=n_samples)
    file_size_mb_log = np.log1p(file_size_mb)

    X = np.column_stack([
        sensitivity, threat_level, access_risk, external_sharing,
        personal_data, financial_data, confidential_keywords, file_size_mb_log,
    ])

    # small label noise to avoid a perfectly separable toy problem
    noise = rng.normal(0, 3.0, n_samples)
    scores = np.array([_policy_score(row) for row in X]) + noise
    scores = np.clip(scores, 0, 100)
    y = np.array([_score_to_class(s) for s in scores])

    return X, y, scores


class RiskModel:
    def __init__(self):
        self.model = None
        self.feature_names = FEATURE_NAMES

    def train(self, n_samples: int = 8000, seed: int = RANDOM_SEED) -> dict:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix, classification_report,
        )
        import joblib

        X, y, scores = generate_synthetic_dataset(n_samples, seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        self.model = clf

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=RISK_CLASSES)
        report = classification_report(y_test, y_pred, labels=RISK_CLASSES, zero_division=0)

        joblib.dump({"model": clf, "feature_names": self.feature_names}, MODEL_PATH)

        metrics = {
            "n_samples": n_samples,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": acc,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": RISK_CLASSES,
            "classification_report": report,
            "feature_importances": dict(zip(self.feature_names, clf.feature_importances_.tolist())),
            "note": (
                "Labels are generated from a transparent weighted policy formula "
                "(see docstring), not from manually labelled real-world data. "
                "These metrics validate that the classifier learned the policy's "
                "decision boundary, NOT that the policy reflects real-world risk."
            ),
        }
        self._write_metrics_report(metrics)
        return metrics

    def _write_metrics_report(self, metrics: dict):
        ai_dir = RESULTS_DIR / "ai"
        ai_dir.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "ai_metrics.txt", "w") as f:
            f.write("AI RISK MODEL METRICS (synthetic policy-labelled data)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Samples: {metrics['n_samples']} (train={metrics['n_train']}, test={metrics['n_test']})\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (weighted): {metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall (weighted):    {metrics['recall_weighted']:.4f}\n")
            f.write(f"F1 (weighted):        {metrics['f1_weighted']:.4f}\n\n")
            f.write("Confusion matrix (rows=true, cols=pred), labels="
                    f"{metrics['confusion_matrix_labels']}\n")
            for row in metrics["confusion_matrix"]:
                f.write(str(row) + "\n")
            f.write("\nClassification report:\n")
            f.write(metrics["classification_report"] + "\n")
            f.write("Feature importances:\n")
            for k, v in metrics["feature_importances"].items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\nNOTE: " + metrics["note"] + "\n")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            names = list(metrics["feature_importances"].keys())
            vals = list(metrics["feature_importances"].values())
            order = np.argsort(vals)
            plt.figure(figsize=(8, 5))
            plt.barh([names[i] for i in order], [vals[i] for i in order])
            plt.xlabel("Importance")
            plt.title("Risk model feature importances")
            plt.tight_layout()
            plt.savefig(ai_dir / "feature_importance.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"[warn] could not save feature importance plot: {e}")

    def load(self):
        import joblib
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH}. Run with `train` first."
            )
        payload = joblib.load(MODEL_PATH)
        self.model = payload["model"]
        self.feature_names = payload["feature_names"]
        return self

    def predict(self, feature_vector: np.ndarray):
        if self.model is None:
            self.load()
        pred_class = self.model.predict(feature_vector)[0]
        proba = self.model.predict_proba(feature_vector)[0]
        classes = list(self.model.classes_)
        confidence = float(max(proba))
        proba_map = {c: float(p) for c, p in zip(classes, proba)}
        return pred_class, confidence, proba_map


# --------------------------------------------------------------------------
# 4B. GROQ LLM-BASED AI RISK ASSESSMENT  (primary risk-assessment engine)
# --------------------------------------------------------------------------
#
# This replaces the RandomForest-on-synthetic-labels classifier above as
# the primary decision-maker. Instead of a model trained on a hand-written
# policy formula, an actual reasoning LLM (served by Groq) reads the
# automated evidence (PII/keyword regex hits) plus a bounded excerpt of the
# file's own text and makes the risk call itself, with a written
# justification you can show in a viva.
#
# The RandomForest RiskModel above is KEPT and used as an offline fallback
# — if no GROQ_API_KEY is configured, the Groq package isn't installed, or
# the API call fails/rate-limits, the pipeline automatically falls back to
# it and clearly marks the result as a fallback (never silently swaps a
# real Groq verdict for the heuristic one).

GROQ_MODEL_DEFAULT = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TEXT_CHARS = 6000  # keep prompts small/cheap; evidence is already summarized

RISK_ASSESSMENT_SYSTEM_PROMPT = """You are the risk-assessment agent inside an adaptive post-quantum file encryption system.

For every file you are shown you must decide how much data-security risk the file's CONTENT represents, so the system can pick a stronger or lighter post-quantum encryption profile. You are given:
- the file name
- automated evidence extracted by regex/keyword scanners (PII pattern hits such as Aadhaar/PAN/credit-card/phone numbers, matched confidential/financial/personal-data keywords, email domains, URLs, threat-related language, access-control language)
- a text excerpt from the file (may be truncated, may be empty if the file is a binary/unsupported type)

Classify the file into EXACTLY one of these four risk classes:
- LOW: no sensitive content; public, trivial, or purely structural data
- MEDIUM: some internal/business information, limited sensitivity, nothing clearly regulated
- HIGH: personal data, financial data, or confidential business information is present
- CRITICAL: highly sensitive PII, financial records, legal/medical data, explicit confidentiality markers ("TOP SECRET", "strictly confidential", etc.), or signs of an active threat/breach/litigation context

Also score these 8 factors from 0-100 using your own judgement of the evidence and excerpt (do not just restate raw counts):
sensitivity, threat_level, access_risk, external_sharing, personal_data, financial_data, confidential_keywords, file_size_mb_log (set file_size_mb_log to 0; the caller fills it in from the real file size).

Respond with ONLY a single JSON object — no markdown code fences, no text outside the JSON — in exactly this shape:
{
  "risk_class": "LOW|MEDIUM|HIGH|CRITICAL",
  "confidence": 0.0-1.0,
  "class_probabilities": {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0, "CRITICAL": 0.0},
  "feature_scores": {"sensitivity": 0, "threat_level": 0, "access_risk": 0, "external_sharing": 0, "personal_data": 0, "financial_data": 0, "confidential_keywords": 0, "file_size_mb_log": 0},
  "key_concerns": ["short phrase", "short phrase"],
  "reasoning": "2-4 sentence justification a security reviewer could read"
}

class_probabilities should sum to approximately 1.0. Never invent facts that are not supported by the evidence or excerpt; if the excerpt is empty or unreadable, say so in "reasoning" and default toward MEDIUM unless the PII-pattern evidence alone clearly indicates otherwise."""


class GroqUnavailableError(RuntimeError):
    """Raised whenever the Groq engine cannot produce a verdict, so the
    caller can cleanly fall back to the heuristic RandomForest model."""
    pass


class GroqRiskAssessor:
    """
    LLM-based risk assessor using Groq's OpenAI-compatible chat completion
    API. This is the PRIMARY risk-assessment engine of the framework.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or GROQ_MODEL_DEFAULT
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise GroqUnavailableError(
                "No Groq API key configured. Set the GROQ_API_KEY environment "
                "variable (or put it in a .env file next to this script) or "
                "pass --groq-api-key on the CLI."
            )
        try:
            from groq import Groq
        except ImportError as e:
            raise GroqUnavailableError(
                "The 'groq' package is not installed. Run: pip install groq"
            ) from e
        try:
            self._client = Groq(api_key=self.api_key)
        except Exception as e:
            raise GroqUnavailableError(f"Could not construct Groq client: {e}") from e
        return self._client

    def assess(self, file_name: str, auto: "AutoFeatures", raw_text: str) -> dict:
        """
        Sends the automated evidence + a bounded text excerpt to the Groq
        model and returns a normalized verdict dict. Raises
        GroqUnavailableError on any failure (missing key, network error,
        bad/unparseable response) so the caller can fall back cleanly.
        """
        client = self._get_client()
        excerpt = raw_text[:GROQ_MAX_TEXT_CHARS]
        truncated = len(raw_text) > GROQ_MAX_TEXT_CHARS

        user_payload = {
            "file_name": file_name,
            "automated_evidence": {
                "matched_confidential_keywords": auto.matched_indicators,
                "regex_evidence": auto.evidence,
            },
            "text_excerpt": excerpt,
            "text_excerpt_truncated": truncated,
        }

        t0 = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=900,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": RISK_ASSESSMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            )
        except Exception as e:
            raise GroqUnavailableError(f"Groq API call failed: {e}") from e
        latency = time.perf_counter() - t0

        raw_content = (completion.choices[0].message.content or "").strip()
        # Defensive cleanup in case the model wraps JSON in ```json fences
        # despite instructions not to.
        if raw_content.startswith("```"):
            raw_content = raw_content.strip("`")
            if raw_content.lower().startswith("json"):
                raw_content = raw_content[4:]
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise GroqUnavailableError(
                f"Groq returned non-JSON output that could not be parsed: {e}"
            ) from e

        risk_class = str(parsed.get("risk_class", "")).upper().strip()
        if risk_class not in RISK_CLASSES:
            raise GroqUnavailableError(
                f"Groq returned an unrecognised risk_class: {parsed.get('risk_class')!r}"
            )

        try:
            confidence = float(parsed.get("confidence", 0.75))
        except (TypeError, ValueError):
            confidence = 0.75
        confidence = max(0.0, min(1.0, confidence))

        proba_map = {c: (1.0 if c == risk_class else 0.0) for c in RISK_CLASSES}
        if isinstance(parsed.get("class_probabilities"), dict):
            for c in RISK_CLASSES:
                if c in parsed["class_probabilities"]:
                    try:
                        proba_map[c] = float(parsed["class_probabilities"][c])
                    except (TypeError, ValueError):
                        pass

        feature_scores = parsed.get("feature_scores", {})
        if not isinstance(feature_scores, dict):
            feature_scores = {}

        key_concerns = parsed.get("key_concerns", [])
        if not isinstance(key_concerns, list):
            key_concerns = []

        return {
            "engine": "groq",
            "model": self.model,
            "risk_class": risk_class,
            "confidence": confidence,
            "class_probabilities": proba_map,
            "reasoning": str(parsed.get("reasoning", "")),
            "key_concerns": [str(c) for c in key_concerns][:8],
            "feature_scores": feature_scores,
            "latency_s": latency,
        }


def assess_risk(file_path: Path, heuristic_model: "RiskModel",
                 overrides: Optional[dict] = None,
                 engine: str = "auto",
                 groq_assessor: Optional[GroqRiskAssessor] = None) -> dict:
    """
    Unified risk-assessment entry point used by the CLI and the web UI.

    engine: "groq" (force Groq, error if unavailable), "heuristic" (force
    the local RandomForest/policy path, never calls the network), or
    "auto" (default — try Groq, transparently fall back to heuristic on
    any failure).

    Always returns a dict with at least: engine_used, fallback_used,
    fallback_reason, risk_class, confidence, class_probabilities,
    reasoning, key_concerns, auto_features (dict), feature_scores.
    """
    raw_text = extract_text(file_path)
    auto = extract_auto_features(file_path, raw_text=raw_text)

    fallback_used = False
    fallback_reason = None

    def run_heuristic():
        fv = build_feature_vector(auto, overrides)
        risk_class, confidence, proba_map = heuristic_model.predict(fv)
        return {
            "engine": "heuristic_random_forest",
            "model": "RandomForestClassifier (synthetic policy labels)",
            "risk_class": risk_class,
            "confidence": confidence,
            "class_probabilities": proba_map,
            "reasoning": (
                "Produced by the local RandomForest classifier trained on a "
                "transparent weighted-policy formula (no network call)."
            ),
            "key_concerns": auto.matched_indicators[:8],
            "feature_scores": {
                "sensitivity": auto.sensitivity, "threat_level": auto.threat_level,
                "access_risk": auto.access_risk, "external_sharing": auto.external_sharing,
                "personal_data": auto.personal_data, "financial_data": auto.financial_data,
                "confidential_keywords": auto.confidential_keywords,
                "file_size_mb_log": auto.file_size_mb_log,
            },
            "latency_s": 0.0,
        }

    if engine == "heuristic":
        verdict = run_heuristic()
    elif engine == "groq":
        assessor = groq_assessor or GroqRiskAssessor()
        verdict = assessor.assess(file_path.name, auto, raw_text)
    else:  # "auto"
        assessor = groq_assessor or GroqRiskAssessor()
        try:
            verdict = assessor.assess(file_path.name, auto, raw_text)
        except GroqUnavailableError as e:
            fallback_used = True
            fallback_reason = str(e)
            if not heuristic_model.model and not MODEL_PATH.exists():
                # No local model trained either -> nothing we can do.
                raise
            verdict = run_heuristic()

    overrides = overrides or {}
    for k, v in overrides.items():
        if v is not None and k in verdict.get("feature_scores", {}):
            verdict["feature_scores"][k] = float(v)

    verdict["engine_used"] = verdict.pop("engine")
    verdict["fallback_used"] = fallback_used
    verdict["fallback_reason"] = fallback_reason
    verdict["auto_features"] = asdict(auto)
    return verdict


# --------------------------------------------------------------------------
# 5. SECURITY PROFILES
# --------------------------------------------------------------------------
#
#   Risk      Cipher               Key protection               Security
#   --------  -------------------  ---------------------------  -----------
#   LOW       ChaCha20-Poly1305    Conventional key             Basic
#   MEDIUM    AES-256-GCM          Conventional key management  Strong
#   HIGH      AES-256-GCM          ML-KEM-768  (PQC)            Very Strong
#   CRITICAL  AES-256-GCM          ML-KEM-1024 (PQC)            Maximum
#
# "Conventional key"            = a fresh random 256-bit key per file, kept
#                                 in the local key store.
# "Conventional key management" = envelope encryption: a fresh random file
#                                 key per file, wrapped (AES key wrap,
#                                 RFC 3394) under a master key. Only the
#                                 wrapped key is stored in the container.
# "ML-KEM-xxx (PQC)"            = the file key is established with ML-KEM
#                                 (FIPS 203) + HKDF-SHA256, so no key is
#                                 stored at all - the container carries the
#                                 KEM ciphertext.
# ChaCha20-Poly1305 is the authenticated form of ChaCha20 (AEAD): it
# encrypts AND detects tampering, same guarantee as AES-GCM.

CIPHER_CHACHA = "ChaCha20-Poly1305"
CIPHER_AES = "AES-256-GCM"

KEY_MODE_CONVENTIONAL = "CONVENTIONAL"
KEY_MODE_CONVENTIONAL_MANAGED = "CONVENTIONAL_MANAGED"
KEY_MODE_PQC = "PQC"


# Key-lifetime policy (crypto-period shrinks as risk rises - the direction
# recommended by NIST SP 800-57). The values below are DEFAULTS for this
# prototype; override per file with --ttl-seconds (0 = never expire).
DAY = 86400
KEY_LIFETIME_POLICY = {
    "LOW": None,            # never expires
    "MEDIUM": 90 * DAY,
    "HIGH": 30 * DAY,
    "CRITICAL": 7 * DAY,
}


# Attempt budget: how many WRONG passphrases a file tolerates before its key
# self-destructs. Higher risk => fewer guesses. None = unlimited (no passphrase).
MAX_ATTEMPTS_POLICY = {
    "LOW": None,
    "MEDIUM": 20,
    "HIGH": 10,
    "CRITICAL": 3,
}


def _fmt_duration(seconds: Optional[float]) -> str:
    if not seconds:
        return "No expiry"
    seconds = int(seconds)
    for unit, n in (("day", DAY), ("hour", 3600), ("minute", 60)):
        if seconds >= n and seconds % n == 0:
            v = seconds // n
            return f"{v} {unit}{'s' if v != 1 else ''}"
    return f"{seconds} seconds"


@dataclass(frozen=True)
class SecurityProfile:
    name: str
    cipher: str                 # symmetric AEAD used for the file data
    kem_algo: Optional[str]     # ML-KEM variant, or None for non-PQC profiles
    key_mode: str               # one of the KEY_MODE_* constants
    security_level: str
    description: str
    ttl_seconds: Optional[int] = None   # key lifetime policy; None = keys never expire
    max_attempts: Optional[int] = None  # wrong-passphrase budget; None = unlimited / no passphrase

    @property
    def key_lifetime(self) -> str:
        return _fmt_duration(self.ttl_seconds)

    @property
    def key_protection(self) -> str:
        if self.key_mode == KEY_MODE_PQC:
            return f"{self.kem_algo} (PQC)"
        if self.key_mode == KEY_MODE_CONVENTIONAL_MANAGED:
            return "Conventional key management"
        return "Conventional key"


SECURITY_PROFILES = {
    "LOW": SecurityProfile(
        "LOW", CIPHER_CHACHA, None, KEY_MODE_CONVENTIONAL, "Basic",
        "Lightweight and efficient protection for low-risk files.", KEY_LIFETIME_POLICY["LOW"], MAX_ATTEMPTS_POLICY["LOW"]),
    "MEDIUM": SecurityProfile(
        "MEDIUM", CIPHER_AES, None, KEY_MODE_CONVENTIONAL_MANAGED, "Strong",
        "Stronger confidentiality for moderately sensitive files.", KEY_LIFETIME_POLICY["MEDIUM"], MAX_ATTEMPTS_POLICY["MEDIUM"]),
    "HIGH": SecurityProfile(
        "HIGH", CIPHER_AES, "ML-KEM-768", KEY_MODE_PQC, "Very Strong",
        "Strong file encryption + quantum-resistant key establishment.", KEY_LIFETIME_POLICY["HIGH"], MAX_ATTEMPTS_POLICY["HIGH"]),
    "CRITICAL": SecurityProfile(
        "CRITICAL", CIPHER_AES, "ML-KEM-1024", KEY_MODE_PQC, "Maximum",
        "Strongest symmetric encryption + stronger PQC key establishment.", KEY_LIFETIME_POLICY["CRITICAL"], MAX_ATTEMPTS_POLICY["CRITICAL"]),
}


def select_profile(risk_class: str) -> SecurityProfile:
    if risk_class not in SECURITY_PROFILES:
        raise UnsupportedKEMError(f"Unknown risk class: {risk_class!r}")
    return SECURITY_PROFILES[risk_class]


def profile_to_dict(profile: SecurityProfile) -> dict:
    d = asdict(profile)
    d["key_protection"] = profile.key_protection
    return d


FIXED_BASELINE_PROFILE = SecurityProfile(
    "FIXED", CIPHER_AES, "ML-KEM-768", KEY_MODE_PQC, "Fixed",
    "Fixed non-adaptive baseline profile."
)


# --------------------------------------------------------------------------
# 6. ENCRYPTION ENGINE  (ChaCha20-Poly1305 / AES-256-GCM + key protection)
# --------------------------------------------------------------------------

from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.keywrap import (
    aes_key_wrap, aes_key_unwrap, InvalidUnwrap,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
import hashlib


class UnsupportedCipherError(ValueError):
    pass


class IntegrityError(Exception):
    pass


def _hkdf_derive_aes_key(shared_secret: bytes, info: bytes = b"adaptive-pqc-v1-aes-key",
                         salt: Optional[bytes] = None) -> bytes:
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=info)
    return hkdf.derive(shared_secret)


def _new_recovery_passphrase() -> str:
    """80-bit random passphrase, e.g. 3fa9-0c1d-77be-a210-9e4c."""
    return "-".join(secrets.token_hex(2) for _ in range(5))


def _derive_pw(passphrase: str, salt: bytes) -> tuple:
    """scrypt(passphrase) -> (pw_key, verifier). pw_key is mixed into the file key;
    verifier is what the key store compares against (independent of pw_key)."""
    import unicodedata
    pw = unicodedata.normalize("NFC", passphrase).encode("utf-8")
    out = hashlib.scrypt(pw, salt=salt, n=2 ** 14, r=8, p=1, maxmem=2 ** 26, dklen=64)
    return out[:32], out[32:]


def _mix_salt(shred_key: Optional[bytes], pw_key: Optional[bytes]) -> Optional[bytes]:
    if shred_key is None:
        return None
    if pw_key is None:
        return shred_key
    return hashlib.sha256(b"adaptive-pqc-v3-guard|" + shred_key + pw_key).digest()


def _derive_subkeys(data_key: bytes, aad: bytes, nonce: bytes) -> tuple:
    """
    Key-committing derivation (container v3).

    AES-GCM and ChaCha20-Poly1305 are NOT key-committing: one ciphertext can be
    crafted that authenticates under two different keys (the basis of
    partitioning-oracle attacks). To prevent that, the key material is expanded
    into TWO independent values:
        enc_key  - the AEAD key that actually encrypts the file
        commit   - a 256-bit commitment to (data_key, header, nonce)
    `commit` is stored in the container and checked, in constant time,
    BEFORE the AEAD runs. Two different data keys cannot share a commitment
    (HKDF-SHA256 is collision resistant), so a ciphertext can only ever be
    opened by the key it was committed to.
    """
    out = HKDF(algorithm=hashes.SHA256(), length=64, salt=None,
               info=b"adaptive-pqc-v3-subkeys|" + aad + b"|" + nonce).derive(data_key)
    return out[:32], out[32:]


def _make_aead(cipher: str, key: bytes):
    if cipher == CIPHER_AES:
        return AESGCM(key)
    if cipher == CIPHER_CHACHA:
        return ChaCha20Poly1305(key)
    raise UnsupportedCipherError(f"Unsupported cipher: {cipher!r}")


def _build_aad(version: int, cipher: str, key_mode: str, kem_algo: Optional[str],
               created_at: Optional[int] = None, expires_at: Optional[int] = None,
               shred_id: Optional[str] = None,
               max_attempts: Optional[int] = None) -> bytes:
    """
    Associated data authenticated by the AEAD. It binds the container
    header (cipher, key mode, KEM) to the ciphertext, so an attacker cannot
    flip a header field (e.g. downgrade HIGH to a conventional key) without
    decryption failing.
    """
    base = f"{version}|{cipher}|{key_mode}|{kem_algo or '-'}"
    if version >= 3:
        # expiry + shred id are authenticated too, so extending / removing an
        # expiry date in the container header makes decryption fail.
        base += f"|{created_at if created_at is not None else '-'}" \
                f"|{expires_at if expires_at is not None else '-'}|{shred_id or '-'}"
        if max_attempts:
            base += f"|g{max_attempts}"   # attempt budget is authenticated too
    return base.encode()


def _aad_from_container(c: dict) -> bytes:
    return _build_aad(c.get("version", 3), c["cipher"], c["key_mode"], c.get("kem_algo"),
                      c.get("created_at"), c.get("expires_at"), c.get("shred_id"),
                      c.get("max_attempts"))


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _coerce_profile(p) -> SecurityProfile:
    """Accepts a SecurityProfile, a risk-class name, or (legacy) a bare ML-KEM name."""
    if isinstance(p, SecurityProfile):
        return p
    if isinstance(p, str):
        if p in SECURITY_PROFILES:
            return SECURITY_PROFILES[p]
        if p.startswith("ML-KEM-"):
            return SecurityProfile(f"PQC-{p}", CIPHER_AES, p, KEY_MODE_PQC, "-",
                                    "Ad-hoc PQC profile.")
    raise UnsupportedKEMError(f"Cannot build a security profile from {p!r}")


class CryptoEngine:
    """
    Profile-driven encryption:

      LOW       random file key (stored)                 -> ChaCha20-Poly1305
      MEDIUM    random file key, wrapped by master key   -> AES-256-GCM
      HIGH      ML-KEM-768  -> HKDF-SHA256 -> file key   -> AES-256-GCM
      CRITICAL  ML-KEM-1024 -> HKDF-SHA256 -> file key   -> AES-256-GCM

    Container v3 adds two features on top of that:

      * KEY COMMITMENT   every container carries a 256-bit commitment to the
                         file key; it is verified (constant time) before the
                         AEAD runs, so a ciphertext cannot be opened by two keys.
      * KEY EXPIRY /     each risk tier has a key lifetime. When it runs out
        CRYPTO-SHREDDING the key material is destroyed (not just flagged), so
                         the file becomes permanently unreadable.
    """

    def __init__(self, key_manager: Optional[KeyManager] = None, clock=None,
                 apply_attempt_policy: bool = True):
        self.km = key_manager or KeyManager()
        self.clock = clock or time.time   # injectable so tests can fast-forward time
        # False = ignore the per-tier attempt budget unless one is passed explicitly
        self.apply_attempt_policy = apply_attempt_policy

    # ---------------------------------------------------------------- encrypt
    def encrypt_bytes(self, plaintext: bytes, profile, ttl_seconds: Optional[int] = None,
                      label: str = "", max_attempts: Optional[int] = None,
                      passphrase: Optional[str] = None) -> tuple[dict, dict]:
        """ttl_seconds:  None = profile policy, 0 = never expire, N = destroyed after N s.
        max_attempts: None = profile policy, 0 = unlimited, N = key self-destructs after
                      N wrong passphrases. A guarded file needs a passphrase; if none is
                      given a random one is generated and returned ONCE in the metrics
                      (key 'recovery_passphrase')."""
        profile = _coerce_profile(profile)
        t0 = time.perf_counter()

        ttl = profile.ttl_seconds if ttl_seconds is None else (ttl_seconds or None)
        if ttl is not None and ttl < 0:
            raise ValueError("ttl_seconds must be >= 0")
        if max_attempts is not None and max_attempts < 0:
            raise ValueError("max_attempts must be >= 0")
        if max_attempts is None:
            budget = profile.max_attempts if self.apply_attempt_policy else None
        else:
            budget = max_attempts or None
        now = int(self.clock())
        created_at = now if ttl else None
        expires_at = now + int(ttl) if ttl else None

        recovery_passphrase = None
        pw_key = pw_salt = verifier = None
        if budget:
            if not passphrase:
                passphrase = recovery_passphrase = _new_recovery_passphrase()
            pw_salt = os.urandom(16)
            pw_key, verifier = _derive_pw(passphrase, pw_salt)
        need_shred = bool(ttl) or bool(budget)

        container = {
            "version": CONTAINER_VERSION,
            "cipher": profile.cipher,
            "key_mode": profile.key_mode,
            "kem_algo": profile.kem_algo,
        }

        t_key_start = time.perf_counter()
        t_encaps = 0.0
        key_material_len = 0
        shred_id = None
        if profile.key_mode == KEY_MODE_PQC:
            ek = self.km.load_ek(profile.kem_algo)
            t_e = time.perf_counter()
            shared_secret, kem_ciphertext = kem_encaps(profile.kem_algo, ek)
            t_encaps = time.perf_counter() - t_e
            shred_key = None
            if need_shred:
                shred_id, shred_key = self.km.new_shred_key()
            data_key = _hkdf_derive_aes_key(shared_secret, salt=_mix_salt(shred_key, pw_key))
            container["kem_ciphertext"] = _b64e(kem_ciphertext)
            key_material_len = len(kem_ciphertext)
        elif profile.key_mode == KEY_MODE_CONVENTIONAL_MANAGED:
            data_key = os.urandom(32)
            wrap_key = self.km.master_kek()
            if need_shred:
                shred_id, shred_key = self.km.new_shred_key()
                wrap_key = _hkdf_derive_aes_key(wrap_key, info=b"adaptive-pqc-v3-wrap",
                                                salt=_mix_salt(shred_key, pw_key))
            wrapped = aes_key_wrap(wrap_key, data_key)
            container["wrapped_key"] = _b64e(wrapped)
            key_material_len = len(wrapped)
        elif profile.key_mode == KEY_MODE_CONVENTIONAL:
            key_id, data_key = self.km.new_conventional_key()
            container["key_id"] = key_id
            if need_shred:
                shred_id = key_id   # destroying the stored per-file key IS the shred
            if pw_key:
                data_key = _hkdf_derive_aes_key(data_key, info=b"adaptive-pqc-v3-pw", salt=pw_key)
        else:
            raise UnsupportedCipherError(f"Unknown key mode: {profile.key_mode!r}")
        key_setup_time = time.perf_counter() - t_key_start

        if shred_id:
            container["shred_id"] = shred_id
        if ttl:
            container.update({"created_at": created_at, "expires_at": expires_at})
        if budget:
            container.update({"max_attempts": int(budget), "pw_salt": _b64e(pw_salt)})

        nonce = os.urandom(12)
        aad = _build_aad(CONTAINER_VERSION, profile.cipher, profile.key_mode, profile.kem_algo,
                         created_at, expires_at, shred_id, budget)
        enc_key, commitment = _derive_subkeys(data_key, aad, nonce)
        ciphertext = _make_aead(profile.cipher, enc_key).encrypt(nonce, plaintext, aad)

        container.update({
            "nonce": _b64e(nonce),
            "ciphertext": _b64e(ciphertext),
            "aad": _b64e(aad),
            "key_commitment": _b64e(commitment),
            # NOTE: v3 deliberately stores NO plaintext hash. A cleartext
            # SHA-256 of the plaintext would survive crypto-shredding and let
            # anyone confirm guesses about the deleted content. The AEAD tag
            # already provides integrity.
        })

        if shred_id:
            self.km.register_expiry(shred_id, created_at or now, expires_at, label,
                                    profile.name, profile.key_mode, max_attempts=budget,
                                    pw_verifier=verifier, commitment=commitment)

        metrics = {
            "encryption_time_s": time.perf_counter() - t0,
            "key_setup_time_s": key_setup_time,
            "kem_encapsulation_time_s": t_encaps,   # 0.0 for non-PQC profiles
            "original_size_bytes": len(plaintext),
            "encrypted_size_bytes": (len(ciphertext) + len(nonce) + key_material_len
                                     + len(commitment) + (len(pw_salt) if pw_salt else 0)),
            "key_commitment_bytes": len(commitment),
            "key_lifetime": _fmt_duration(ttl),
            "created_at": created_at,
            "expires_at": expires_at,
            "expires_at_text": (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(expires_at))
                                if expires_at else None),
            "shred_id": shred_id,
            "max_attempts": int(budget) if budget else None,
        }
        if recovery_passphrase:
            metrics["recovery_passphrase"] = recovery_passphrase   # shown once, never stored
        return container, metrics

    # ---------------------------------------------------------------- decrypt
    def decrypt_bytes(self, container: dict, passphrase: Optional[str] = None) -> tuple[bytes, dict]:
        t0 = time.perf_counter()
        created_at = expires_at = shred_id = commitment = None
        max_attempts = pw_salt = None
        try:
            version = container.get("version", 1)
            if version == 1:
                # Legacy containers (before profiles): always ML-KEM + AES-256-GCM.
                cipher, key_mode, kem_algo = CIPHER_AES, KEY_MODE_PQC, container["kem_algo"]
                expected_aad = f"1|{kem_algo}".encode()
            elif version in (2, 3):
                cipher = container["cipher"]
                key_mode = container["key_mode"]
                kem_algo = container.get("kem_algo")
                if version == 2:
                    expected_aad = _build_aad(2, cipher, key_mode, kem_algo)
                else:
                    created_at = container.get("created_at")
                    expires_at = container.get("expires_at")
                    shred_id = container.get("shred_id")
                    max_attempts = container.get("max_attempts")
                    for v in (created_at, expires_at):
                        if v is not None and (not isinstance(v, int) or isinstance(v, bool)):
                            raise ValueError("created_at / expires_at must be integers")
                    if shred_id is not None and not isinstance(shred_id, str):
                        raise ValueError("shred_id must be a string")
                    if max_attempts is not None:
                        if (not isinstance(max_attempts, int) or isinstance(max_attempts, bool)
                                or max_attempts <= 0):
                            raise ValueError("max_attempts must be a positive integer")
                        pw_salt = base64.b64decode(container["pw_salt"])
                    expected_aad = _build_aad(3, cipher, key_mode, kem_algo,
                                              created_at, expires_at, shred_id, max_attempts)
                    if "key_commitment" not in container:
                        raise IntegrityError(
                            "v3 container has no key commitment - refusing to decrypt "
                            "(possible downgrade attack).")
                    commitment = base64.b64decode(container["key_commitment"])
            else:
                raise IntegrityError(f"Unsupported container version: {version!r}")

            nonce = base64.b64decode(container["nonce"])
            ciphertext = base64.b64decode(container["ciphertext"])
            aad = base64.b64decode(container["aad"])
        except IntegrityError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise IntegrityError(f"Malformed container: {type(e).__name__}: {e}") from e

        # Header must match what was authenticated at encryption time.
        if aad != expected_aad:
            raise IntegrityError(
                "Container header does not match its authenticated data - "
                "the file was modified or tampered with."
            )

        # ---- key expiry / crypto-shredding --------------------------------
        # The sweep uses ONLY the local ledger (trusted), never the container's
        # own claims - otherwise a crafted .pqc could make us destroy another
        # file's key.
        now = self.clock()
        self.km.sweep_expired(now)
        if expires_at is not None and now >= expires_at:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(expires_at))
            raise KeyShreddedError(
                f"This file's key lifetime ended at {when}. Its key material has been "
                "destroyed (crypto-shredded) - the data is permanently unrecoverable."
            )

        # ---- attempt budget / self-destruct --------------------------------
        pw_key = None
        entry = self.km.ledger().get(shred_id) if shred_id else None
        if entry and entry.get("status") == "SHREDDED":
            self.km._raise_missing(shred_id)
        if entry and entry.get("max_attempts") and not max_attempts:
            raise IntegrityError("Attempt guard missing from container header "
                                 "(possible downgrade attack).")
        if max_attempts:
            if not entry or not entry.get("max_attempts"):
                raise KeyStoreError("No local attempt-guard record for this file - it was "
                                    "encrypted with a different keys/ folder.")
            # Only the genuine container may spend this file's budget; a forged
            # container naming someone else's shred_id cannot burn it.
            if not hmac.compare_digest(base64.b64decode(entry["commitment"]), commitment):
                raise IntegrityError("This container does not match the guarded key record.")
            if not passphrase:
                raise PassphraseRequiredError(
                    f"This file is guarded: a passphrase is required. Only "
                    f"{entry['max_attempts'] - int(entry.get('failed_attempts', 0))} wrong "
                    "attempt(s) are allowed before its key is destroyed.")
            with self.km._LEDGER_LOCK:
                entry = self.km.ledger().get(shred_id)
                if not entry or entry.get("status") != "ACTIVE":
                    self.km._raise_missing(shred_id)
                pw_key, verifier = _derive_pw(passphrase, pw_salt)
                if hmac.compare_digest(verifier, base64.b64decode(entry["pw_verifier"])):
                    self.km.reset_attempts(shred_id)
                else:
                    left, destroyed = self.km.record_failed_attempt(shred_id, now)
                    if destroyed:
                        raise KeyShreddedError(
                            "Too many wrong passphrases. The attempt budget is used up and this "
                            "file's key has been destroyed - the data is permanently unrecoverable.")
                    raise WrongPassphraseError(
                        f"Wrong passphrase. {left} attempt(s) left before this file's key "
                        "is destroyed.", left)

        t_key_start = time.perf_counter()
        t_decaps = 0.0
        try:
            if key_mode == KEY_MODE_PQC:
                _get_kem(kem_algo)  # rejects unknown / attacker-chosen algorithm names
                dk = self.km.load_dk(kem_algo)
                kem_ciphertext = base64.b64decode(container["kem_ciphertext"])
                shred_key = self.km.load_shred_key(shred_id) if shred_id else None
                t_d = time.perf_counter()
                shared_secret = kem_decaps(kem_algo, dk, kem_ciphertext)
                t_decaps = time.perf_counter() - t_d
                data_key = _hkdf_derive_aes_key(shared_secret, salt=_mix_salt(shred_key, pw_key))
            elif key_mode == KEY_MODE_CONVENTIONAL_MANAGED:
                wrapped = base64.b64decode(container["wrapped_key"])
                wrap_key = self.km.master_kek(create=False)
                if shred_id:
                    wrap_key = _hkdf_derive_aes_key(
                        wrap_key, info=b"adaptive-pqc-v3-wrap",
                        salt=_mix_salt(self.km.load_shred_key(shred_id), pw_key))
                try:
                    data_key = aes_key_unwrap(wrap_key, wrapped)
                except InvalidUnwrap as e:
                    raise IntegrityError(
                        "Could not unwrap the file key - wrong master key or tampered container."
                    ) from e
            elif key_mode == KEY_MODE_CONVENTIONAL:
                if shred_id is not None and shred_id != container.get("key_id"):
                    raise IntegrityError("shred_id does not match key_id.")
                data_key = self.km.load_conventional_key(container.get("key_id"))
                if pw_key:
                    data_key = _hkdf_derive_aes_key(data_key, info=b"adaptive-pqc-v3-pw", salt=pw_key)
            else:
                raise IntegrityError(f"Unknown key mode in container: {key_mode!r}")
        except (KeyError, TypeError, ValueError, AssertionError) as e:
            if isinstance(e, UnsupportedKEMError):
                raise
            raise IntegrityError(f"Malformed container: {type(e).__name__}: {e}") from e
        key_recovery_time = time.perf_counter() - t_key_start

        # ---- key commitment (v3) ------------------------------------------
        commitment_verified = None
        if version == 3:
            aead_key, expected_commit = _derive_subkeys(data_key, aad, nonce)
            if not hmac.compare_digest(expected_commit, commitment):
                raise IntegrityError(
                    "Key commitment check failed - wrong key, or the container was modified.")
            commitment_verified = True
        else:
            aead_key = data_key

        try:
            plaintext = _make_aead(cipher, aead_key).decrypt(nonce, ciphertext, aad)
        except UnsupportedCipherError:
            raise
        except Exception as e:
            raise IntegrityError(f"Authenticated decryption failed: {e}") from e

        recomputed = hashlib.sha256(plaintext).hexdigest()
        if version == 3:
            integrity_ok = True    # AEAD tag + key commitment both verified
        else:
            integrity_ok = recomputed == container.get("sha256")
            if not integrity_ok:
                raise IntegrityError(
                    f"SHA-256 mismatch: expected {container.get('sha256')}, got {recomputed}"
                )

        metrics = {
            "decryption_time_s": time.perf_counter() - t0,
            "key_recovery_time_s": key_recovery_time,
            "kem_decapsulation_time_s": t_decaps,   # 0.0 for non-PQC profiles
            "integrity_ok": integrity_ok,
            "key_commitment_verified": commitment_verified,
            "sha256": recomputed,
            "cipher": cipher,
            "key_mode": key_mode,
            "kem_algo": kem_algo,
            "expires_at": expires_at,
            "max_attempts": max_attempts,
        }
        return plaintext, metrics

    # ------------------------------------------------------------------ files
    def encrypt_file(self, in_path: Path, out_path: Path, profile,
                     ttl_seconds: Optional[int] = None, max_attempts: Optional[int] = None,
                     passphrase: Optional[str] = None) -> dict:
        data = in_path.read_bytes()
        container, metrics = self.encrypt_bytes(data, profile, ttl_seconds=ttl_seconds,
                                                label=in_path.name, max_attempts=max_attempts,
                                                passphrase=passphrase)
        out_path.write_text(json.dumps(container))
        metrics["output_path"] = str(out_path)
        metrics["output_name"] = out_path.name   # bare filename for download links (works on Windows too)
        return metrics

    def decrypt_file(self, in_path: Path, out_path: Path, passphrase: Optional[str] = None) -> dict:
        try:
            container = json.loads(in_path.read_text())
        except (ValueError, UnicodeDecodeError) as e:
            raise IntegrityError(f"Not a valid .pqc container: {e}") from e
        if not isinstance(container, dict):
            raise IntegrityError("Not a valid .pqc container.")
        plaintext, metrics = self.decrypt_bytes(container, passphrase=passphrase)
        out_path.write_bytes(plaintext)
        metrics["output_path"] = str(out_path)
        metrics["output_name"] = out_path.name   # bare filename for download links (works on Windows too)
        return metrics


# --------------------------------------------------------------------------
# 7. END-TO-END PIPELINE HELPERS
# --------------------------------------------------------------------------

def run_full_pipeline(file_path: Path, risk_model: RiskModel, engine: CryptoEngine,
                       out_dir: Path = ENCRYPTED_DIR,
                       overrides: Optional[dict] = None,
                       risk_engine: str = "auto",
                       groq_assessor: Optional[GroqRiskAssessor] = None,
                       ttl_seconds: Optional[int] = None,
                       max_attempts: Optional[int] = None,
                       passphrase: Optional[str] = None) -> dict:
    """
    The only required input is `file_path`. All risk features are derived
    automatically from the file. Risk CLASSIFICATION is now performed by
    `assess_risk()`: Groq LLM by default (risk_engine="auto"/"groq"), with
    the local RandomForest as an automatic offline fallback. `overrides` is
    optional, for power users who want to manually correct one or more of
    the heuristic evidence scores.
    """
    verdict = assess_risk(file_path, risk_model, overrides=overrides,
                           engine=risk_engine, groq_assessor=groq_assessor)
    risk_class = verdict["risk_class"]
    profile = select_profile(risk_class)

    out_path = out_dir / (file_path.name + ".pqc")
    enc_metrics = engine.encrypt_file(file_path, out_path, profile, ttl_seconds=ttl_seconds,
                                      max_attempts=max_attempts, passphrase=passphrase)

    orig_size = file_path.stat().st_size
    expansion_pct = (
        ((enc_metrics["encrypted_size_bytes"] - orig_size) / orig_size * 100)
        if orig_size > 0 else float("nan")
    )

    return {
        "file": str(file_path),
        "auto_features": verdict["auto_features"],
        "risk_class": risk_class,
        "risk_confidence": verdict["confidence"],
        "risk_class_probabilities": verdict["class_probabilities"],
        "risk_engine_used": verdict["engine_used"],
        "risk_engine_model": verdict.get("model"),
        "risk_fallback_used": verdict["fallback_used"],
        "risk_fallback_reason": verdict["fallback_reason"],
        "risk_reasoning": verdict["reasoning"],
        "risk_key_concerns": verdict["key_concerns"],
        "risk_feature_scores": verdict["feature_scores"],
        "selected_profile": profile_to_dict(profile),
        "encryption_metrics": enc_metrics,
        "ciphertext_expansion_pct": expansion_pct,
    }


# --------------------------------------------------------------------------
# 8. BENCHMARKING
# --------------------------------------------------------------------------

def _make_test_bytes(size_bytes: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=size_bytes, dtype=np.uint8).tobytes()


def run_benchmark(sizes_kb=(1, 10, 100, 1024, 5120, 10240), reps: int = 5, profiles=None):
    """
    Benchmarks every security profile (LOW / MEDIUM / HIGH / CRITICAL) x
    file size, `reps` times each. Reports mean and standard deviation.
    No fabricated numbers: if the KEM backend is unavailable, this raises
    rather than inventing data.

    Runs against a throw-away key store so benchmarking never fills your
    real keys/ folder with test keys.
    """
    import psutil
    import tempfile
    process = psutil.Process(os.getpid())
    profiles = list(profiles) if profiles else [SECURITY_PROFILES[c] for c in RISK_CLASSES]

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        engine = CryptoEngine(KeyManager(Path(tmp)))
        engine.km.ensure_all()

        for profile in profiles:
            for size_kb in sizes_kb:
                size_bytes = size_kb * 1024
                enc_times, dec_times, encaps_times, decaps_times = [], [], [], []
                key_setup_times, enc_sizes, mem_deltas = [], [], []
                for r in range(reps):
                    data = _make_test_bytes(size_bytes, seed=RANDOM_SEED + r)
                    mem_before = process.memory_info().rss
                    container, enc_metrics = engine.encrypt_bytes(data, profile)
                    mem_after = process.memory_info().rss
                    mem_deltas.append(mem_after - mem_before)

                    plaintext, dec_metrics = engine.decrypt_bytes(
                        container, passphrase=enc_metrics.get("recovery_passphrase"))
                    assert plaintext == data, "roundtrip mismatch during benchmark"

                    enc_times.append(enc_metrics["encryption_time_s"])
                    dec_times.append(dec_metrics["decryption_time_s"])
                    encaps_times.append(enc_metrics["kem_encapsulation_time_s"])
                    decaps_times.append(dec_metrics["kem_decapsulation_time_s"])
                    key_setup_times.append(enc_metrics["key_setup_time_s"])
                    enc_sizes.append(enc_metrics["encrypted_size_bytes"])

                expansion_pct = (statistics.mean(enc_sizes) - size_bytes) / size_bytes * 100 if size_bytes else float("nan")
                rows.append({
                    "profile": profile.name,
                    "cipher": profile.cipher,
                    "key_protection": profile.key_protection,
                    "kem_algo": profile.kem_algo or "none",
                    "file_size_kb": size_kb,
                    "reps": reps,
                    "enc_time_mean_s": statistics.mean(enc_times),
                    "enc_time_std_s": statistics.pstdev(enc_times) if reps > 1 else 0.0,
                    "dec_time_mean_s": statistics.mean(dec_times),
                    "dec_time_std_s": statistics.pstdev(dec_times) if reps > 1 else 0.0,
                    "key_setup_time_mean_s": statistics.mean(key_setup_times),
                    "kem_encaps_time_mean_s": statistics.mean(encaps_times),
                    "kem_decaps_time_mean_s": statistics.mean(decaps_times),
                    "original_size_bytes": size_bytes,
                    "encrypted_size_bytes_mean": statistics.mean(enc_sizes),
                    "ciphertext_expansion_pct": expansion_pct,
                    "memory_delta_bytes_mean": statistics.mean(mem_deltas),
                })
                print(f"[benchmark] {profile.name:8s} {profile.cipher:18s} "
                      f"{(profile.kem_algo or 'no-KEM'):12s} {size_kb:6d} KB  "
                      f"enc={rows[-1]['enc_time_mean_s']*1000:.3f}ms  "
                      f"dec={rows[-1]['dec_time_mean_s']*1000:.3f}ms")

    csv_path = RESULTS_DIR / "benchmark_results.csv"
    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSaved: {csv_path}\nSaved: {json_path}")
    return rows


# --------------------------------------------------------------------------
# 9. SELF-TEST / SECURITY VALIDATION (lightweight, no pytest dependency)
# --------------------------------------------------------------------------

def run_selftest() -> dict:
    import tempfile
    results = {}

    def record(name, ok, detail=""):
        results[name] = {"passed": ok, "detail": detail}
        print(f"[selftest] {'PASS' if ok else 'FAIL'}  {name}  {detail}")

    def check(name, fn):
        try:
            out = fn()
            ok, detail = out if isinstance(out, tuple) else (bool(out), "")
            record(name, ok, detail)
        except Exception as e:
            record(name, False, f"{type(e).__name__}: {e}")

    def raises(exc_types, fn) -> bool:
        try:
            fn()
        except exc_types:
            return True
        return False

    # Two independent key stores in throw-away folders: `engine` encrypts,
    # `other` plays "a different machine / wrong keys". Your real keys/
    # folder is never touched by the self-test.
    with tempfile.TemporaryDirectory() as tmp_a, \
         tempfile.TemporaryDirectory() as tmp_b, \
         tempfile.TemporaryDirectory() as tmp_files:
        engine = CryptoEngine(KeyManager(Path(tmp_a)), apply_attempt_policy=False)
        engine.km.ensure_all()
        other = CryptoEngine(KeyManager(Path(tmp_b)), apply_attempt_policy=False)
        other.km.ensure_all()
        profiles = [SECURITY_PROFILES[c] for c in RISK_CLASSES]
        LOW, MEDIUM, HIGH, CRITICAL = profiles

        def flip_ciphertext(c):
            raw = bytearray(base64.b64decode(c["ciphertext"]))
            raw[0] ^= 0xFF
            c = dict(c)
            c["ciphertext"] = _b64e(bytes(raw))
            return c

        # 1. Risk class -> profile mapping matches the project design table
        def t_mapping():
            expected = {
                "LOW": ("ChaCha20-Poly1305", KEY_MODE_CONVENTIONAL, None),
                "MEDIUM": ("AES-256-GCM", KEY_MODE_CONVENTIONAL_MANAGED, None),
                "HIGH": ("AES-256-GCM", KEY_MODE_PQC, "ML-KEM-768"),
                "CRITICAL": ("AES-256-GCM", KEY_MODE_PQC, "ML-KEM-1024"),
            }
            got = {k: (select_profile(k).cipher, select_profile(k).key_mode,
                       select_profile(k).kem_algo) for k in expected}
            return got == expected, ""
        check("profile_mapping_matches_design", t_mapping)

        # 2. Roundtrip + correct container shape for every profile
        def t_roundtrip():
            data = b"hello adaptive pqc framework " * 100
            for p in profiles:
                c, _ = engine.encrypt_bytes(data, p)
                recovered, m = engine.decrypt_bytes(c)
                if recovered != data or not m["integrity_ok"]:
                    return False, f"{p.name}: roundtrip failed"
                if c["cipher"] != p.cipher or c["key_mode"] != p.key_mode:
                    return False, f"{p.name}: wrong header"
                if ("kem_ciphertext" in c) != (p.key_mode == KEY_MODE_PQC):
                    return False, f"{p.name}: unexpected KEM material"
                if ("wrapped_key" in c) != (p.key_mode == KEY_MODE_CONVENTIONAL_MANAGED):
                    return False, f"{p.name}: unexpected wrapped key"
                if ("key_id" in c) != (p.key_mode == KEY_MODE_CONVENTIONAL):
                    return False, f"{p.name}: unexpected key_id"
            return True, ""
        check("roundtrip_all_profiles", t_roundtrip)

        # 3. Tampered ciphertext is rejected (every profile)
        def t_tamper():
            for p in profiles:
                c, _ = engine.encrypt_bytes(b"tamper test payload", p)
                if not raises(IntegrityError, lambda: engine.decrypt_bytes(flip_ciphertext(c))):
                    return False, f"{p.name}: tampered ciphertext accepted"
            return True, ""
        check("tampered_ciphertext_rejected", t_tamper)

        # 4. Header tampering / downgrade is rejected
        def t_header():
            for p in profiles:
                c, _ = engine.encrypt_bytes(b"header test", p)
                c2 = dict(c)
                c2["cipher"] = CIPHER_CHACHA if p.cipher == CIPHER_AES else CIPHER_AES
                if not raises(IntegrityError, lambda: engine.decrypt_bytes(c2)):
                    return False, f"{p.name}: cipher swap accepted"
            c, _ = engine.encrypt_bytes(b"downgrade test", HIGH)
            c2 = dict(c)
            c2["key_mode"] = KEY_MODE_CONVENTIONAL
            if not raises(IntegrityError, lambda: engine.decrypt_bytes(c2)):
                return False, "HIGH->conventional downgrade accepted"
            c, _ = engine.encrypt_bytes(b"kem swap test", CRITICAL)
            c2 = dict(c)
            c2["kem_algo"] = "ML-KEM-768"
            if not raises(IntegrityError, lambda: engine.decrypt_bytes(c2)):
                return False, "CRITICAL->ML-KEM-768 swap accepted"
            return True, ""
        check("header_tamper_and_downgrade_rejected", t_header)

        # 5. Wrong stored AAD is rejected
        def t_aad():
            for p in profiles:
                c, _ = engine.encrypt_bytes(b"aad test payload", p)
                c2 = dict(c)
                c2["aad"] = _b64e(b"tampered-aad")
                if not raises(IntegrityError, lambda: engine.decrypt_bytes(c2)):
                    return False, f"{p.name}: wrong AAD accepted"
            return True, ""
        check("wrong_aad_rejected", t_aad)

        # 6. Wrong keys: a different key store cannot decrypt
        def t_wrong_keys():
            data = b"wrong key test"
            for p in (HIGH, CRITICAL):  # foreign store: no shred key / wrong ML-KEM key
                c, _ = engine.encrypt_bytes(data, p)
                if not raises((IntegrityError, KeyStoreError), lambda: other.decrypt_bytes(c)):
                    return False, f"{p.name}: foreign ML-KEM key accepted"
            c, _ = engine.encrypt_bytes(data, MEDIUM)  # different master key
            if not raises((IntegrityError, KeyStoreError), lambda: other.decrypt_bytes(c)):
                return False, "MEDIUM: foreign master key accepted"
            c, _ = engine.encrypt_bytes(data, LOW)  # key_id unknown to other store
            if not raises(KeyStoreError, lambda: other.decrypt_bytes(c)):
                return False, "LOW: foreign key store accepted"
            return True, ""
        check("wrong_keys_rejected", t_wrong_keys)

        # 7. Crafted key_id / kem_algo in an untrusted container cannot escape keys/
        def t_untrusted_ids():
            c, _ = engine.encrypt_bytes(b"x", LOW)
            for bad in ("../../etc/passwd", "..\\..\\secret", "A" * 32, "", None):
                c2 = dict(c)
                c2["key_id"] = bad
                if not raises(KeyStoreError, lambda: engine.decrypt_bytes(c2)):
                    return False, f"key_id {bad!r} not rejected"
            c, _ = engine.encrypt_bytes(b"x", HIGH)
            c2 = dict(c)
            c2["kem_algo"] = "..\\..\\evil"
            c2["aad"] = _b64e(_aad_from_container(c2))
            if not raises(UnsupportedKEMError, lambda: engine.decrypt_bytes(c2)):
                return False, "crafted kem_algo not rejected"
            return True, ""
        check("untrusted_container_ids_rejected", t_untrusted_ids)

        # 8. Unsupported algorithms -> clean errors
        def t_unsupported():
            ok1 = raises(UnsupportedKEMError, lambda: _get_kem("ML-KEM-9999"))
            ok2 = raises(UnsupportedKEMError, lambda: engine.encrypt_bytes(b"x", "ML-KEM-9999"))
            ok3 = raises(UnsupportedCipherError, lambda: _make_aead("ROT13", b"\0" * 32))
            return ok1 and ok2 and ok3, ""
        check("unsupported_algorithms_clean_error", t_unsupported)

        # 9. Empty file, every profile
        def t_empty():
            for p in profiles:
                c, _ = engine.encrypt_bytes(b"", p)
                recovered, m = engine.decrypt_bytes(c)
                if recovered != b"" or not m["integrity_ok"]:
                    return False, p.name
            return True, ""
        check("empty_file_roundtrip", t_empty)

        # 10. 1 MB binary payload, every profile
        def t_large():
            data = _make_test_bytes(1024 * 1024, seed=7)
            for p in profiles:
                c, _ = engine.encrypt_bytes(data, p)
                recovered, m = engine.decrypt_bytes(c)
                if recovered != data or not m["integrity_ok"]:
                    return False, p.name
            return True, ""
        check("large_binary_roundtrip_1mb", t_large)

        # 11. Real file on disk: encrypt_file -> .pqc -> decrypt_file
        def t_files():
            src = Path(tmp_files) / "sample.txt"
            src.write_bytes(b"file on disk\n" * 50)
            for p in profiles:
                enc = Path(tmp_files) / f"sample_{p.name}.txt.pqc"
                dec = Path(tmp_files) / f"sample_{p.name}.txt"
                engine.encrypt_file(src, enc, p)
                engine.decrypt_file(enc, dec)
                if dec.read_bytes() != src.read_bytes():
                    return False, p.name
            return True, ""
        check("file_roundtrip_all_profiles", t_files)

        # 12. Containers written by the previous (v1, ML-KEM-only) version still decrypt
        def t_legacy():
            data = b"legacy v1 container"
            algo = "ML-KEM-768"
            shared_secret, kem_ct = kem_encaps(algo, engine.km.load_ek(algo))
            nonce = os.urandom(12)
            aad = f"1|{algo}".encode()
            ct = AESGCM(_hkdf_derive_aes_key(shared_secret)).encrypt(nonce, data, aad)
            c = {"version": 1, "kem_algo": algo, "aes_algo": "AES-256-GCM",
                 "nonce": _b64e(nonce), "kem_ciphertext": _b64e(kem_ct),
                 "ciphertext": _b64e(ct), "aad": _b64e(aad),
                 "sha256": hashlib.sha256(data).hexdigest()}
            recovered, m = engine.decrypt_bytes(c)
            return recovered == data and m["integrity_ok"], ""
        check("legacy_v1_container_still_decrypts", t_legacy)


        # ------------------------------------------------------------------
        # NEW: key commitment (container v3)
        # ------------------------------------------------------------------
        class FakeClock:
            def __init__(self, t): self.t = float(t)
            def __call__(self): return self.t

        def _gf_mul(x, y):
            R = 0xE1 << 120
            z, v = 0, y
            for i in range(127, -1, -1):
                if (x >> i) & 1:
                    z ^= v
                v = (v >> 1) ^ R if v & 1 else v >> 1
            return z

        def _gf_inv(x):
            r, b, e = 1 << 127, x, (1 << 128) - 2   # 1<<127 is the field's multiplicative identity (GCM bit order)
            while e:
                if e & 1:
                    r = _gf_mul(r, b)
                b = _gf_mul(b, b)
                e >>= 1
            return r

        def _ghash(h, aad_b, blocks):
            def pad(b): return b + b"\0" * (-len(b) % 16)
            y = 0
            data = pad(aad_b) + b"".join(blocks)
            for i in range(0, len(data), 16):
                y = _gf_mul(y ^ int.from_bytes(data[i:i + 16], "big"), h)
            y = _gf_mul(y ^ ((len(aad_b) * 8) << 64 | (len(blocks) * 128)), h)
            return y

        def _gcm_two_key_ciphertext(k1, k2, nonce, aad_b):
            """Craft ONE AES-GCM ciphertext+tag that authenticates under BOTH keys
            (the 'key multiplication' attack behind partitioning-oracle attacks)."""
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            def ecb(k, blk):
                e = Cipher(algorithms.AES(k), modes.ECB()).encryptor()
                return e.update(blk) + e.finalize()
            h1 = int.from_bytes(ecb(k1, bytes(16)), "big")
            h2 = int.from_bytes(ecb(k2, bytes(16)), "big")
            j0 = nonce + b"\0\0\0\1"
            e1 = int.from_bytes(ecb(k1, j0), "big")
            e2 = int.from_bytes(ecb(k2, j0), "big")
            c1 = os.urandom(16)
            def diff(x):
                blocks = [c1, x.to_bytes(16, "big")]
                return _ghash(h1, aad_b, blocks) ^ e1 ^ _ghash(h2, aad_b, blocks) ^ e2
            coef = _gf_mul(h1, h1) ^ _gf_mul(h2, h2)       # difference is affine in x
            x = _gf_mul(diff(0), _gf_inv(coef))
            blocks = [c1, x.to_bytes(16, "big")]
            tag = _ghash(h1, aad_b, blocks) ^ e1
            return c1 + x.to_bytes(16, "big") + tag.to_bytes(16, "big")

        def t_multikey():
            k1, k2 = os.urandom(32), os.urandom(32)
            nonce, aad_b = os.urandom(12), b"3|AES-256-GCM|header"
            # (a) PLAIN AES-GCM is NOT key-committing: one ciphertext opens under both keys.
            ct = _gcm_two_key_ciphertext(k1, k2, nonce, aad_b)
            try:
                AESGCM(k1).decrypt(nonce, ct, aad_b)
                AESGCM(k2).decrypt(nonce, ct, aad_b)
            except Exception as e:
                return False, f"attack construction failed: {e}"
            # (b) With the v3 commitment the two keys can never share a commitment,
            #     so the same trick is caught before the AEAD is even run.
            _, commit1 = _derive_subkeys(k1, aad_b, nonce)
            _, commit2 = _derive_subkeys(k2, aad_b, nonce)
            if hmac.compare_digest(commit1, commit2):
                return False, "two different keys produced the same commitment"
            return True, "plain GCM accepted the crafted ciphertext under 2 keys; commitment separates them"
        check("key_commitment_blocks_multi_key_ciphertext", t_multikey)

        def t_v3_shape():
            c, m = engine.encrypt_bytes(b"shape test", HIGH)
            if c["version"] != 3 or "key_commitment" not in c:
                return False, "no key commitment in v3 container"
            if len(base64.b64decode(c["key_commitment"])) != 32:
                return False, "commitment is not 256-bit"
            if "sha256" in c:
                return False, "plaintext hash leaked in container"
            return True, ""
        check("v3_container_has_commitment_and_no_plaintext_hash", t_v3_shape)

        def t_commit_tamper():
            for p in profiles:
                c, _ = engine.encrypt_bytes(b"commit tamper", p)
                c2 = dict(c); raw = bytearray(base64.b64decode(c["key_commitment"])); raw[0] ^= 1
                c2["key_commitment"] = _b64e(bytes(raw))
                if not raises(IntegrityError, lambda: engine.decrypt_bytes(c2)):
                    return False, f"{p.name}: flipped commitment accepted"
                c3 = {k: v for k, v in c.items() if k != "key_commitment"}
                if not raises(IntegrityError, lambda: engine.decrypt_bytes(c3)):
                    return False, f"{p.name}: missing commitment accepted (downgrade)"
            return True, ""
        check("commitment_tamper_and_removal_rejected", t_commit_tamper)

        # ------------------------------------------------------------------
        # NEW: risk-driven key expiry + crypto-shredding (fake clock, no sleeping)
        # ------------------------------------------------------------------
        T0 = 1_800_000_000
        clock = FakeClock(T0)
        tdir = Path(tmp_files) / "ks_time"
        teng = CryptoEngine(KeyManager(tdir), clock=clock, apply_attempt_policy=False)
        teng.km.ensure_all()

        def key_files_exist(c):
            kid = c.get("shred_id") or c.get("key_id")
            return ((teng.km.keys_dir / "shred" / f"{kid}.key").exists() or
                    (teng.km.keys_dir / "conventional" / f"{kid}.key").exists())

        def t_policy():
            if not (KEY_LIFETIME_POLICY["MEDIUM"] > KEY_LIFETIME_POLICY["HIGH"]
                    > KEY_LIFETIME_POLICY["CRITICAL"] > 0 and KEY_LIFETIME_POLICY["LOW"] is None):
                return False, "policy is not monotone (higher risk => shorter lifetime)"
            for name, prof in zip(RISK_CLASSES, profiles):
                c, m = teng.encrypt_bytes(b"policy", prof)
                ttl = KEY_LIFETIME_POLICY[name]
                if ttl is None:
                    if "expires_at" in c or "shred_id" in c:
                        return False, f"{name}: should not expire"
                else:
                    if c.get("expires_at") != T0 + ttl:
                        return False, f"{name}: wrong expiry"
                    if teng.km.ledger().get(c["shred_id"], {}).get("status") != "ACTIVE":
                        return False, f"{name}: not in ledger"
            return True, ""
        check("key_lifetime_policy_follows_risk_tier", t_policy)

        def t_shred_lifecycle():
            data = b"top secret payload " * 20
            for prof in profiles:
                clock.t = T0
                c, _ = teng.encrypt_bytes(data, prof, ttl_seconds=100)
                clock.t = T0 + 50
                rec, m = teng.decrypt_bytes(c)
                if rec != data or not m["key_commitment_verified"]:
                    return False, f"{prof.name}: not readable before expiry"
                clock.t = T0 + 101
                if not raises(KeyShreddedError, lambda: teng.decrypt_bytes(c)):
                    return False, f"{prof.name}: readable after expiry"
                if key_files_exist(c):
                    return False, f"{prof.name}: key material still on disk after expiry"
                # the long-term keys are still there - shredding is per-file, not global
                if prof.kem_algo and not teng.km._paths(prof.kem_algo)[1].exists():
                    return False, "ML-KEM private key was destroyed"
                if prof.key_mode == KEY_MODE_CONVENTIONAL_MANAGED and not (tdir / "master_kek.bin").exists():
                    return False, "master key was destroyed"
                # Setting the clock BACK must not resurrect the file.
                clock.t = T0 + 10
                if not raises(KeyShreddedError, lambda: teng.decrypt_bytes(c)):
                    return False, f"{prof.name}: clock rollback resurrected the file"
            return True, ""
        check("expired_key_is_destroyed_and_file_unrecoverable", t_shred_lifecycle)

        def t_sweep():
            clock.t = T0
            c1, _ = teng.encrypt_bytes(b"a", MEDIUM, ttl_seconds=10)
            c2, _ = teng.encrypt_bytes(b"b", LOW, ttl_seconds=10)
            c3, _ = teng.encrypt_bytes(b"c", HIGH, ttl_seconds=10_000)
            clock.t = T0 + 11
            done = teng.km.sweep_expired(clock.t)
            ok = (set(done) == {c1["shred_id"], c2["shred_id"]}
                  and not key_files_exist(c1) and not key_files_exist(c2) and key_files_exist(c3))
            clock.t = T0
            if ok and teng.decrypt_bytes(c3)[0] != b"c":
                ok = False
            return ok, f"swept {len(done)} expired key(s), kept the unexpired one"
        check("sweep_destroys_only_expired_keys", t_sweep)

        def t_expiry_tamper():
            clock.t = T0
            c, _ = teng.encrypt_bytes(b"expiry tamper", HIGH, ttl_seconds=100)
            a = dict(c); a["expires_at"] = c["expires_at"] + 10**7           # header only
            b = dict(a); b["aad"] = _b64e(_aad_from_container(b))             # header + aad
            d = {k: v for k, v in c.items() if k not in ("expires_at", "created_at", "shred_id")}
            d["aad"] = _b64e(_aad_from_container(d))                          # strip expiry entirely
            for name, cc in (("extend", a), ("extend+aad", b), ("strip", d)):
                if not raises(IntegrityError, lambda: teng.decrypt_bytes(cc)):
                    return False, f"expiry {name} attack accepted"
            return teng.decrypt_bytes(c)[0] == b"expiry tamper", ""
        check("expiry_extension_or_removal_rejected", t_expiry_tamper)

        def t_forged_shred():
            clock.t = T0
            victim, _ = teng.encrypt_bytes(b"victim data", MEDIUM, ttl_seconds=10_000)
            c, _ = teng.encrypt_bytes(b"decoy", HIGH, ttl_seconds=10_000)
            f = dict(c); f["shred_id"] = victim["shred_id"]
            f["created_at"] = T0 - 100; f["expires_at"] = T0 - 1
            f["aad"] = _b64e(_aad_from_container(f))
            if not raises(KeyStoreError, lambda: teng.decrypt_bytes(f)):
                return False, "forged expired container was processed"
            if not key_files_exist(victim) or teng.decrypt_bytes(victim)[0] != b"victim data":
                return False, "forged container destroyed another file's key"
            return True, ""
        check("forged_container_cannot_shred_other_files", t_forged_shred)

        def t_ttl_override():
            clock.t = T0
            c, _ = teng.encrypt_bytes(b"x", CRITICAL, ttl_seconds=0)
            if "expires_at" in c or "shred_id" in c:
                return False, "ttl=0 should disable expiry"
            c, _ = teng.encrypt_bytes(b"x", LOW, ttl_seconds=60)
            if c.get("shred_id") != c.get("key_id") or c.get("expires_at") != T0 + 60:
                return False, "LOW ttl override not applied"
            if not raises(ValueError, lambda: teng.encrypt_bytes(b"x", LOW, ttl_seconds=-5)):
                return False, "negative ttl accepted"
            return True, ""
        check("ttl_override_and_validation", t_ttl_override)

        def t_legacy_v2():
            key_id, key = engine.km.new_conventional_key()
            data, nonce = b"legacy v2 container", os.urandom(12)
            aad = _build_aad(2, CIPHER_CHACHA, KEY_MODE_CONVENTIONAL, None)
            c = {"version": 2, "cipher": CIPHER_CHACHA, "key_mode": KEY_MODE_CONVENTIONAL,
                 "kem_algo": None, "key_id": key_id, "nonce": _b64e(nonce),
                 "ciphertext": _b64e(ChaCha20Poly1305(key).encrypt(nonce, data, aad)),
                 "aad": _b64e(aad), "sha256": hashlib.sha256(data).hexdigest()}
            rec, m = engine.decrypt_bytes(c)
            return rec == data and m["integrity_ok"], ""
        check("legacy_v2_container_still_decrypts", t_legacy_v2)


        # ------------------------------------------------------------------
        # NEW: risk-scaled self-destruct (attempt budget + passphrase gate)
        # ------------------------------------------------------------------
        gclock = FakeClock(T0)
        gdir = Path(tmp_files) / "ks_guard"
        geng = CryptoEngine(KeyManager(gdir), clock=gclock)   # per-tier budgets ON
        geng.km.ensure_all()

        def burn(c, limit=60):
            """Guess wrong passphrases until the key self-destructs; return how many it took."""
            for i in range(1, limit + 1):
                try:
                    geng.decrypt_bytes(c, passphrase=f"wrong-guess-{i}")
                except WrongPassphraseError:
                    continue
                except KeyShreddedError:
                    return i
            return None

        def t_budget_policy():
            m = MAX_ATTEMPTS_POLICY
            ok = (m["LOW"] is None and m["MEDIUM"] > m["HIGH"] > m["CRITICAL"] > 0)
            return ok, "LOW=unlimited MEDIUM=%s HIGH=%s CRITICAL=%s" % (m["MEDIUM"], m["HIGH"], m["CRITICAL"])
        check("attempt_budget_policy_follows_risk_tier", t_budget_policy)

        def t_guard_roundtrip():
            for prof in (MEDIUM, HIGH, CRITICAL):
                c, m = geng.encrypt_bytes(b"guarded payload", prof)
                pw = m.get("recovery_passphrase")
                if not pw or c.get("max_attempts") != MAX_ATTEMPTS_POLICY[prof.name] or "pw_salt" not in c:
                    return False, f"{prof.name}: guard not applied"
                if not raises(PassphraseRequiredError, lambda: geng.decrypt_bytes(c)):
                    return False, f"{prof.name}: opened without a passphrase"
                if geng.km.ledger()[c["shred_id"]]["failed_attempts"] != 0:
                    return False, f"{prof.name}: a missing passphrase was counted as an attempt"
                rec, dm = geng.decrypt_bytes(c, passphrase=pw)
                if rec != b"guarded payload" or not dm["key_commitment_verified"]:
                    return False, f"{prof.name}: correct passphrase failed"
            c, m = geng.encrypt_bytes(b"low", LOW)            # LOW: unguarded, no passphrase
            if "max_attempts" in c or "recovery_passphrase" in m:
                return False, "LOW should not be guarded"
            return geng.decrypt_bytes(c)[0] == b"low", ""
        check("guarded_tiers_need_passphrase_low_does_not", t_guard_roundtrip)

        def t_self_destruct():
            counts = {}
            for prof in (MEDIUM, HIGH, CRITICAL):
                c, m = geng.encrypt_bytes(b"self destruct " + prof.name.encode(), prof)
                n = burn(c)
                counts[prof.name] = n
                if n != MAX_ATTEMPTS_POLICY[prof.name]:
                    return False, f"{prof.name}: destroyed after {n} guesses, expected {MAX_ATTEMPTS_POLICY[prof.name]}"
                kid = c["shred_id"]
                if (gdir / "shred" / f"{kid}.key").exists():
                    return False, f"{prof.name}: key still on disk"
                # even the CORRECT passphrase is useless now
                if not raises(KeyShreddedError, lambda: geng.decrypt_bytes(c, passphrase=m["recovery_passphrase"])):
                    return False, f"{prof.name}: correct passphrase worked after self-destruct"
                if prof.kem_algo and not geng.km._paths(prof.kem_algo)[1].exists():
                    return False, "ML-KEM private key was destroyed (should be per-file only)"
            return True, "guesses allowed: " + ", ".join(f"{k}={v}" for k, v in counts.items())
        check("wrong_guesses_destroy_key_at_tier_budget", t_self_destruct)

        def t_reset():
            c, m = geng.encrypt_bytes(b"reset", CRITICAL)
            pw = m["recovery_passphrase"]
            for _ in range(2):
                if not raises(WrongPassphraseError, lambda: geng.decrypt_bytes(c, passphrase="nope")):
                    return False, "wrong passphrase not rejected"
            geng.decrypt_bytes(c, passphrase=pw)                 # success resets the counter
            for _ in range(2):
                if not raises(WrongPassphraseError, lambda: geng.decrypt_bytes(c, passphrase="nope")):
                    return False, "budget not reset after a correct passphrase"
            return geng.decrypt_bytes(c, passphrase=pw)[0] == b"reset", ""
        check("correct_passphrase_resets_attempt_counter", t_reset)

        def t_guard_tamper():
            c, m = geng.encrypt_bytes(b"guard tamper", CRITICAL)
            pw = m["recovery_passphrase"]
            a = dict(c); a["max_attempts"] = 1000                              # raise budget
            b = dict(a); b["aad"] = _b64e(_aad_from_container(b))
            d = {k: v for k, v in c.items() if k not in ("max_attempts", "pw_salt")}
            d["aad"] = _b64e(_aad_from_container(d))                           # strip guard
            for name, cc in (("raise", a), ("raise+aad", b), ("strip", d)):
                if not raises(IntegrityError, lambda: geng.decrypt_bytes(cc, passphrase=pw)):
                    return False, f"guard {name} attack accepted"
            return geng.decrypt_bytes(c, passphrase=pw)[0] == b"guard tamper", ""
        check("attempt_budget_cannot_be_raised_or_stripped", t_guard_tamper)

        def t_forged_burn():
            victim, vm = geng.encrypt_bytes(b"victim", HIGH)
            decoy, _ = geng.encrypt_bytes(b"decoy", HIGH)
            f = dict(decoy); f["shred_id"] = victim["shred_id"]
            f["aad"] = _b64e(_aad_from_container(f))
            for _ in range(15):
                if not raises(IntegrityError, lambda: geng.decrypt_bytes(f, passphrase="guess")):
                    return False, "forged container was processed"
            if geng.km.ledger()[victim["shred_id"]]["failed_attempts"] != 0:
                return False, "forged container spent the victim's budget"
            return geng.decrypt_bytes(victim, passphrase=vm["recovery_passphrase"])[0] == b"victim", ""
        check("forged_container_cannot_burn_other_files_budget", t_forged_burn)

        def t_public_ledger():
            c, _ = geng.encrypt_bytes(b"ledger", HIGH)
            e = geng.km.public_ledger()[c["shred_id"]]
            if "pw_verifier" in e or "commitment" in e:
                return False, "secret fields exposed"
            return e.get("max_attempts") == MAX_ATTEMPTS_POLICY["HIGH"], ""
        check("public_ledger_hides_verifier", t_public_ledger)

        def t_custom_pw_and_override():
            c, m = geng.encrypt_bytes(b"mine", LOW, max_attempts=2, passphrase="correct horse")
            if "recovery_passphrase" in m:
                return False, "generated a passphrase although one was supplied"
            if geng.decrypt_bytes(c, passphrase="correct horse")[0] != b"mine":
                return False, "custom passphrase failed"
            if burn(c) != 2:
                return False, "LOW override budget not applied"
            c, _ = geng.encrypt_bytes(b"x", CRITICAL, max_attempts=0)
            if "max_attempts" in c or "shred_id" in c and "expires_at" not in c:
                return False, "max_attempts=0 should disable the guard"
            if not raises(ValueError, lambda: geng.encrypt_bytes(b"x", LOW, max_attempts=-1)):
                return False, "negative budget accepted"
            return True, ""
        check("custom_passphrase_and_budget_override", t_custom_pw_and_override)

    n_pass = sum(1 for r in results.values() if r["passed"])
    print(f"\n[selftest] {n_pass}/{len(results)} checks passed")
    with open(RESULTS_DIR / "selftest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# --------------------------------------------------------------------------
# 10. FLASK WEB DASHBOARD (templates inlined since this is a single file)
# --------------------------------------------------------------------------

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Adaptive PQC Framework</title>
<style>
  :root{
    --bg:#0a0f1e; --bg-grad:#0d1428; --panel:#111a2e; --panel-2:#0d1524;
    --border:#1e293f; --accent:#38bdf8; --accent-2:#818cf8; --text:#e6ebf5;
    --muted:#8b96ad; --ok:#34d399; --bad:#f87171; --warn:#fbbf24;
    --low:#34d399; --medium:#fbbf24; --high:#fb923c; --critical:#f87171;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:radial-gradient(circle at 15% 0%, var(--bg-grad), var(--bg) 55%);
       color:var(--text);font-family:'Segoe UI',Inter,Roboto,Arial,sans-serif;line-height:1.5;}
  header{padding:28px 32px 22px;border-bottom:1px solid var(--border);
         display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px;}
  header .brand{display:flex;align-items:center;gap:14px;}
  header .logo{width:42px;height:42px;border-radius:12px;flex:none;
      background:linear-gradient(135deg,var(--accent),var(--accent-2));
      display:flex;align-items:center;justify-content:center;font-size:20px;}
  header h1{margin:0;font-size:19px;font-weight:700;letter-spacing:-.01em;}
  header p{margin:3px 0 0;color:var(--muted);font-size:12.5px;}
  .status-pill{display:flex;align-items:center;gap:8px;background:var(--panel);
      border:1px solid var(--border);padding:8px 14px;border-radius:999px;font-size:12.5px;}
  .dot{width:8px;height:8px;border-radius:50%;flex:none;}
  .dot.on{background:var(--ok);box-shadow:0 0 8px var(--ok);}
  .dot.off{background:var(--warn);box-shadow:0 0 8px var(--warn);}
  .wrap{max-width:980px;margin:0 auto;padding:28px 24px 60px;}
  .card{background:linear-gradient(180deg,var(--panel),var(--panel-2));
        border:1px solid var(--border);border-radius:14px;padding:22px 24px;margin-bottom:20px;
        box-shadow:0 8px 24px -12px rgba(0,0,0,.5);}
  .card h2{margin:0 0 4px;font-size:13px;color:var(--accent);text-transform:uppercase;
           letter-spacing:.08em;font-weight:700;}
  .card .sub{color:var(--muted);font-size:12.5px;margin:0 0 16px;}
  .banner{border-radius:10px;padding:12px 16px;font-size:13px;margin-bottom:16px;
          border:1px solid transparent;}
  .banner.warn{background:rgba(251,191,36,.08);border-color:rgba(251,191,36,.3);color:#fcd34d;}
  .banner.bad{background:rgba(248,113,113,.08);border-color:rgba(248,113,113,.3);color:#fca5a5;}
  .banner.ok{background:rgba(52,211,153,.08);border-color:rgba(52,211,153,.3);color:#6ee7b7;}
  .dropzone{border:1.5px dashed var(--border);border-radius:12px;padding:30px 20px;
      text-align:center;cursor:pointer;transition:.15s;background:rgba(255,255,255,.015);}
  .dropzone:hover,.dropzone.drag{border-color:var(--accent);background:rgba(56,189,248,.05);}
  .dropzone .big{font-size:14px;font-weight:600;margin-bottom:4px;}
  .dropzone .fname{color:var(--accent);font-size:13px;margin-top:8px;word-break:break-all;}
  input[type=file]{display:none;}
  .engine-select{display:flex;gap:10px;margin:16px 0 4px;flex-wrap:wrap;}
  .engine-opt{flex:1;min-width:150px;border:1px solid var(--border);border-radius:10px;
      padding:12px 14px;cursor:pointer;transition:.15s;}
  .engine-opt:hover{border-color:var(--accent-2);}
  .engine-opt input{margin-right:8px;}
  .engine-opt .t{font-weight:600;font-size:13px;}
  .engine-opt .d{color:var(--muted);font-size:11.5px;margin-top:2px;}
  .engine-opt.checked{border-color:var(--accent);background:rgba(56,189,248,.06);}
  button{background:linear-gradient(135deg,var(--accent),var(--accent-2));color:#08101f;
         border:none;padding:12px 22px;border-radius:9px;font-weight:700;font-size:13.5px;
         cursor:pointer;margin-top:16px;letter-spacing:.02em;}
  button:hover{opacity:.92;}
  .kv{display:grid;grid-template-columns:1fr 1fr;gap:8px 20px;font-size:13.5px;}
  .kv div.k{color:var(--muted);}
  .badge{display:inline-block;padding:4px 14px;border-radius:999px;font-size:13px;font-weight:800;
         letter-spacing:.03em;}
  .LOW{background:rgba(52,211,153,.15);color:var(--low);}
  .MEDIUM{background:rgba(251,191,36,.15);color:var(--medium);}
  .HIGH{background:rgba(251,146,60,.15);color:var(--high);}
  .CRITICAL{background:rgba(248,113,113,.15);color:var(--critical);}
  .ok{color:var(--ok);font-weight:700;} .bad{color:var(--bad);font-weight:700;}
  pre{white-space:pre-wrap;font-size:11.5px;background:#080d18;padding:12px;border-radius:8px;
      overflow:auto;border:1px solid var(--border);}
  a.dl{display:inline-block;color:#08101f;background:var(--accent);text-decoration:none;
       font-weight:700;font-size:13px;padding:10px 18px;border-radius:9px;margin-top:4px;}
  a.dl:hover{opacity:.9;}
  .risk-head{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:18px;}
  .confidence-bar{flex:1;min-width:180px;}
  .confidence-bar .track{background:#0b1220;border-radius:999px;height:8px;overflow:hidden;
      border:1px solid var(--border);}
  .confidence-bar .fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent-2));}
  .confidence-bar .lbl{font-size:11.5px;color:var(--muted);margin-top:5px;}
  .reasoning{background:rgba(129,140,248,.06);border:1px solid rgba(129,140,248,.25);
      border-radius:10px;padding:14px 16px;font-size:13.5px;margin-bottom:14px;color:#dbe2f5;}
  .reasoning .tag{color:var(--accent-2);font-weight:700;font-size:11px;text-transform:uppercase;
      letter-spacing:.06em;margin-bottom:6px;display:block;}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 14px;}
  .chip{background:var(--panel-2);border:1px solid var(--border);border-radius:999px;
      padding:4px 11px;font-size:11.5px;color:var(--muted);}
  .meter-row{display:flex;align-items:center;gap:10px;font-size:12px;margin:6px 0;}
  .meter-row .label{width:150px;color:var(--muted);flex:none;}
  .meter-row .track{flex:1;background:#0b1220;border-radius:999px;height:6px;overflow:hidden;
      border:1px solid var(--border);}
  .meter-row .fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent-2));}
  .meter-row .val{width:36px;text-align:right;color:var(--text);flex:none;}
  .stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;}
  .stat{background:var(--panel-2);border:1px solid var(--border);border-radius:10px;padding:14px;}
  .stat .n{font-size:19px;font-weight:800;}
  .stat .l{color:var(--muted);font-size:11px;margin-top:2px;text-transform:uppercase;letter-spacing:.04em;}
  details summary{cursor:pointer;color:var(--accent);font-size:12.5px;font-weight:600;}
  footer{max-width:980px;margin:0 auto;padding:0 24px 40px;color:var(--muted);font-size:11.5px;}
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="logo">&#128272;</div>
    <div>
      <h1>Adaptive AI-Driven Post-Quantum File Encryption</h1>
      <p>Risk-based security profiles &middot; Groq LLM risk assessment &middot; ChaCha20 / AES-256-GCM + ML-KEM key protection</p>
    </div>
  </div>
  <div class="status-pill">
    <span class="dot {{ 'on' if groq_configured else 'off' }}"></span>
    {% if groq_configured %}
      Groq connected &middot; {{ groq_model }}
    {% else %}
      Groq not configured &middot; set GROQ_API_KEY
    {% endif %}
  </div>
</header>
<div class="wrap">

  {% if not groq_configured %}
  <div class="banner warn">
    <strong>GROQ_API_KEY not set.</strong>
    The AI risk engine will not run until you configure it (see README —
    export GROQ_API_KEY or add it to a .env file next to main.py).
    {% if heuristic_ready %}Falling back to the local heuristic model for now.
    {% else %}No local fallback model is trained yet either — run
      <code>python main.py train</code> first.{% endif %}
  </div>
  {% endif %}

  <div class="card">
    <h2>1 &middot; Upload a file</h2>
    <p class="sub">The file is the only required input — risk signals are extracted automatically.</p>
    <form action="/encrypt" method="post" enctype="multipart/form-data" id="encForm">
      <label class="dropzone" id="dropzone" for="fileInput">
        <div class="big">Drop a file here, or click to browse</div>
        <div style="color:var(--muted);font-size:12px;">.txt .csv .json .md .log .xml .pdf .docx and more</div>
        <div class="fname" id="fname"></div>
      </label>
      <input type="file" name="file" id="fileInput" required>

      <div class="engine-select">
        <label class="engine-opt checked" data-eng>
          <input type="radio" name="engine" value="auto" checked>
          <span class="t">Auto</span>
          <div class="d">Groq AI, falls back to local model if unavailable</div>
        </label>
        <label class="engine-opt" data-eng>
          <input type="radio" name="engine" value="groq">
          <span class="t">Groq only</span>
          <div class="d">Force the LLM risk agent; errors if unavailable</div>
        </label>
        <label class="engine-opt" data-eng>
          <input type="radio" name="engine" value="heuristic">
          <span class="t">Heuristic only</span>
          <div class="d">Local RandomForest, no network call</div>
        </label>
      </div>

      <div style="margin:16px 0 4px;font-size:13px;color:var(--muted);">
        Key lifetime override (seconds, optional):
        <input type="number" name="ttl_seconds" min="0" placeholder="risk-tier default"
               style="width:150px;margin-left:6px;padding:6px 8px;border-radius:8px;border:1px solid var(--border,#2a3350);background:transparent;color:inherit;">
        <div style="margin-top:4px;font-size:12px;">Leave blank to use the policy for the detected risk tier. 0 = never expire.
        When the lifetime ends the key is destroyed and the file can no longer be decrypted.</div>
      </div>

      <div style="margin:14px 0 4px;font-size:13px;color:var(--muted);">
        Attempt budget override (wrong passphrases before the key self-destructs, optional):
        <input type="number" name="max_attempts" min="0" placeholder="risk-tier default"
               style="width:150px;margin-left:6px;padding:6px 8px;border-radius:8px;border:1px solid var(--border,#2a3350);background:transparent;color:inherit;">
        <div style="margin-top:8px;">Passphrase (optional):
        <input type="password" name="passphrase" autocomplete="new-password" placeholder="auto-generate"
               style="width:190px;margin-left:6px;padding:6px 8px;border-radius:8px;border:1px solid var(--border,#2a3350);background:transparent;color:inherit;"></div>
        <div style="margin-top:4px;font-size:12px;">Files in guarded tiers (MEDIUM / HIGH / CRITICAL by default) need a passphrase to open.
        If you leave it blank one is generated and shown once. 0 = unlimited attempts, no passphrase.</div>
      </div>

      <button type="submit">Analyze &amp; Encrypt</button>
    </form>
    {% if enc_error %}<div class="banner bad" style="margin-top:14px;">{{ enc_error }}</div>{% endif %}
  </div>

  {% if result %}
  <div class="card">
    <h2>2 &middot; AI Risk Assessment</h2>
    <p class="sub">
      Engine used: <strong>{{ result.risk_engine_used }}</strong>
      {% if result.risk_engine_model %} ({{ result.risk_engine_model }}){% endif %}
    </p>
    {% if result.risk_fallback_used %}
    <div class="banner warn">
      Groq was unavailable, so this file was assessed by the local fallback
      model instead. Reason: {{ result.risk_fallback_reason }}
    </div>
    {% endif %}

    <div class="risk-head">
      <span class="badge {{ result.risk_class }}">{{ result.risk_class }} RISK</span>
      <div class="confidence-bar">
        <div class="track"><div class="fill" style="width:{{ (result.risk_confidence*100)|round(1) }}%;"></div></div>
        <div class="lbl">Confidence: {{ "%.1f"|format(result.risk_confidence*100) }}%</div>
      </div>
    </div>

    {% if result.risk_reasoning %}
    <div class="reasoning">
      <span class="tag">AI reasoning</span>
      {{ result.risk_reasoning }}
    </div>
    {% endif %}

    {% if result.risk_key_concerns %}
    <div class="chips">
      {% for c in result.risk_key_concerns %}<span class="chip">{{ c }}</span>{% endfor %}
    </div>
    {% endif %}

    {% if result.risk_feature_scores %}
    <div style="margin-top:6px;">
      {% for k, v in result.risk_feature_scores.items() %}
      {% if k != 'file_size_mb_log' %}
      <div class="meter-row">
        <div class="label">{{ k.replace('_',' ') }}</div>
        <div class="track"><div class="fill" style="width:{{ v|round(0) }}%;"></div></div>
        <div class="val">{{ v|round(0) }}</div>
      </div>
      {% endif %}
      {% endfor %}
    </div>
    {% endif %}

    <details style="margin-top:14px;">
      <summary>Show raw automated evidence (regex/keyword scan)</summary>
      <pre>{{ result.auto_features.evidence | tojson(indent=2) }}</pre>
    </details>
  </div>

  <div class="card">
    <h2>3 &middot; Selected Security Profile</h2>
    <p class="sub">Chosen automatically from the risk class above.</p>
    <div class="stat-grid">
      <div class="stat"><div class="n">{{ result.selected_profile.name }}</div><div class="l">Profile</div></div>
      <div class="stat"><div class="n">{{ result.selected_profile.cipher }}</div><div class="l">Symmetric cipher</div></div>
      <div class="stat"><div class="n">{{ result.selected_profile.key_protection }}</div><div class="l">Key protection</div></div>
      <div class="stat"><div class="n">{{ result.selected_profile.security_level }}</div><div class="l">Security level</div></div>
      <div class="stat"><div class="n">{{ result.encryption_metrics.key_lifetime }}</div><div class="l">Key lifetime</div></div>
      <div class="stat"><div class="n">256-bit</div><div class="l">Key commitment</div></div>
      <div class="stat"><div class="n">{{ result.encryption_metrics.max_attempts or 'Unlimited' }}</div><div class="l">Wrong-passphrase budget</div></div>
    </div>
    <p class="sub" style="margin-top:12px;">{{ result.selected_profile.description }}</p>
  </div>

  <div class="card">
    <h2>4 &middot; Encryption Metrics</h2>
    <div class="stat-grid">
      <div class="stat"><div class="n">{{ result.encryption_metrics.original_size_bytes }} B</div><div class="l">Original size</div></div>
      <div class="stat"><div class="n">{{ result.encryption_metrics.encrypted_size_bytes }} B</div><div class="l">Encrypted size</div></div>
      <div class="stat"><div class="n">{{ "%.2f"|format(result.ciphertext_expansion_pct) }}%</div><div class="l">Ciphertext expansion</div></div>
      <div class="stat"><div class="n">{{ "%.4f"|format(result.encryption_metrics.encryption_time_s) }}s</div><div class="l">Encryption time</div></div>
      <div class="stat"><div class="n">{{ "%.4f"|format(result.encryption_metrics.key_setup_time_s) }}s</div><div class="l">Key setup</div></div>
      {% if result.selected_profile.kem_algo %}
      <div class="stat"><div class="n">{{ "%.4f"|format(result.encryption_metrics.kem_encapsulation_time_s) }}s</div><div class="l">KEM encapsulation</div></div>
      {% endif %}
      {% if result.encryption_metrics.expires_at_text %}
      <div class="stat"><div class="n" style="font-size:15px;">{{ result.encryption_metrics.expires_at_text }}</div><div class="l">Key destroyed at</div></div>
      {% endif %}
    </div>
    {% if result.encryption_metrics.recovery_passphrase %}
    <div class="banner" style="margin-top:14px;word-break:break-all;">
      Passphrase for this file (shown once, not stored):
      <strong>{{ result.encryption_metrics.recovery_passphrase }}</strong><br>
      After {{ result.encryption_metrics.max_attempts }} wrong attempts the key is destroyed and the file is lost for good.
    </div>
    {% endif %}
    {% if result.encryption_metrics.expires_at_text %}
    <p class="sub" style="margin-top:12px;">After this time the key material is shredded and this file is permanently undecryptable.
      Do not copy the keys/ folder if you rely on this guarantee.</p>
    {% endif %}
    <a class="dl" style="margin-top:18px;" href="/download/{{ result.encryption_metrics.output_name }}">&#11015; Download encrypted (.pqc) file</a>
  </div>
  {% endif %}

  <div class="card">
    <h2>5 &middot; Decrypt a File</h2>
    <p class="sub">Upload a .pqc container produced by this tool to recover and verify the original.</p>
    <form action="/decrypt" method="post" enctype="multipart/form-data" id="decForm">
      <label class="dropzone" id="dropzoneDec" for="fileInputDec">
        <div class="big">Drop a .pqc file here, or click to browse</div>
        <div class="fname" id="fnameDec"></div>
      </label>
      <input type="file" name="file" id="fileInputDec" required>
      <div style="margin:12px 0 4px;font-size:13px;color:var(--muted);">Passphrase (for guarded files):
        <input type="password" name="passphrase" autocomplete="off"
               style="width:210px;margin-left:6px;padding:6px 8px;border-radius:8px;border:1px solid var(--border,#2a3350);background:transparent;color:inherit;"></div>
      <button type="submit">Decrypt &amp; Verify</button>
    </form>
    {% if dec_result %}
    <div class="stat-grid" style="margin-top:18px;">
      <div class="stat"><div class="n">{{ "%.4f"|format(dec_result.decryption_time_s) }}s</div><div class="l">Decryption time</div></div>
      <div class="stat"><div class="n">{{ dec_result.cipher }}</div><div class="l">Cipher used</div></div>
      <div class="stat"><div class="n {{ 'ok' if dec_result.integrity_ok else 'bad' }}">{{ 'VERIFIED' if dec_result.integrity_ok else 'FAILED' }}</div><div class="l">Integrity check</div></div>
      {% if dec_result.key_commitment_verified %}
      <div class="stat"><div class="n ok">VERIFIED</div><div class="l">Key commitment</div></div>
      {% endif %}
    </div>
    <p class="sub" style="margin-top:10px;word-break:break-all;">SHA-256: {{ dec_result.sha256 }}</p>
    <a class="dl" href="/download_decrypted/{{ dec_result.output_name }}">&#11015; Download recovered file</a>
    {% endif %}
    {% if dec_error %}<div class="banner bad" style="margin-top:14px;">Decryption failed: {{ dec_error }}</div>{% endif %}
  </div>

</div>
<footer>
  Key expiry destroys the local per-file key; it cannot protect copies of keys/ you have backed up elsewhere.<br>
  Academic research prototype — not a production security product.
  Post-quantum KEM (HIGH / CRITICAL profiles): pure-Python FIPS-203 ML-KEM (kyber-py), not liboqs.
  LOW / MEDIUM use conventional symmetric keys. Key storage under keys/ is local plaintext. Risk classification is
  performed by an LLM (Groq) with a local RandomForest as an automatic
  offline fallback — see README for details.
</footer>
<script>
  function wireDropzone(zoneId, inputId, fnameId){
    var zone = document.getElementById(zoneId);
    var input = document.getElementById(inputId);
    var fname = document.getElementById(fnameId);
    if(!zone || !input) return;
    input.addEventListener('change', function(){
      if(input.files && input.files[0]){ fname.textContent = input.files[0].name; }
    });
    ['dragover','dragenter'].forEach(function(evt){
      zone.addEventListener(evt, function(e){ e.preventDefault(); zone.classList.add('drag'); });
    });
    ['dragleave','drop'].forEach(function(evt){
      zone.addEventListener(evt, function(e){ e.preventDefault(); zone.classList.remove('drag'); });
    });
    zone.addEventListener('drop', function(e){
      if(e.dataTransfer.files && e.dataTransfer.files[0]){
        input.files = e.dataTransfer.files;
        fname.textContent = e.dataTransfer.files[0].name;
      }
    });
  }
  wireDropzone('dropzone','fileInput','fname');
  wireDropzone('dropzoneDec','fileInputDec','fnameDec');

  document.querySelectorAll('[data-eng]').forEach(function(opt){
    opt.addEventListener('click', function(){
      document.querySelectorAll('[data-eng]').forEach(function(o){ o.classList.remove('checked'); });
      opt.classList.add('checked');
    });
  });
</script>
</body>
</html>
"""


def _safe_name(filename: str) -> str:
    """Strip any directory components from a client-supplied filename."""
    name = Path((filename or "").replace("\\", "/")).name
    return name or "upload.bin"


def create_app():
    from flask import Flask, request, render_template_string, send_from_directory, abort, jsonify

    app = Flask(__name__)
    risk_model = RiskModel()
    if MODEL_PATH.exists():
        risk_model.load()
    engine = CryptoEngine()
    engine.km.ensure_all()
    groq_assessor = GroqRiskAssessor()
    groq_configured = bool(os.environ.get("GROQ_API_KEY"))

    # Background crypto-shredder: destroys keys whose lifetime has ended even
    # if nobody tries to open the file.
    def _sweeper():
        while True:
            try:
                engine.km.sweep_expired()
            except Exception:
                pass
            time.sleep(15)
    engine.km.sweep_expired()
    threading.Thread(target=_sweeper, daemon=True, name="key-sweeper").start()

    def render_index(**kwargs):
        ctx = {
            "result": None, "dec_result": None, "dec_error": None, "enc_error": None,
            "groq_configured": groq_configured, "groq_model": groq_assessor.model,
            "heuristic_ready": MODEL_PATH.exists(),
        }
        ctx.update(kwargs)
        return render_template_string(INDEX_HTML, **ctx)

    @app.route("/", methods=["GET"])
    def index():
        return render_index()

    @app.route("/encrypt", methods=["POST"])
    def do_encrypt():
        f = request.files.get("file")
        if not f or f.filename == "":
            return render_index(enc_error="No file uploaded"), 400

        requested_engine = request.form.get("engine", "auto")
        if requested_engine not in ("auto", "groq", "heuristic"):
            requested_engine = "auto"
        if requested_engine == "heuristic" and not MODEL_PATH.exists():
            return render_index(
                enc_error="Heuristic engine selected but no local model is trained yet. "
                           "Run: python main.py train"
            ), 400
        if requested_engine != "heuristic" and not groq_configured and not MODEL_PATH.exists():
            return render_index(
                enc_error="No GROQ_API_KEY configured and no local fallback model trained. "
                           "Set GROQ_API_KEY (see README) or run: python main.py train"
            ), 400

        ttl_raw = (request.form.get("ttl_seconds") or "").strip()
        ttl_seconds = None
        if ttl_raw:
            try:
                ttl_seconds = int(ttl_raw)
                if ttl_seconds < 0:
                    raise ValueError
            except ValueError:
                return render_index(enc_error="Key lifetime must be a whole number of seconds (0 = never)."), 400

        att_raw = (request.form.get("max_attempts") or "").strip()
        max_attempts = None
        if att_raw:
            try:
                max_attempts = int(att_raw)
                if max_attempts < 0:
                    raise ValueError
            except ValueError:
                return render_index(enc_error="Attempt budget must be a whole number (0 = unlimited)."), 400
        passphrase = request.form.get("passphrase") or None

        save_path = UPLOADS_DIR / _safe_name(f.filename)
        f.save(save_path)

        try:
            result = run_full_pipeline(save_path, risk_model, engine,
                                        risk_engine=requested_engine,
                                        groq_assessor=groq_assessor,
                                        ttl_seconds=ttl_seconds,
                                        max_attempts=max_attempts, passphrase=passphrase)
        except KEMUnavailableError as e:
            return f"Post-quantum backend unavailable: {e}", 500
        except GroqUnavailableError as e:
            return render_index(enc_error=f"Groq risk assessment failed: {e}"), 502

        return render_index(result=result)

    @app.route("/decrypt", methods=["POST"])
    def do_decrypt():
        f = request.files.get("file")
        if not f or f.filename == "":
            return render_index(dec_error="No file uploaded"), 400
        safe_name = _safe_name(f.filename)
        save_path = ENCRYPTED_DIR / safe_name
        f.save(save_path)
        out_name = safe_name[:-4] if safe_name.endswith(".pqc") else safe_name + ".decrypted"
        out_path = DECRYPTED_DIR / out_name
        try:
            metrics = engine.decrypt_file(save_path, out_path,
                                           passphrase=request.form.get("passphrase") or None)
            return render_index(dec_result=metrics)
        except Exception as e:
            return render_index(dec_error=str(e))

    @app.route("/api/keys")
    def api_keys():
        """Key-lifecycle ledger: id -> expiry / ACTIVE or SHREDDED (no key material)."""
        return jsonify(engine.km.public_ledger())

    @app.route("/api/status")
    def api_status():
        return jsonify({
            "groq_configured": groq_configured,
            "groq_model": groq_assessor.model,
            "heuristic_model_ready": MODEL_PATH.exists(),
        })

    @app.route("/download/<path:fname>")
    def download(fname):
        return send_from_directory(ENCRYPTED_DIR, fname, as_attachment=True)

    @app.route("/download_decrypted/<path:fname>")
    def download_decrypted(fname):
        return send_from_directory(DECRYPTED_DIR, fname, as_attachment=True)

    return app


# --------------------------------------------------------------------------
# 11. CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the synthetic-policy risk model")
    p_train.add_argument("--samples", type=int, default=8000)

    p_analyze = sub.add_parser(
        "analyze", help="Run fully automatic feature extraction + risk assessment (file only)"
    )
    p_analyze.add_argument("--file", required=True)
    p_analyze.add_argument("--engine", choices=["auto", "groq", "heuristic"], default="auto",
                            help="Risk engine: groq (LLM), heuristic (local RandomForest), "
                                 "or auto (try groq, fall back to heuristic). Default: auto")
    p_analyze.add_argument("--groq-api-key", default=None, help="Overrides GROQ_API_KEY env var")
    p_analyze.add_argument("--groq-model", default=None, help="Overrides GROQ_MODEL env var")
    p_analyze.add_argument("--override-sensitivity", type=float, default=None,
                            help="Optional manual override (0-100)")
    p_analyze.add_argument("--override-threat-level", type=float, default=None)
    p_analyze.add_argument("--override-access-risk", type=float, default=None)
    p_analyze.add_argument("--override-external-sharing", type=float, default=None)

    p_encrypt = sub.add_parser(
        "encrypt", help="Full pipeline: automatic analysis + encrypt (file only required)"
    )
    p_encrypt.add_argument("--file", required=True)
    p_encrypt.add_argument("--engine", choices=["auto", "groq", "heuristic"], default="auto",
                            help="Risk engine: groq (LLM), heuristic (local RandomForest), "
                                 "or auto (try groq, fall back to heuristic). Default: auto")
    p_encrypt.add_argument("--groq-api-key", default=None, help="Overrides GROQ_API_KEY env var")
    p_encrypt.add_argument("--groq-model", default=None, help="Overrides GROQ_MODEL env var")
    p_encrypt.add_argument("--override-sensitivity", type=float, default=None,
                            help="Optional manual override (0-100)")
    p_encrypt.add_argument("--override-threat-level", type=float, default=None)
    p_encrypt.add_argument("--override-access-risk", type=float, default=None)
    p_encrypt.add_argument("--override-external-sharing", type=float, default=None)

    p_encrypt.add_argument("--ttl-seconds", type=int, default=None,
                            help="Key lifetime override in seconds (0 = never expire). "
                                 "Default: the risk-tier policy (LOW none, MEDIUM 90d, HIGH 30d, CRITICAL 7d)")

    p_encrypt.add_argument("--max-attempts", type=int, default=None,
                            help="Wrong-passphrase budget before the key self-destructs "
                                 "(0 = unlimited). Default: risk-tier policy "
                                 "(LOW unlimited, MEDIUM 20, HIGH 10, CRITICAL 3)")
    p_encrypt.add_argument("--passphrase", default=None,
                            help="Passphrase for guarded files (default: auto-generated and shown once)")

    p_decrypt = sub.add_parser("decrypt", help="Decrypt a .pqc container")
    p_decrypt.add_argument("--file", required=True)
    p_decrypt.add_argument("--out", required=False)
    p_decrypt.add_argument("--passphrase", default=None,
                           help="Passphrase for guarded files (omit to be prompted securely)")

    p_bench = sub.add_parser("benchmark", help="Run KEM/AES benchmark suite")
    p_bench.add_argument("--sizes-kb", type=int, nargs="+",
                          default=[1, 10, 100, 1024, 5120, 10240])
    p_bench.add_argument("--reps", type=int, default=5)

    sub.add_parser("selftest", help="Run security validation checks")

    sub.add_parser("sweep", help="Destroy (crypto-shred) every key whose lifetime has ended")

    p_shred = sub.add_parser("shred", help="Crypto-shred a .pqc file NOW: destroy its per-file key")
    p_shred.add_argument("--file", required=True)

    p_inspect = sub.add_parser("inspect", help="Show a .pqc header + key lifetime status (no decryption)")
    p_inspect.add_argument("--file", required=True)

    p_serve = sub.add_parser("serve", help="Run the Flask web dashboard")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.command == "train":
        rm = RiskModel()
        metrics = rm.train(n_samples=args.samples)
        print(json.dumps({k: v for k, v in metrics.items()
                           if k not in ("confusion_matrix", "classification_report")}, indent=2))
        print(f"\nModel saved to {MODEL_PATH}")
        print(f"Metrics written to {RESULTS_DIR / 'ai_metrics.txt'}")

    elif args.command == "analyze":
        rm = RiskModel()
        if MODEL_PATH.exists():
            rm.load()
        overrides = {
            "sensitivity": args.override_sensitivity,
            "threat_level": args.override_threat_level,
            "access_risk": args.override_access_risk,
            "external_sharing": args.override_external_sharing,
        }
        groq_assessor = GroqRiskAssessor(api_key=args.groq_api_key, model=args.groq_model)
        verdict = assess_risk(Path(args.file), rm, overrides=overrides,
                               engine=args.engine, groq_assessor=groq_assessor)
        profile = select_profile(verdict["risk_class"])
        print(json.dumps({
            "auto_features": verdict["auto_features"],
            "risk_class": verdict["risk_class"],
            "confidence": verdict["confidence"],
            "class_probabilities": verdict["class_probabilities"],
            "engine_used": verdict["engine_used"],
            "fallback_used": verdict["fallback_used"],
            "fallback_reason": verdict["fallback_reason"],
            "reasoning": verdict["reasoning"],
            "key_concerns": verdict["key_concerns"],
            "selected_profile": profile_to_dict(profile),
        }, indent=2))

    elif args.command == "encrypt":
        rm = RiskModel()
        if MODEL_PATH.exists():
            rm.load()
        engine = CryptoEngine()
        engine.km.ensure_all()
        overrides = {
            "sensitivity": args.override_sensitivity,
            "threat_level": args.override_threat_level,
            "access_risk": args.override_access_risk,
            "external_sharing": args.override_external_sharing,
        }
        groq_assessor = GroqRiskAssessor(api_key=args.groq_api_key, model=args.groq_model)
        result = run_full_pipeline(Path(args.file), rm, engine, overrides=overrides,
                                    risk_engine=args.engine, groq_assessor=groq_assessor,
                                    ttl_seconds=args.ttl_seconds,
                                    max_attempts=args.max_attempts, passphrase=args.passphrase)
        if result["encryption_metrics"].get("recovery_passphrase"):
            print("\n*** Passphrase for this file (shown once, not stored): "
                  f"{result['encryption_metrics']['recovery_passphrase']} ***", file=sys.stderr)
        print(json.dumps(result, indent=2))

    elif args.command == "decrypt":
        engine = CryptoEngine()
        in_path = Path(args.file)
        out_path = Path(args.out) if args.out else DECRYPTED_DIR / in_path.stem
        try:
            try:
                metrics = engine.decrypt_file(in_path, out_path, passphrase=args.passphrase)
            except PassphraseRequiredError:
                if not sys.stdin.isatty():
                    raise
                import getpass
                metrics = engine.decrypt_file(in_path, out_path,
                                              passphrase=getpass.getpass("Passphrase: "))
        except (IntegrityError, KeyStoreError, UnsupportedKEMError, UnsupportedCipherError) as e:
            print(f"Decryption failed: {e}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(metrics, indent=2))

    elif args.command == "benchmark":
        run_benchmark(sizes_kb=args.sizes_kb, reps=args.reps)

    elif args.command == "selftest":
        run_selftest()

    elif args.command == "sweep":
        km = KeyManager()
        done = km.sweep_expired()
        print(json.dumps({"shredded": done, "count": len(done)}, indent=2))

    elif args.command in ("shred", "inspect"):
        try:
            c = json.loads(Path(args.file).read_text())
            assert isinstance(c, dict)
        except Exception as e:
            print(f"Not a valid .pqc container: {e}", file=sys.stderr)
            sys.exit(2)
        km = KeyManager()
        kid = c.get("shred_id") or c.get("key_id")
        entry = km.ledger().get(kid) if kid else None
        if args.command == "inspect":
            exp = c.get("expires_at")
            print(json.dumps({
                "version": c.get("version", 1), "cipher": c.get("cipher"),
                "key_mode": c.get("key_mode"), "kem_algo": c.get("kem_algo"),
                "key_commitment": "present" if "key_commitment" in c else "absent (pre-v3)",
                "expires_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp)) if exp else None,
                "seconds_remaining": max(0, int(exp - time.time())) if exp else None,
                "ledger_status": entry.get("status") if entry else None,
                "max_attempts": c.get("max_attempts"),
                "attempts_left": (entry["max_attempts"] - entry.get("failed_attempts", 0))
                                 if entry and entry.get("max_attempts") and entry.get("status") == "ACTIVE" else None,
            }, indent=2))
        else:
            if not kid:
                print("This container has no per-file key to destroy. Only containers made with a "
                      "key lifetime (v3) or LOW-tier files can be shredded individually.", file=sys.stderr)
                sys.exit(2)
            destroyed = km.shred(kid, reason="manual")
            print(json.dumps({"shredded_id": kid, "key_material_destroyed": destroyed,
                              "note": "The file can no longer be decrypted."}, indent=2))

    elif args.command == "serve":
        app = create_app()
        app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()