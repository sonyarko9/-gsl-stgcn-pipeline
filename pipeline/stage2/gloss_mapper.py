"""
CGOP Gloss Mapper - Stage 2 (Real-Time)
========================================
Pipeline:
  1. Preprocess  - strip stopwords, normalize text
  2. Cache lookup - return cached gloss if previously mapped
  3. Dictionary override - deterministic match (multi-word first)
  4. Mistral-7B fallback via Ollama (temperature=0, strict prompt)
  5. Output cleaning - uppercase, no punctuation, validated tokens
  6. FS(TERM) safety net - fingerspell anything still unresolved

Watchdog loop monitors live_text_buffer.txt and maps glosses
immediately whenever new text is detected.
Output -> pipeline/stage3/live_gloss_buffer.json
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
import logging

from pipeline.logging_setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
_DIR         = Path(__file__).parent
CACHE_PATH   = _DIR / "gloss_cache.json"
DICT_PATH    = _DIR / "gloss_dictionary.json"
BUFFER_PATH  = _DIR / "live_text_buffer.txt"
OUTPUT_PATH  = _DIR.parent / "stage3" / "live_gloss_buffer.json"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# STOP WORDS
# ─────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "that", "this", "it",
    "its", "we", "you", "i", "they", "and", "or", "but", "so",
    "as", "not", "about", "going", "just", "very", "also", "some",
    "our", "their", "there", "here", "which", "then", "than", "into",
    "onto", "up", "down", "let", "now", "get", "got", "use", "used",
    "one", "two", "three", "can", "all", "each", "every", "any",
    "such", "these", "those", "both", "either", "neither", "new",
    "first", "second", "last", "next", "other", "same", "see",
    "know", "make", "think", "take", "give", "show", "tell",
    "go", "come", "look", "want", "need", "mean", "include",
    "consider", "allow", "require", "provide", "represent",
    "following", "given", "based", "per", "according"
}

# ─────────────────────────────────────────────────────────────────
# DICTIONARY LOADER
# ─────────────────────────────────────────────────────────────────
def _load_dictionary() -> dict:
    if not DICT_PATH.exists():
        raise FileNotFoundError(
            f"[ERROR] gloss_dictionary.json not found at: {DICT_PATH}\n"
            f"Make sure gloss_dictionary.json is in the same folder as gloss_mapper.py."
        )
    try:
        with open(DICT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Dictionary loaded: %d entries from %s", len(data), DICT_PATH.name)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"[ERROR] gloss_dictionary.json is not valid JSON: {e}")

GLOSS_DICTIONARY: dict = _load_dictionary()

# ─────────────────────────────────────────────────────────────────
# CONSISTENCY CACHE
# ─────────────────────────────────────────────────────────────────
def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            logger.debug("Failed to load cache; starting with empty cache.")
    return {}

def _save_cache(cache: dict) -> None:
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("Could not save cache: %s", e)

_GLOSS_CACHE: dict = _load_cache()

def cache_lookup(word: str) -> str | None:
    return _GLOSS_CACHE.get(word.lower())

def cache_store(word: str, gloss: str) -> None:
    key = word.lower()
    if key in GLOSS_DICTIONARY:
        return
    _GLOSS_CACHE[key] = gloss
    _save_cache(_GLOSS_CACHE)

# ─────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ─────────────────────────────────────────────────────────────────
# OUTPUT CLEANING & VALIDATION
# ─────────────────────────────────────────────────────────────────
_VALID_GLOSS = re.compile(r"^[A-Z][A-Z0-9]*(-[A-Z][A-Z0-9]*)*$")
_FS_PATTERN  = re.compile(r"^FS\([A-Z0-9-]+\)$")

def clean_gloss_token(raw: str) -> str:
    token = raw.upper().strip()
    token = re.sub(r"[^\w\s-]", "", token)
    token = re.sub(r"\s+", "-", token).strip("-")
    token = re.sub(r"-{2,}", "-", token)
    if _VALID_GLOSS.match(token) or _FS_PATTERN.match(token):
        return token
    fs_inner = re.sub(r"[^A-Z0-9-]", "", token)
    return f"FS({fs_inner})" if fs_inner else "FS(UNKNOWN)"

def clean_gloss_list(raw_tokens: list[str]) -> list[str]:
    cleaned, seen = [], set()
    for t in raw_tokens:
        g = clean_gloss_token(t)
        if g and g not in seen:
            cleaned.append(g)
            seen.add(g)
    return cleaned

# ─────────────────────────────────────────────────────────────────
# LAYER 1: DETERMINISTIC MAPPER
# ─────────────────────────────────────────────────────────────────
def deterministic_map(text: str) -> tuple[list[str], list[str]]:
    words = preprocess(text).split()
    glosses, unmapped = [], []
    i = 0
    while i < len(words):
        matched = False
        for phrase_len in range(4, 0, -1):
            phrase = " ".join(words[i: i + phrase_len])
            if phrase in GLOSS_DICTIONARY:
                glosses.append(GLOSS_DICTIONARY[phrase])
                i += phrase_len
                matched = True
                break
        if not matched:
            if words[i] not in STOP_WORDS:
                unmapped.append(words[i])
            i += 1
    return glosses, unmapped

# ─────────────────────────────────────────────────────────────────
# LAYER 1.5: CACHE LAYER
# ─────────────────────────────────────────────────────────────────
def cache_map(unmapped: list[str]) -> tuple[list[str], list[str]]:
    cached, still = [], []
    for word in unmapped:
        hit = cache_lookup(word)
        if hit:
            cached.append(hit)
        else:
            still.append(word)
    return cached, still

# ─────────────────────────────────────────────────────────────────
# LAYER 2: MISTRAL-7B FALLBACK VIA OLLAMA
# ─────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a Ghana Sign Language (GSL) gloss generator for a CS/IT university lecture system.

STRICT OUTPUT RULES (no exceptions):
- Output ONLY uppercase gloss tokens, one per line
- Use hyphens for compound glosses: BINARY-TREE, TIME-COMPLEXITY
- Output EXACTLY one gloss per input word, in the SAME ORDER
- Do NOT write explanations, numbers, punctuation, or sentences
- Do NOT output articles (a, the), prepositions, or conjunctions
- If a word has no known GSL equivalent, output the word itself in UPPERCASE
- Domain: Computer Science and IT terms only"""

_EXAMPLE_FEW_SHOT = """Example:
Input words: pointer, traversal, efficient
Output:
POINTER
TRAVERSAL
EFFICIENT"""

def ollama_fallback(unmapped_words: list[str]) -> list[str]:
    if not unmapped_words:
        return []
    try:
        import requests
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n{_EXAMPLE_FEW_SHOT}\n\n"
            f"Now convert these words:\nInput words: {', '.join(unmapped_words)}\nOutput:"
        )
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":   "mistral",
                "prompt":  prompt,
                "stream":  False,
                "options": {
                    "temperature": 0,
                    "top_p":       1,
                    "top_k":       1,
                    "num_predict": 256,
                    "stop":        ["\n\n", "---"]
                }
            },
            timeout=60
        )
        if response.status_code == 200:
            raw       = response.json().get("response", "").strip()
            raw_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            while len(raw_lines) < len(unmapped_words):
                raw_lines.append(unmapped_words[len(raw_lines)])
            raw_lines = raw_lines[:len(unmapped_words)]
            glosses   = [clean_gloss_token(ln) for ln in raw_lines]
            for word, gloss in zip(unmapped_words, glosses):
                if not gloss.startswith("FS("):
                    cache_store(word, gloss)
            return glosses
        else:
            logger.warning("Ollama HTTP %s - using FS() fallback.", response.status_code)
    except Exception as e:
        logger.warning("Ollama unavailable (%s) - using FS() fallback.", type(e).__name__)
    return [f"FS({re.sub(r'[^A-Z0-9]', '', w.upper()) or 'UNKNOWN'})" for w in unmapped_words]

# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
def map_to_glosses(text: str) -> dict:
    clean_text                     = preprocess(text)
    det_glosses, unmapped          = deterministic_map(clean_text)
    cached_glosses, still_unmapped = cache_map(unmapped)
    fallback_glosses               = ollama_fallback(still_unmapped)
    final_glosses                  = clean_gloss_list(det_glosses + cached_glosses + fallback_glosses)

    logger.info("[CGOP] IN  : %s", clean_text)
    logger.info("[CGOP] OUT : %s\n", " | ".join(final_glosses))

    return {
        "source_text":         text,
        "glosses":             final_glosses,
        "deterministic_count": len(det_glosses),
        "cache_count":         len(cached_glosses),
        "fallback_count":      len(fallback_glosses),
        "timestamp":           datetime.now().isoformat(),
    }

# ─────────────────────────────────────────────────────────────────
# REAL-TIME WATCHDOG LOOP
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("[GLOSS MAPPER] Running - watching for new transcriptions.\n")

    # Stamp the current mtime so we ignore whatever is already in the buffer
    last_modified = BUFFER_PATH.stat().st_mtime if BUFFER_PATH.exists() else 0.0
    last_text     = ""

    while True:
        try:
            if not BUFFER_PATH.exists():
                time.sleep(0.2)
                continue

            mtime = BUFFER_PATH.stat().st_mtime
            if mtime <= last_modified:
                time.sleep(0.1)
                continue

            last_modified = mtime

            raw  = BUFFER_PATH.read_bytes()
            if not raw:
                continue

            # Handle both UTF-16 (Windows echo) and UTF-8 (pipeline-written)
            text = raw.decode(
                "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
            ).strip().replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

            if not text or text == last_text:
                continue

            last_text = text
            result    = map_to_glosses(text)
            OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

        except KeyboardInterrupt:
            logger.info("\n[GLOSS MAPPER] Stopped.")
            break
        except Exception as e:
            logger.exception("[GLOSS MAPPER ERROR] %s", e)
            time.sleep(0.5)
