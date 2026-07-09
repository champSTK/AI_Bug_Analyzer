from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time

import httpx
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from schemas import StructuredBugReport

logger = logging.getLogger("groq_client")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

# Groq's free-tier endpoint rejects oversized request bodies with a plain
# HTTP 413 before the model ever sees them. Retrying an oversized payload
# unchanged will always fail again, so instead we truncate defensively —
# the full original text is still stored in the document store untouched;
# only what's *sent to the model* is capped.
MAX_INPUT_CHARS = 6000

# Groq's confirmed free-tier limit for llama-3.1-8b-instant is 30 requests
# per minute. Default to a few under that so normal clock jitter and any
# other process sharing this API key don't tip it over 30. Override with
# the GROQ_RPM env var if your account/model has different limits.
GROQ_RPM = int(os.getenv("GROQ_RPM", "28"))

SYSTEM_PROMPT = """You are a strict log-parsing engine. You receive a raw \
crash log, stack trace, or bug report text and you extract it into JSON.

Output ONLY a JSON object with exactly these keys, no prose, no markdown:
{
  "error_code": string or null,
  "message": string,
  "stack_trace": string or null,
  "environment": {"os": string or null, "runtime": string or null, "version": string or null},
  "metadata": {"title": string or null, "severity": "LOW"|"MEDIUM"|"HIGH"|"CRITICAL", "component": string or null, "timestamp": string or null}
}

Rules:
- "message" is required: summarize the core error in one clear sentence if the log doesn't state it explicitly.
- Infer "severity" from the log content (e.g. "CRITICAL" for crashes/data loss, "LOW" for warnings) if not stated.
- If a field truly cannot be determined, use null. Never invent stack traces, versions, or IDs that aren't present.
- Preserve the stack trace text as-is (don't summarize it) if one is present.
"""


class GroqRateLimitError(Exception):
    """Raised on HTTP 429 from Groq so tenacity knows to retry."""


class GroqExtractionError(Exception):
    """
    Raised for any non-recoverable Groq API problem (bad request, payload
    too large, malformed response shape, ...). Retrying the exact same
    request will not help, so this is never retried automatically.
    """


def _is_rate_limit(exc: BaseException) -> bool:
    return isinstance(exc, GroqRateLimitError)


# ---------------------------------------------------------------------------
# Proactive rate limiter — sliding 60-second window, shared across every
# call in this process (single ingest AND bulk dataset rows alike), so no
# caller can accidentally exceed GROQ_RPM regardless of how fast Groq itself
# responds. This runs BEFORE a request is sent, not after it fails.
# ---------------------------------------------------------------------------

_rate_limit_lock = threading.Lock()
_request_timestamps: collections.deque[float] = collections.deque()


def _throttle_for_rate_limit() -> None:
    with _rate_limit_lock:
        now = time.monotonic()
        while _request_timestamps and now - _request_timestamps[0] >= 60:
            _request_timestamps.popleft()

        if len(_request_timestamps) >= GROQ_RPM:
            wait_seconds = 60 - (now - _request_timestamps[0]) + 0.05
            if wait_seconds > 0:
                logger.info(
                    "Pacing for Groq's %d RPM free-tier limit: waiting %.1fs before the next request "
                    "(this is expected for large dataset imports, not an error)",
                    GROQ_RPM, wait_seconds,
                )
                time.sleep(wait_seconds)
            now = time.monotonic()
            while _request_timestamps and now - _request_timestamps[0] >= 60:
                _request_timestamps.popleft()

        _request_timestamps.append(time.monotonic())


@retry(
    retry=retry_if_exception(_is_rate_limit),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _call_groq(raw_text: str, api_key: str) -> dict:
    _throttle_for_rate_limit()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    text_for_model = raw_text
    if len(text_for_model) > MAX_INPUT_CHARS:
        text_for_model = text_for_model[:MAX_INPUT_CHARS] + "\n...[truncated for extraction]"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text_for_model},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(GROQ_API_URL, headers=headers, json=payload)

    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after")
        logger.warning(
            "Groq rate limit hit (429) despite pacing%s. Backing off and retrying...",
            f" — server asked for {retry_after}s" if retry_after else "",
        )
        raise GroqRateLimitError(resp.text)

    if resp.status_code != 200:
        # 413, 400, 5xx, etc. — none of these are fixed by resending the
        # same request, so this is intentionally NOT retried here.
        raise GroqExtractionError(
            f"Groq API error {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise GroqExtractionError(f"Unexpected Groq response shape: {data}") from exc

    return json.loads(content)


def extract_structured_report(raw_text: str) -> tuple[StructuredBugReport, int]:
    """
    Runs the raw text through Groq's JSON-mode extraction and validates
    the result against StructuredBugReport.

    Returns (structured_report, attempts_used).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqExtractionError(
            "GROQ_API_KEY is not set. Get a free key at console.groq.com "
            "and set it as an environment variable."
        )

    attempts = 0

    # Only retried when the MODEL's output was unusable (malformed JSON or
    # it didn't match our schema) — that's the one failure mode where
    # asking again can plausibly produce a different, valid result.
    # GroqRateLimitError is already handled inside _call_groq; GroqExtractionError
    # (413/400/5xx/etc.) is a request-level problem that retrying won't fix.
    @retry(
        retry=retry_if_exception_type((ValidationError, json.JSONDecodeError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _extract_and_validate() -> StructuredBugReport:
        nonlocal attempts
        attempts += 1
        raw_json = _call_groq(raw_text, api_key)
        return StructuredBugReport.model_validate(raw_json)

    structured = _extract_and_validate()
    return structured, attempts
