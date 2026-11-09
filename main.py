"""Source-restricted RAA roadside-assistance agent.

The agent reads a single approved RAA URL at request time and passes only that
page's visible main-content text to the managed Gemini model. It neither follows
links nor uses other sources, prior knowledge or other providers' products.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from pydantic import BaseModel, Field

app = FastAPI(title="Roadside Assistant", version="1.0.0")

SOURCE_URL = "https://www.raa.com.au/motor/road-service"
SOURCE_HOST = "www.raa.com.au"
SOURCE_PATH = "/motor/road-service"
MODEL = os.getenv("LLM_MODEL", "gemini-3-flash-preview")
TRACER_NAME = "roadside-assistant.gemini"
MAX_SOURCE_CHARACTERS = 28_000
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 roadside-assistant-agent/1.0"
)

# raa.com.au sits behind an edge that silently drops requests which do not look
# like a browser navigation (a minimal User-Agent gets "Server disconnected"
# instead of a response), so send a complete set of navigation headers.
SOURCE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Connection": "keep-alive",
    "Accept-Encoding": "identity",
}

SYSTEM_PROMPT = f"""You are Roadside Assistant, a source-restricted RAA roadside-assistance assistant.
Only look at and use this exact approved source: {SOURCE_URL}
Answer only about RAA Road Service information that is explicitly stated on that exact page: cover levels and their prices, call-out allowances, attendance and towing limits, what is included or excluded, how to get help, and how to join, renew or check cover. Do not use any other webpages, links, sources, prior knowledge, or information about other motoring clubs or insurers. Do not invent, estimate, infer unstated details, or claim a price, fee, eligibility rule, benefit, timeframe or process that is not in the approved page content.
If the approved page does not explicitly support the answer, say: "I can't confirm that from the approved RAA Road Service page. Please review {SOURCE_URL} or call RAA on 13 11 11." Do not guess and do not redirect to another provider.
You are not a dispatch or emergency service: never promise that a patrol has been sent, never quote an arrival time, and never process a membership purchase, claim or refund. If someone is in immediate danger, tell them to contact emergency services now and then ring RAA on 13 11 11.
Keep answers short and practical. Quote prices exactly as they appear on the page and make clear which cover level each figure belongs to.
"""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=6_000)
    session_id: str | None = Field(default=None, max_length=256)


@dataclass
class _CachedSource:
    content: str
    expires_at: float


_source_cache: _CachedSource | None = None


class ApprovedSourceUnavailable(RuntimeError):
    """Raised when the only approved source cannot be safely retrieved."""


class ModelUnavailable(RuntimeError):
    """Raised when the managed model stays throttled after a retry."""


class GuardrailBlocked(RuntimeError):
    """Raised when a platform guardrail rejects the request before the model."""

    def __init__(self, guardrail: str, reason: str) -> None:
        super().__init__(f"{guardrail}: {reason}")
        self.guardrail = guardrail
        self.reason = reason


def _cache_ttl_seconds() -> int:
    try:
        return min(3_600, max(60, int(os.getenv("RAA_SOURCE_TTL_SECONDS", "900"))))
    except ValueError:
        return 900


def _extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(
        ["script", "style", "noscript", "svg", "form", "iframe", "nav", "header", "footer"]
    ):
        element.decompose()
    root = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
        or soup.body
        or soup
    )
    lines = [" ".join(line.split()) for line in root.get_text("\n").splitlines()]
    text = "\n".join(line for line in lines if line)
    if len(text) < 200:
        raise ApprovedSourceUnavailable("Approved source did not provide enough visible content")
    return text[:MAX_SOURCE_CHARACTERS]


def get_approved_source() -> str:
    """Fetch only the approved RAA URL without following redirects or links."""
    global _source_cache
    now = time.monotonic()
    if _source_cache and _source_cache.expires_at > now:
        return _source_cache.content
    last_error: Exception | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(1.5 * attempt)
        try:
            response = httpx.get(
                SOURCE_URL,
                headers=SOURCE_HEADERS,
                follow_redirects=False,
                timeout=25.0,
            )
            if response.status_code != 200:
                raise ApprovedSourceUnavailable(
                    f"Approved source returned HTTP {response.status_code}"
                )
            if response.url.host != SOURCE_HOST or response.url.path.rstrip("/") != SOURCE_PATH:
                raise ApprovedSourceUnavailable(
                    "Approved source request resolved to an unapproved URL"
                )
            source = _extract_visible_text(response.text)
            break
        except ApprovedSourceUnavailable:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            # the RAA edge occasionally closes the connection mid-body; retry
            last_error = exc
            continue
    else:
        raise ApprovedSourceUnavailable("Approved source request failed") from last_error
    _source_cache = _CachedSource(content=source, expires_at=now + _cache_ttl_seconds())
    return source


def _visible_gemini_text(response_body: dict[str, Any]) -> str:
    candidates = response_body.get("candidates")
    candidate = candidates[0] if isinstance(candidates, list) and candidates else None
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    text = "".join(
        str(part.get("text", ""))
        for part in parts or []
        if isinstance(part, dict) and part.get("text") and not part.get("thought", False)
    ).strip()
    if not text:
        raise ValueError("Gemini returned no visible response")
    return text


def _guardrail_block(response: "httpx.Response") -> tuple[str, str] | None:
    """Return (guardrail, reason) when the gateway rejected a request by policy."""
    try:
        body = response.json()
    except (ValueError, TypeError):
        return None
    message = body.get("message") if isinstance(body, dict) else None
    if not isinstance(message, dict):
        return None
    if message.get("action") != "GUARDRAIL_INTERVENED":
        return None
    return (
        str(message.get("interveningGuardrail") or body.get("type") or "unknown"),
        str(message.get("actionReason") or "policy violation"),
    )


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _redacted_trace_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: [redacted]" for m in messages)


def generate_gemini_response(question: str, source: str) -> tuple[str, int, int]:
    base_url = (os.getenv("LLM_PROVIDER_URL") or "").rstrip("/")
    api_key = os.getenv("LLM_PROVIDER_KEY") or ""
    if not base_url or not api_key:
        raise HTTPException(
            status_code=503,
            detail="Agent Manager managed model binding is not available",
        )
    endpoint = f"{base_url}/v1beta/models/{quote(MODEL, safe='._-')}:generateContent"
    messages = [{"role": "user", "content": question}]
    payload = {
        "systemInstruction": {
            "parts": [{"text": f"{SYSTEM_PROMPT}\n\nApproved page content:\n{source}"}]
        },
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 900},
    }
    tracer = trace.get_tracer(TRACER_NAME)
    with tracer.start_as_current_span(f"chat {MODEL}", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.system", "gemini")
        span.set_attribute("gen_ai.provider.name", "gemini")
        span.set_attribute("gen_ai.request.model", MODEL)
        span.set_attribute("gen_ai.input.messages", _redacted_trace_messages(messages))
        span.set_attribute("app.source.url", SOURCE_URL)
        try:
            response = None
            for attempt in range(2):
                response = httpx.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json", "API-Key": api_key},
                    timeout=60.0,
                )
                # The shared Gemini deployment throttles bursts (HTTP 429) and
                # occasionally returns 5xx; one short retry usually clears it.
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 0:
                        span.set_attribute("app.model.retry_status", response.status_code)
                        time.sleep(4.0)
                        continue
                    raise ModelUnavailable(
                        f"Managed Gemini model is busy (HTTP {response.status_code})"
                    )
                break
            assert response is not None
            # A gateway guardrail rejects the call before it reaches the model
            # (HTTP 422/403 with action GUARDRAIL_INTERVENED). Surface that as a
            # policy block rather than as a model failure.
            if 400 <= response.status_code < 500:
                blocked = _guardrail_block(response)
                if blocked is not None:
                    raise GuardrailBlocked(*blocked)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("Gemini returned an invalid response")
            answer = _visible_gemini_text(body)
        except GuardrailBlocked as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, "guardrail blocked the request"))
            span.set_attribute("error.type", type(exc).__name__)
            span.set_attribute("app.guardrail", exc.guardrail or "unknown")
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Request blocked by the platform guardrail "
                    f"'{exc.guardrail or 'unknown'}' ({exc.reason or 'policy violation'})."
                ),
            ) from exc
        except ModelUnavailable as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, "managed model unavailable"))
            span.set_attribute("error.type", type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail=(
                    "The platform-managed Gemini model is rate limited right now. "
                    "Please try again in a few seconds."
                ),
            ) from exc
        except (httpx.HTTPError, TypeError, ValueError, KeyError, IndexError) as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, "Gemini request failed"))
            span.set_attribute("error.type", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Managed Gemini request failed") from exc
        usage = body.get("usageMetadata") if isinstance(body.get("usageMetadata"), dict) else {}
        input_tokens = _nonnegative_int(usage.get("promptTokenCount"))
        output_tokens = _nonnegative_int(usage.get("candidatesTokenCount"))
        span.set_attribute("gen_ai.response.model", str(body.get("modelVersion") or MODEL))
        span.set_attribute(
            "gen_ai.output.messages",
            _redacted_trace_messages([{"role": "assistant", "content": answer}]),
        )
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_status(Status(StatusCode.OK))
        return answer, input_tokens, output_tokens


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "agent": "roadside-assistant",
        "model": MODEL,
        "source": SOURCE_URL,
        "managedModelConfigured": bool(
            os.getenv("LLM_PROVIDER_URL") and os.getenv("LLM_PROVIDER_KEY")
        ),
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    tracer = trace.get_tracer("roadside-assistant.agent")
    with tracer.start_as_current_span("invoke_agent", kind=SpanKind.INTERNAL) as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "roadside-assistant")
        span.set_attribute("gen_ai.request.model", MODEL)
        span.set_attribute(
            "gen_ai.input.messages",
            _redacted_trace_messages([{"role": "user", "content": request.message}]),
        )
        span.set_attribute("app.source.url", SOURCE_URL)
        try:
            source = get_approved_source()
            answer, input_tokens, output_tokens = generate_gemini_response(request.message, source)
        except ApprovedSourceUnavailable as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, "approved source unavailable"))
            raise HTTPException(
                status_code=503,
                detail="The approved RAA Road Service page is temporarily unavailable, so no answer can be provided.",
            ) from exc
        except HTTPException as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc.detail)))
            raise
        span.set_attribute(
            "gen_ai.output.messages",
            _redacted_trace_messages([{"role": "assistant", "content": answer}]),
        )
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_status(Status(StatusCode.OK))
        return {
            "response": answer,
            "sessionId": request.session_id,
            "model": MODEL,
            "source": SOURCE_URL,
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
