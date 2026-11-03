# Roadside Assistant

A source-restricted roadside-assistance agent for RAA Road Service, built to run as a
**platform-hosted agent on WSO2 Agent Manager** and to use the platform's managed LLM
provider (Gemini) rather than a hard-coded API key.

At request time the agent fetches one approved page,
<https://www.raa.com.au/motor/road-service>, strips the site chrome, and passes only that
page's visible text to the model. It does not follow links, does not browse anything else,
and refuses questions the approved page does not answer — so cover levels, prices, call-out
and towing limits, joining/renewing steps and the 13 11 11 contact number all come from the
page itself.

## How it uses the platform model

Agent Manager injects two environment variables when you bind an LLM provider to the agent:

| Variable | Meaning |
| --- | --- |
| `LLM_PROVIDER_URL` | Per-agent proxy URL on the platform's AI gateway |
| `LLM_PROVIDER_KEY` | Platform-issued key for that proxy |
| `LLM_MODEL` | Optional model override, defaults to `gemini-3-flash-preview` |

The agent calls `{LLM_PROVIDER_URL}/v1beta/models/{LLM_MODEL}:generateContent` with the
Gemini-native payload and an `API-Key` header, so no Google credentials live in the
container and every call is metered, logged and traceable on the gateway.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness plus the resolved model, source URL and whether the binding is present |
| `POST` | `/chat` | `{"message": "...", "session_id": "optional"}` → `{"response": "...", "model": "...", "source": "..."}` |

Both are traced with OpenTelemetry (`invoke_agent` plus a `chat <model>` span carrying token
usage), which Agent Manager's instrumentation collects and shows under Observability.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8000` | Listen port |
| `LLM_PROVIDER_URL` / `LLM_PROVIDER_KEY` | – | Injected by Agent Manager |
| `LLM_MODEL` | `gemini-3-flash-preview` | Gemini model name |
| `RAA_SOURCE_TTL_SECONDS` | `900` | How long the fetched page text is cached (60–3600) |

## Run locally

```bash
pip install -r requirements.txt
LLM_PROVIDER_URL=http://127.0.0.1:22893/<provider-context> \
LLM_PROVIDER_KEY=<platform key> \
PORT=8000 python main.py
```

```bash
curl -s localhost:8000/health
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"What is the difference between Plus and Premium?"}'
```

Tests: `python -m pytest -q`.

## Deploy on Agent Manager

Create a **platform-hosted agent** from this repository, choose the Python buildpack, add an
LLM provider binding to the Gemini provider, and set the injected variable names to
`LLM_PROVIDER_URL` and `LLM_PROVIDER_KEY`. Add `PORT=8000` to the agent configuration if the
buildpack defaults to 8080. Build, deploy to the `default` environment, then create an API
key under Credentials to call the agent through the gateway.
