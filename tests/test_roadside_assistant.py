import os
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import main

SOURCE_HTML = """
<html><head><style>.x{color:red}</style><script>var a=1</script></head>
<body>
<nav>Insurance Membership Motor Home Travel</nav>
<main>
<h1>Road Service</h1>
<p>24/7 roadside assistance Australia-wide.</p>
<p>Broken down now? Call 13 11 11.</p>
<p>Premium $24.49 per month. Plus $17.87 per month. Standard $11.25 per month.</p>
<p>"""+("RAA Road Service gives you peace of mind. " * 10)+"""</p>
</main>
<footer>Copyright RAA</footer>
</body></html>
"""


class _SourceResponse:
    def __init__(self, text, status_code=200, url=main.SOURCE_URL):
        self.text = text
        self.status_code = status_code
        self.url = httpx.URL(url)


class _GeminiResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "candidates": [{"content": {"parts": [{"text": "Call 13 11 11 for roadside help."}]}}],
            "usageMetadata": {"promptTokenCount": 1200, "candidatesTokenCount": 12},
            "modelVersion": main.MODEL,
        }


class RoadsideAssistantTests(unittest.TestCase):
    def setUp(self):
        main._source_cache = None
        os.environ["LLM_PROVIDER_URL"] = "http://llm.test/proxy"
        os.environ["LLM_PROVIDER_KEY"] = "test-key"
        self.client = TestClient(main.app)

    def test_extraction_keeps_main_content_and_drops_chrome(self):
        text = main._extract_visible_text(SOURCE_HTML)
        self.assertIn("24/7 roadside assistance", text)
        self.assertIn("13 11 11", text)
        self.assertNotIn("Insurance Membership", text)
        self.assertNotIn("Copyright RAA", text)
        self.assertNotIn("var a=1", text)

    def test_source_rejects_unapproved_host(self):
        with patch(
            "httpx.get",
            return_value=_SourceResponse(
                SOURCE_HTML, url="https://evil.example/motor/road-service"
            ),
        ):
            with self.assertRaises(main.ApprovedSourceUnavailable):
                main.get_approved_source()

    def test_source_rejects_error_status(self):
        with patch("httpx.get", return_value=_SourceResponse("nope", status_code=403)):
            with self.assertRaises(main.ApprovedSourceUnavailable):
                main.get_approved_source()

    def test_health(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["agent"], "roadside-assistant")
        self.assertTrue(body["managedModelConfigured"])

    def test_chat_returns_model_answer(self):
        with patch("httpx.get", return_value=_SourceResponse(SOURCE_HTML)), patch(
            "httpx.post", return_value=_GeminiResponse()
        ):
            body = self.client.post("/chat", json={"message": "What number do I call?"}).json()
        self.assertEqual(body["response"], "Call 13 11 11 for roadside help.")
        self.assertEqual(body["source"], main.SOURCE_URL)

    def test_chat_fails_closed_when_source_is_down(self):
        with patch("httpx.get", return_value=_SourceResponse("blocked", status_code=503)):
            response = self.client.post("/chat", json={"message": "Any cover?"})
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
