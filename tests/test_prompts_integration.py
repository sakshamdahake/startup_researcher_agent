# tests/test_prompts_integration.py
import os
import time
import re
import json
import openai
import pytest
from pprint import pprint

from app.llm.openai_llm import get_llm
from app.llm.prompts import (
    get_chunk_summary_prompt,
    get_synthesis_prompt,
    get_final_report_prompt,
    FinalReport,
)

# Try to import the common HumanMessage wrapper; if not available, provide minimal shim
try:
    from langchain_core.messages import HumanMessage
except Exception:
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content

pytestmark = pytest.mark.integration
SKIP_REASON = "OPENAI_API_KEY not set → skipping live prompts integration test."

def _extract_retry_after_from_exception(exc: Exception) -> int | None:
    """
    Try to extract 'retryAfter' seconds from exception text or httpx response headers.
    """
    try:
        txt = str(exc)
        m = re.search(r"retryAfter['\"]?\s*[:=]\s*([0-9]+)", txt)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    for attr in ("response", "httpx_response", "resp"):
        val = getattr(exc, attr, None)
        if val is not None:
            headers = getattr(val, "headers", None) or {}
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra:
                try:
                    return int(float(ra))
                except Exception:
                    pass
    return None


def call_llm_with_retries(llm, prompt_text: str, max_attempts: int = 5, base_backoff: float = 1.0) -> str:
    """
    Call the LLM through several common call patterns and retry on rate-limit / transient errors.
    Returns raw response text.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            # 1) messages-style (chat)
            try:
                r = llm(messages=[HumanMessage(prompt_text)])
                if hasattr(r, "content"):
                    return r.content
                if hasattr(r, "generations"):
                    gens = r.generations
                    if gens:
                        first = gens[0]
                        if isinstance(first, list):
                            return getattr(first[0], "text", str(first[0]))
                        return getattr(first[0], "text", str(first[0]))
                return str(r)
            except Exception:
                pass

            # 2) direct call with string
            try:
                r = llm(prompt_text)
                if isinstance(r, str):
                    return r
                if hasattr(r, "content"):
                    return r.content
                if hasattr(r, "generations"):
                    gens = r.generations
                    if gens:
                        first = gens[0]
                        if isinstance(first, list):
                            return getattr(first[0], "text", str(first[0]))
                        return getattr(first[0], "text", str(first[0]))
                return str(r)
            except Exception:
                pass

            # 3) invoke/generate style
            r = llm.invoke([HumanMessage(prompt_text)])
            if hasattr(r, "content"):
                return r.content
            if hasattr(r, "generations"):
                try:
                    return r.generations[0][0].text
                except Exception:
                    pass
            return str(r)

        except openai.RateLimitError as rte:
            last_exc = rte
            retry_after = _extract_retry_after_from_exception(rte) or (base_backoff * (2 ** (attempt - 1)))
            print(f"[retry {attempt}/{max_attempts}] RateLimitError; sleeping {retry_after}s and retrying...")
            time.sleep(retry_after)
            continue
        except Exception as e:
            last_exc = e
            backoff = base_backoff * (2 ** (attempt - 1))
            print(f"[retry {attempt}/{max_attempts}] transient error: {e!r}; sleeping {backoff}s ...")
            time.sleep(backoff)
            continue

    raise RuntimeError(f"LLM call failed after {max_attempts} attempts. Last error: {last_exc!r}")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason=SKIP_REASON)
def test_prompts_end_to_end_integration():
    """
    Full integration flow:
      1) chunk summary prompt
      2) synthesis prompt
      3) final report prompt (expect JSON matching FinalReport)
    Prints raw responses and validates FinalReport via pydantic.
    """
    llm = get_llm()

    # ---- 1) Chunk summary ----
    sample_chunk = (
        "Subscription meal kits for busy parents: a small pilot showed 2.1 orders/month "
        "and 18% churn after three months; survey: 68% of parents with kids under 5 cite time constraints."
    )
    chunk_prompt = get_chunk_summary_prompt(chunk_text=sample_chunk, metadata="source: pilot_report_2024")
    print("\n--- CHUNK SUMMARY PROMPT PREVIEW ---\n")
    print(chunk_prompt[:1000], "\n")
    chunk_resp = call_llm_with_retries(llm, chunk_prompt)
    print("\n--- CHUNK SUMMARY RESPONSE ---\n")
    print(chunk_resp)
    assert isinstance(chunk_resp, str) and chunk_resp.strip()

    # ---- 2) Synthesis ----
    retrieved_docs = [
        {"id": "doc1", "text": sample_chunk, "meta": {"url": "https://example.com/pilot"}},
        {"id": "doc2", "text": "Competitor X offers weekly meal kits targeted at busy families.", "meta": {"url": "https://example.com/comp"}}
    ]
    web_findings = [{"label": "competitor_scan", "summary": "Found 3 competitors with subscription model"}]

    synth_prompt = get_synthesis_prompt(query="subscription meal kits for busy parents", retrieved_docs=retrieved_docs, web_findings=web_findings)
    print("\n--- SYNTHESIS PROMPT PREVIEW ---\n")
    print(synth_prompt[:1000], "\n")
    synth_resp = call_llm_with_retries(llm, synth_prompt)
    print("\n--- SYNTHESIS RESPONSE ---\n")
    print(synth_resp)
    assert isinstance(synth_resp, str) and synth_resp.strip()

    # ---- 3) Final report (JSON) ----
    syntheses = [{"section": "market_overview", "text": synth_resp}]
    sources_metadata = [
        {"id": "doc1", "url": "https://example.com/pilot", "type": "pdf"},
        {"id": "doc2", "url": "https://example.com/comp", "type": "article"},
    ]

    final_prompt = get_final_report_prompt(query="subscription meal kits for busy parents", syntheses=syntheses, sources_metadata=sources_metadata)
    print("\n--- FINAL REPORT PROMPT PREVIEW ---\n")
    print(final_prompt[:1000], "\n")
    final_resp = call_llm_with_retries(llm, final_prompt)
    print("\n--- FINAL REPORT RAW RESPONSE ---\n")
    print(final_resp)

    # Try to extract JSON
    text = final_resp.strip()
    json_text = None
    if text.startswith("{"):
        json_text = text
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_text = text[start:end+1]

    assert json_text is not None, "Could not find JSON object in final LLM response."

    try:
        parsed = json.loads(json_text)
    except Exception as e:
        print("\nFailed to parse JSON. Raw response (first 2000 chars):\n")
        print(text[:2000])
        raise AssertionError(f"JSON parse failed: {e}")

    # Validate with Pydantic FinalReport
    report = FinalReport(**parsed)
    print("\n--- FINAL REPORT PARSED (Pydantic) ---")
    pprint(report.model_dump())

    # Basic sanity checks
    assert report.title and isinstance(report.title, str)
    assert report.executive_summary and isinstance(report.executive_summary, str)
    assert isinstance(report.key_insights, list)