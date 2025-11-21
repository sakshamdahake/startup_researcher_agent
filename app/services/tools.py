import os
import uuid
import time
import requests
from typing import List, Dict, Any, Optional
import json
import re

from app.llm.openai_llm import get_llm
from app.llm.prompts import (
    get_chunk_summary_prompt,
    get_synthesis_prompt,
    get_final_report_prompt,
    FinalReport,
)
from app.rag.splitter import split_documents_with_metadata, get_recursive_splitter
from app.rag.retriever import get_relevant_documents
from app.rag.vectorstore import upsert_documents

try:
    from langchain_core.messages import HumanMessage
except Exception:
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content


try:
    from langchain_core.documents import Document
except Exception:
    class Document:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}


def call_llm(llm: Any, prompt: str) -> str:
    """
    Consistent LLM call: use llm.invoke([HumanMessage(prompt)]) where available.
    Extract text from common LangChain/OpenAI response shapes.
    """
    try:
        resp = llm.invoke([HumanMessage(prompt)])
    except Exception as e_invoke:
        try:
            resp = llm.generate([HumanMessage(prompt)])  # some versions
        except Exception:
            try:
                resp = llm(messages=[HumanMessage(prompt)])
            except Exception as e:
                raise RuntimeError(f"LLM call failed (invoke/generate/messages): {e_invoke}; {e}") from e

    if hasattr(resp, "content"):
        return resp.content
    if hasattr(resp, "generations"):
        gens = getattr(resp, "generations", [])
        if gens and isinstance(gens, list):
            first = gens[0]
            if isinstance(first, list):
                return getattr(first[0], "text", str(first[0]))
            return getattr(first[0], "text", str(first[0]))
    return str(resp)


### Tool base class + helpers ###

class Tool:
    name: str
    description: str

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, *args, **kwargs):
        raise NotImplementedError


def _make_id(prefix: str = "chunk") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


### Web-search tool ###
class WebSearchTool(Tool):
    """
    Uses SERPAPI if configured via SERPAPI_API_KEY, else returns empty list.
    Returns a list of dicts: {title, url, snippet}
    """

    def __init__(self):
        super().__init__("web_search", "Search the web for URLs and snippets (SerpAPI).")
        self.api_key = os.getenv("SERPAPI_API_KEY")

    def run(self, query: str, num: int = 20) -> List[Dict[str, str]]:
        if not self.api_key:
            return []
        params = {"q": query, "api_key": self.api_key, "num": num}
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        results: List[Dict[str, str]] = []
        for r in data.get("organic_results", [])[:num]:
            results.append({"title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet")})
        return results


### Fetch and Normalize tool ###
class FetchUrlTool(Tool):
    """
    Basic fetcher that returns a Document with cleaned text.
    Uses requests + naive html -> text stripping (BeautifulSoup).
    """

    def __init__(self):
        super().__init__("fetch_url", "Fetch a webpage and return text as Document.")

    def _html_to_text(self, html: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        text = soup.get_text(separator="\n")
        return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    def run(self, url: str) -> Document:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        text = self._html_to_text(resp.text)
        return Document(page_content=text, metadata={"source": url})


### PDF loader ###
class PDFLoaderTool(Tool):
    """
    Load a local PDF and return a list of Documents (each with page content).
    Requires pdfplumber or PyPDF2.
    """

    def __init__(self):
        super().__init__("pdf_loader", "Load PDF path and return list of Documents (pages).")

    def run(self, path: str) -> List[Document]:
        docs: List[Document] = []
        # prefer pdfplumber for better extraction; fall back to PyPDF2
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    docs.append(Document(page_content=text, metadata={"source": path, "page": i + 1}))
            return docs
        except Exception:
            from PyPDF2 import PdfReader

            reader = PdfReader(path)
            for i, p in enumerate(reader.pages):
                try:
                    text = p.extract_text() or ""
                except Exception:
                    text = ""
                docs.append(Document(page_content=text, metadata={"source": path, "page": i + 1}))
            return docs


### Split + Embed tool (wrapper) ###
class SplitAndEmbedTool(Tool):
    """
    Splits documents into chunks and upserts to vectorstore.
    Returns list of chunk metadata records inserted (plain dicts).
    """

    def __init__(self, namespace: str = ""):
        super().__init__("split_and_embed", "Split documents to chunks and upsert embeddings to vectorstore.")
        self.namespace = namespace
        self.splitter = get_recursive_splitter()

    def run(self, documents: List[Document]) -> List[Dict[str, Any]]:
        split_docs = split_documents_with_metadata(documents, splitter=self.splitter)
        now = int(time.time())
        records: List[Dict[str, Any]] = []
        for d in split_docs:
            _id = _make_id("chunk")
            meta = d.metadata or {}
            # only include page if not None
            rec: Dict[str, Any] = {
                "_id": _id,
                "chunk_text": d.page_content,
                "timestamp": now,
            }
            if meta.get("source") is not None:
                rec["source"] = meta.get("source")
            if meta.get("page") is not None:
                # ensure page is a primitive (string or int)
                page_val = meta.get("page")
                if isinstance(page_val, (int, str, float, bool)):
                    rec["page"] = page_val
                else:
                    # convert lists etc to string to avoid Pinecone type errors
                    rec["page"] = str(page_val)
            records.append(rec)

        # Upsert into vectorstore (this function handles metadata-aware upsert)
        upsert_documents(records)
        return records


### Retriever Tool (Wrapper) ###
class DBRetrieverTool(Tool):
    def __init__(self, k: int = 5):
        super().__init__("db_retriever", "Retrieve top-K relevant documents from vectorstore.")
        self.k = k

    def run(self, query: str, k: Optional[int] = None) -> List[Document]:
        k = k or self.k
        docs = get_relevant_documents(query, k=k)
        return docs


### Summarizer Tool ###
class SummarizerTool(Tool):
    def __init__(self, llm: Optional[Any] = None):
        super().__init__("summarizer", "Summarize a chunk of text (topline + bullets).")
        self.llm = llm or get_llm()

    def run(self, chunk_text: str, metadata: Optional[dict] = None) -> str:
        prompt = get_chunk_summary_prompt(chunk_text=chunk_text, metadata=repr(metadata or {}))
        return call_llm(self.llm, prompt)


### Synthesizer tool ###
class SynthesizerTool(Tool):
    def __init__(self, llm: Optional[Any] = None):
        super().__init__("synthesizer", "Synthesize multiple retrieved docs + web findings into a concise synthesis.")
        self.llm = llm or get_llm()

    def run(self, query: str, retrieved_docs: List[Dict], web_findings: List[Dict]) -> str:
        prompt = get_synthesis_prompt(query=query, retrieved_docs=retrieved_docs, web_findings=web_findings)
        return call_llm(self.llm, prompt)


CHUNK_ID_RE = re.compile(r"^chunk-[0-9a-fA-F\-]+$")

def _ensure_list_of_str(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        # split on newlines if multi-line, else return single-element list
        parts = [p.strip() for p in re.split(r"\n\s*\n|\r\n|\n", x) if p.strip()]
        return parts if len(parts) > 1 else [x.strip()]
    return [str(x)]

def _ensure_list_of_dicts_sources(x) -> List[dict]:
    """
    Coerce various shapes into list[dict] for 'sources'.
    - dicts pass through
    - strings that match chunk id -> {"id": ...}
    - strings that look like URLs -> {"url": ...}
    - JSON strings parsed into dicts where possible
    - otherwise -> {"value": str}
    """
    out = []
    if x is None:
        return out
    # flatten lists/tuples
    def _flatten(v):
        if isinstance(v, (list, tuple)):
            for i in v:
                yield from _flatten(i)
        else:
            yield v

    for item in _flatten(x):
        if isinstance(item, dict):
            out.append(item)
            continue
        if isinstance(item, str):
            s = item.strip()
            # try parse JSON dict
            if (s.startswith("{") and s.endswith("}")):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        out.append(parsed)
                        continue
                except Exception:
                    pass
            if CHUNK_ID_RE.match(s):
                out.append({"id": s})
                continue
            if s.startswith("http://") or s.startswith("https://"):
                out.append({"url": s})
                continue
            out.append({"value": s})
            continue
        out.append({"value": str(item)})
    return out

def normalize_final_report_data(data: dict) -> dict:
    """
    Make a best-effort normalization of LLM output to match FinalReport schema.
    This function mutates a shallow copy and returns the normalized dict.
    """
    d = dict(data or {})

    # title
    d["title"] = str(d.get("title") or d.get("name") or "Untitled Report")

    # executive_summary -> string
    exec_sum = d.get("executive_summary") or d.get("summary") or ""
    if isinstance(exec_sum, list):
        exec_sum = "\n".join([str(x) for x in exec_sum])
    d["executive_summary"] = str(exec_sum)

    # key_insights -> list[str]
    d["key_insights"] = _ensure_list_of_str(d.get("key_insights") or d.get("insights") or "")

    # recommendations -> list[str]
    d["recommendations"] = _ensure_list_of_str(d.get("recommendations") or d.get("next_steps") or "")

    # evidence -> list[dict]
    evidence_raw = d.get("evidence") or []
    if isinstance(evidence_raw, list) and evidence_raw and isinstance(evidence_raw[0], str):
        d["evidence"] = [{"excerpt": s} for s in evidence_raw]
    else:
        d["evidence"] = evidence_raw if isinstance(evidence_raw, list) else []

    # sources -> list[dict]
    d["sources"] = _ensure_list_of_dicts_sources(d.get("sources") or d.get("references") or [])

    # sources might still contain some non-dict shapes; ensure final coercion
    final_srcs = []
    for s in d["sources"]:
        if isinstance(s, dict):
            final_srcs.append(s)
            continue
        if isinstance(s, str):
            s2 = s.strip()
            if CHUNK_ID_RE.match(s2):
                final_srcs.append({"id": s2})
            elif s2.startswith("http://") or s2.startswith("https://"):
                final_srcs.append({"url": s2})
            else:
                final_srcs.append({"value": s2})
            continue
        final_srcs.append({"value": str(s)})
    d["sources"] = final_srcs

    return d

class ReportGeneratorTool(Tool):
    def __init__(self, llm=None):
        super().__init__("report_generator", "Produce FinalReport JSON given syntheses and sources metadata.")
        self.llm = llm or get_llm()

    def run(self, query: str, syntheses: List[dict], sources_metadata: List[dict]) -> FinalReport:
        prompt = get_final_report_prompt(query=query, syntheses=syntheses, sources_metadata=sources_metadata)

        # call LLM using best available method
        r = None
        if hasattr(self.llm, "invoke"):
            try:
                r = self.llm.invoke([HumanMessage(prompt)])
            except Exception:
                r = None
        if r is None:
            try:
                r = self.llm.generate([HumanMessage(prompt)])
            except Exception:
                r = None
        if r is None:
            try:
                r = self.llm(messages=[HumanMessage(prompt)])
            except Exception:
                try:
                    r = self.llm(prompt)
                except Exception as e:
                    raise RuntimeError(f"LLM call failed in ReportGeneratorTool: {e}") from e

        # extract raw text from response (support multiple shapes)
        raw = None
        if hasattr(r, "content"):
            raw = r.content
        elif hasattr(r, "generations"):
            try:
                gens = r.generations
                if isinstance(gens, list) and gens:
                    first = gens[0]
                    if isinstance(first, list):
                        raw = getattr(first[0], "text", str(first[0]))
                    else:
                        raw = getattr(first[0], "text", str(first[0]))
            except Exception:
                raw = str(r)
        else:
            raw = str(r)

        # try to find JSON block in raw output
        text = raw.strip() if isinstance(raw, str) else str(raw)
        json_text = None
        if text.startswith("{"):
            json_text = text
        else:
            s = text.find("{")
            e = text.rfind("}")
            if s != -1 and e != -1 and e > s:
                json_text = text[s:e+1]

        if not json_text:
            # If no JSON found, treat raw text as failure
            raise ValueError("Could not find JSON in report output")

        try:
            data = json.loads(json_text)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from LLM output: {e}; raw={text[:4000]}") from e

        # Normalize before constructing FinalReport
        normalized = normalize_final_report_data(data)

        # Construct FinalReport (this is the canonical object downstream)
        report = FinalReport(**normalized)
        return report