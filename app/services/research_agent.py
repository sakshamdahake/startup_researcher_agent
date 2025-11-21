from typing import List, Dict, Any, Optional, Callable

from app.services.tools import (
    WebSearchTool,
    FetchUrlTool,
    PDFLoaderTool,
    SplitAndEmbedTool,
    DBRetrieverTool,
    SummarizerTool,
    SynthesizerTool,
    ReportGeneratorTool,
)

try:
    from app.services.tracer import get_current_tracer
except Exception:
    def get_current_tracer():
        return None


class ResearchAgent:
    """
    Simple synchronous orchestrator wrapping a collection of tools.
    Progress is emitted via optional progress_cb(event_type, payload).

    This implementation is minimal and explicit. It also emits lightweight
    tracer events (via get_current_tracer()) at key milestones so async
    workers (Celery + RedisTracer) can publish progress to clients.
    """

    def __init__(self, namespace: str = "", progress_cb: Optional[Callable] = None):
        self.web = WebSearchTool()
        self.fetch = FetchUrlTool()
        self.pdf = PDFLoaderTool()
        self.split_embed = SplitAndEmbedTool(namespace=namespace)
        self.retriever = DBRetrieverTool()
        self.summarizer = SummarizerTool()
        self.synthesizer = SynthesizerTool()
        self.report_gen = ReportGeneratorTool()

        self.progress_cb = progress_cb or (lambda *a, **k: None)

    def _emit(self, typ: str, payload: dict):
        try:
            self.progress_cb(typ, payload)
        except Exception:
            pass

    def run(self, query: str, pdf_paths: Optional[List[str]] = None, top_k: int = 6) -> Any:
        """
        Run the canonical research pipeline:
          1) web search
          2) fetch pages
          3) load PDFs
          4) split + embed (index)
          5) retrieve
          6) summarize chunks
          7) synthesize
          8) final report generation

        Emits progress via self._emit and best-effort tracer hooks via get_current_tracer().
        """
        pdf_paths = pdf_paths or []
        tracer = get_current_tracer()

        def tracer_progress(step_name: str, payload: dict):
            """Best-effort tracer progress call (non-throwing)."""
            if not tracer:
                return
            try:
                tracer.on_tool_progress("run_pipeline", {"step": step_name, **(payload or {})})
            except Exception:
                try:
                    tracer.on_progress("run_pipeline", {"step": step_name, **(payload or {})})
                except Exception:
                    pass

        self._emit("step", {"state": "searching"})
        tracer_progress("searching", {"query": query})
        try:
            web_results = self.web.run(query=query, num=5)
        except TypeError as e:
            raise RuntimeError(
                "WebSearchTool.run() failed. Did you instantiate the tool (WebSearchTool())? "
                f"Original error: {e}"
            ) from e
        except Exception as e:
            self._emit("step", {"state": "search_failed", "error": str(e)})
            tracer_progress("search_failed", {"error": str(e)})
            web_results = []

        self._emit("progress", {"pct": 10, "web_results": len(web_results)})
        tracer_progress("search_complete", {"count": len(web_results)})

        docs = []
        self._emit("step", {"state": "fetching_urls"})
        tracer_progress("fetching_urls", {"to_fetch": min(len(web_results), 5)})
        for idx, r in enumerate(web_results[:5]):
            url = r.get("url")
            if not url:
                continue
            try:
                d = self.fetch.run(url)
                meta = getattr(d, "metadata", {}) or {}
                meta.update({"source_title": r.get("title")})
                d.metadata = meta
                docs.append(d)
                tracer_progress("fetched_url", {"url": url, "index": idx + 1, "fetched_docs": len(docs)})
            except Exception as e:
                tracer_progress("fetch_error", {"url": url, "error": str(e)})
                continue
        self._emit("progress", {"pct": 25, "fetched_docs": len(docs)})
        tracer_progress("fetch_complete", {"count": len(docs)})

        self._emit("step", {"state": "loading_pdfs"})
        tracer_progress("loading_pdfs", {"pdf_count": len(pdf_paths)})
        for p in pdf_paths:
            try:
                pdf_docs = self.pdf.run(p)
                docs.extend(pdf_docs)
                tracer_progress("pdf_loaded", {"path": p, "pages_loaded": len(pdf_docs)})
            except Exception as e:
                tracer_progress("pdf_load_error", {"path": p, "error": str(e)})
                continue
        self._emit("progress", {"pct": 35, "docs_total": len(docs)})
        tracer_progress("pdfs_complete", {"docs_total": len(docs)})

        self._emit("step", {"state": "indexing"})
        tracer_progress("indexing_start", {"docs": len(docs)})
        records: List[Dict[str, Any]] = []
        if docs:
            try:
                tracer_progress("indexing_split_start", {"docs": len(docs)})
                records = self.split_embed.run(docs)
                tracer_progress("indexing_split_complete", {"indexed": len(records)})
            except Exception as e:
                self._emit("step", {"state": "index_failed", "error": str(e)})
                tracer_progress("index_failed", {"error": str(e)})
                records = []
        else:
            tracer_progress("indexing_skipped", {"reason": "no_docs"})

        self._emit("progress", {"pct": 50, "indexed_chunks": len(records)})
        tracer_progress("indexing_complete", {"indexed_chunks": len(records)})

        self._emit("step", {"state": "retrieving"})
        tracer_progress("retrieving_start", {"query": query, "top_k": top_k})
        try:
            retrieved = self.retriever.run(query=query, k=top_k)
            tracer_progress("retrieved", {"count": len(retrieved)})
        except Exception as e:
            tracer_progress("retrieve_error", {"error": str(e)})
            retrieved = []
        self._emit("progress", {"pct": 65, "retrieved": len(retrieved)})

        self._emit("step", {"state": "summarizing"})
        tracer_progress("summarizing_start", {"chunks": len(retrieved)})
        summaries: List[Dict[str, Any]] = []
        for idx, r in enumerate(retrieved):
            if hasattr(r, "page_content"):
                text = r.page_content
                meta = getattr(r, "metadata", {}) or {}
            else:
                text = r.get("fields", {}).get("chunk_text") or r.get("chunk_text") or ""
                meta = r.get("metadata") or r.get("fields") or {}

            try:
                s = self.summarizer.run(text, metadata=meta)
            except Exception as e:
                s = ""
                tracer_progress("summarize_error", {"index": idx, "error": str(e)})
            summaries.append({"id": meta.get("_id") or meta.get("id"), "summary": s, "metadata": meta})
            tracer_progress("summarized_chunk", {"index": idx + 1, "summary_len": len(s)})

        self._emit("progress", {"pct": 80, "summaries": len(summaries)})
        tracer_progress("summarizing_complete", {"summaries": len(summaries)})

        self._emit("step", {"state": "synthesizing"})
        tracer_progress("synthesizing_start", {"retrieved_count": len(retrieved)})
        try:
            retrieved_for_synth: List[Dict[str, Any]] = []
            for r in retrieved:
                if hasattr(r, "page_content"):
                    text = r.page_content
                    rid = getattr(r, "metadata", {}).get("_id") or getattr(r, "metadata", {}).get("id")
                else:
                    text = r.get("fields", {}).get("chunk_text") or r.get("chunk_text") or ""
                    rid = r.get("metadata", {}).get("_id") or r.get("id") or r.get("_id")
                retrieved_for_synth.append({"id": rid, "text": text})
            synth_text = self.synthesizer.run(query=query, retrieved_docs=retrieved_for_synth, web_findings=web_results)
            tracer_progress("synthesized", {"synth_len": len(synth_text) if isinstance(synth_text, str) else None})
        except Exception as e:
            tracer_progress("synthesize_error", {"error": str(e)})
            synth_text = ""
        self._emit("progress", {"pct": 90})
        tracer_progress("synthesizing_complete", {})

        self._emit("step", {"state": "reporting"})
        tracer_progress("reporting_start", {})
        try:
            sources_metadata = [{"id": rec.get("_id"), "url": rec.get("source")} for rec in records]
            final = self.report_gen.run(
                query=query,
                syntheses=[{"section": "main", "text": synth_text}],
                sources_metadata=sources_metadata,
            )
            tracer_progress("report_generated", {"title": getattr(final, "title", None)})
        except Exception as e:
            self._emit("step", {"state": "report_failed", "error": str(e)})
            tracer_progress("report_failed", {"error": str(e)})
            raise

        self._emit("progress", {"pct": 100, "report_title": getattr(final, "title", None)})
        tracer_progress("done", {"report_title": getattr(final, "title", None)})
        return final