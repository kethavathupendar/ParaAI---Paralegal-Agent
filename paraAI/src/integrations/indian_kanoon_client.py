"""
Thin, typed wrapper around Indian Kanoon's API.

Docs: https://api.indiankanoon.org/  (token-based auth, POST form requests)

We only use three endpoints, matching what the Legal Research Agent needs:
  - search()            full text search across judgments/acts
  - get_doc()            full judgment/act text + metadata
  - get_doc_fragment()   short highlighted fragment matching a query,
                         used to cheaply check relevance before pulling a
                         full document.

All requests are retried with exponential backoff since these calls sit on
the critical path of an interactive "ask a question" request but the API
itself is a third party we don't control.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.models.schemas import PrecedentResult


class IndianKanoonError(RuntimeError):
    pass


class IndianKanoonClient:
    def __init__(self, api_token: Optional[str] = None, base_url: Optional[str] = None):
        settings = get_settings()
        self.api_token = api_token or settings.ik_api_token
        self.base_url = (base_url or settings.ik_base_url).rstrip("/")
        if not self.api_token:
            # Don't hard-crash at import time — allow the app to boot and
            # the Legal Research Agent to degrade gracefully if unset.
            self._enabled = False
        else:
            self._enabled = True

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_token}"}

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    async def _post(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self._enabled:
            raise IndianKanoonError(
                "IK_API_TOKEN is not configured; Legal Research Agent cannot query Indian Kanoon."
            )
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, headers=self._headers, params=params)
        if resp.status_code != 200:
            raise IndianKanoonError(
                f"Indian Kanoon API error {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()

    async def search(
        self,
        query: str,
        page_num: int = 0,
        doc_types: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[PrecedentResult]:
        """
        Full text search. `doc_types` follows IK's filter syntax, e.g.
        "judgments" or "acts" — left None to search everything.
        """
        params: dict[str, Any] = {"formInput": query, "pagenum": page_num}
        if doc_types:
            params["formInput"] += f" doctypes:{doc_types}"
        if from_date:
            params["fromdate"] = from_date
        if to_date:
            params["todate"] = to_date

        data = await self._post("/search/", params)
        results: list[PrecedentResult] = []
        for doc in data.get("docs", []):
            results.append(
                PrecedentResult(
                    ik_doc_id=str(doc.get("tid", "")),
                    title=doc.get("title", "Untitled"),
                    court=doc.get("docsource"),
                    date=doc.get("publishdate"),
                    relevance_snippet=_strip_html(doc.get("headline", "")),
                    url=f"https://indiankanoon.org/doc/{doc.get('tid')}/" if doc.get("tid") else None,
                )
            )
        return results

    async def get_doc(self, doc_id: str) -> dict[str, Any]:
        """Full judgment/act text plus metadata (court, bench, date, citing docs)."""
        data = await self._post(f"/doc/{doc_id}/", params={})
        return data

    async def get_doc_fragment(self, doc_id: str, query: str) -> str:
        """A short highlighted fragment of `doc_id` matching `query` — cheap relevance check."""
        data = await self._post(f"/docfragment/{doc_id}/", params={"formInput": query})
        return _strip_html(data.get("fragment", ""))

    async def check_still_good_law(self, doc_id: str) -> dict[str, Any]:
        """
        Uses /origdoc citation graph metadata to flag if a judgment has been
        referenced by later judgments that overrule or distinguish it. IK
        doesn't give a boolean "overruled" flag directly, so we surface the
        citing documents and let the Legal Research Agent / LLM reason about
        them rather than pretending we have ground truth.
        """
        data = await self._post(f"/docmeta/{doc_id}/", params={})
        return {
            "doc_id": doc_id,
            "citedby_count": data.get("citedbycount"),
            "cited_by": data.get("citedby", [])[:10],
        }


def _strip_html(text: str) -> str:
    import re

    return re.sub(r"<[^>]+>", "", text or "").strip()
