"""Optional FMP transcript access that never substitutes news for a transcript."""

from __future__ import annotations

from typing import Any

import requests
from pydantic import SecretStr

from financial_analyst.models import Availability, DataResult, EvidenceRef
from financial_analyst.security import safe_error_message


class FMPTranscriptClient:
    """Retrieve an earnings-call transcript only when an optional FMP key exists."""

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        timeout: float,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()

    def fetch(self, ticker: str, year: int, quarter: int) -> DataResult:
        source = "Financial Modeling Prep earnings-call transcript"
        if quarter not in {1, 2, 3, 4}:
            return DataResult.unavailable(
                name="earnings_transcript",
                source=source,
                message="Transcript quarter must be between 1 and 4.",
                content_type="transcript",
            )
        if year < 1990 or year > 2200:
            return DataResult.unavailable(
                name="earnings_transcript",
                source=source,
                message="Transcript year is outside the supported range.",
                content_type="transcript",
            )
        if self.api_key is None or not self.api_key.get_secret_value().strip():
            return DataResult.unavailable(
                name="earnings_transcript",
                source=source,
                message=(
                    f"Transcript unavailable for {ticker} Q{quarter} {year}: "
                    "the optional FMP_API_KEY is not configured."
                ),
                content_type="transcript",
            )

        endpoint = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}"
        secret = self.api_key.get_secret_value()
        try:
            response = self.session.get(
                endpoint,
                params={
                    "quarter": quarter,
                    "year": year,
                    "apikey": secret,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload: Any = response.json()
            if not isinstance(payload, list) or not payload:
                return DataResult.unavailable(
                    name="earnings_transcript",
                    source=source,
                    message=f"No transcript was returned for {ticker} Q{quarter} {year}.",
                    content_type="transcript",
                )
            transcript = str(payload[0].get("content") or "").strip()
            if not transcript:
                return DataResult.unavailable(
                    name="earnings_transcript",
                    source=source,
                    message=f"No transcript text was returned for {ticker} Q{quarter} {year}.",
                    content_type="transcript",
                )
            truncated = len(transcript) > 50_000
            return DataResult(
                name="earnings_transcript",
                status=Availability.PARTIAL if truncated else Availability.AVAILABLE,
                source=source,
                values={
                    "ticker": ticker,
                    "year": year,
                    "quarter": quarter,
                    "text": transcript[:50_000],
                    "truncated_for_analysis": truncated,
                },
                evidence=[
                    EvidenceRef(
                        source=source,
                        url=endpoint,
                        fiscal_year=year,
                        fiscal_period=f"Q{quarter}",
                    )
                ],
                content_type="transcript",
            )
        except (requests.RequestException, ValueError, TypeError) as error:
            return DataResult.unavailable(
                name="earnings_transcript",
                source=source,
                message=safe_error_message(
                    error,
                    context=f"Transcript unavailable for {ticker} Q{quarter} {year}",
                    secrets=[secret],
                ),
                content_type="transcript",
            )
