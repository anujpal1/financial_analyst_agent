# Evidence-Grounded Financial Research Workbench

A local Streamlit application for auditable public-company research. It combines
provider-labelled market and statement data, SEC filing facts, deterministic historical
analytics, transparent DCF scenarios, claim-level evidence, a consistency gate, optional
PDF evidence, and qualitative synthesis from a user-selected language model.

Financial values, ratios, trend observations, scorecards, and valuation comparisons are
calculated in code. The language model does not populate numeric dashboard fields and is
not allowed to invent facts, citations, recommendations, or confidence percentages.

## Main features

- OpenAI, Google Gemini, Anthropic, and local Ollama support through one research
  workflow.
- Session-only password inputs for cloud keys; page load makes no paid provider call.
- One canonical yfinance market object for price, trading date, timestamp, market state,
  previous close, market capitalisation, and six-month price history.
- Up to five aligned annual periods for revenue, income, cash flow, capital expenditure,
  free cash flow, cash, debt, and diluted shares.
- An executive dashboard sourced only from validated structured data.
- Deterministic annual growth, CAGR, margins, cash/debt, and share-count observations.
- A transparent six-component financial scorecard with visible rules and calculation
  traces. Missing inputs are labelled `Not scored — insufficient data`.
- Bear, base, and bull DCF scenarios plus a discount-rate / terminal-growth sensitivity
  table. Missing essential inputs stop the public valuation tool.
- Claim-to-evidence links for market, statement, filing, calculation, transcript, and
  uploaded-document evidence.
- A deterministic consistency validator with pass, warning, and blocking outcomes; one
  controlled report retry; and an explicit partial-report fallback.
- Dataset-specific source records, URL/title/accession deduplication, and a data-quality
  panel.
- Deterministic news relevance filters with company-signal requirements and duplicate
  removal. News is never represented as a transcript.
- Full-document, in-memory PDF extraction with filename sanitization, upload limits,
  page metadata, and untrusted-content handling.
- Downloadable Markdown research reports and a fully offline mocked test/evaluation
  suite.

The workbench intentionally does not implement peer comparison, portfolio management,
automated trading, broker integration, crypto analysis, social-media sentiment, or
investment commands.

## Supported providers

| Provider | Package | Credential |
| --- | --- | --- |
| OpenAI | `langchain-openai` | UI key or `OPENAI_API_KEY` |
| Google Gemini | `langchain-google-genai` | UI key or `GOOGLE_API_KEY` |
| Anthropic | `langchain-anthropic` | UI key or `ANTHROPIC_API_KEY` |
| Ollama | `langchain-ollama` | No API key |

The UI offers suggested models and a custom-model field because account availability
changes over time. Known reasoning-model families that do not accept temperature are
called without that parameter.

## Research pipeline

1. Validate the question and deterministically resolve one ticker.
2. Collect a canonical market snapshot, aligned annual statements, and optional SEC,
   news, transcript, and uploaded-PDF evidence.
3. Compare like-period provider and SEC facts without averaging disagreements.
4. Calculate historical trends, dashboard metrics, score components, DCF scenarios, and
   evidence coverage locally.
5. Build dataset-specific source records and claim-level evidence links.
6. Ask the selected model only for constrained qualitative conclusions, risks, and
   assumptions.
7. Assemble the deterministic report and run consistency validation.
8. If blocked, retry synthesis once; if the issue remains, return a clearly labelled
   partial report.

Quick analysis skips company news. Standard and Detailed analysis collect relevant news;
Detailed uses the same verified source pipeline and fuller report surface. A refresh is
an explicit rerun and creates new per-analysis source clients. API keys and provider
clients are never cached.

## Interface

The restrained finance-style interface includes:

- A compact header with provider status, data timestamp, and disclaimer.
- Grouped LLM, optional-source, analysis, and session controls.
- Research question, ticker, PDF, analysis-depth, and DCF input controls.
- Overview, Financials, Valuation, Evidence, Sources, and Full report tabs.
- Structured metric cards, canonical price history, annual financial charts, score
  traces, DCF sensitivity, claim evidence, validation details, and data-quality rows.

Screenshot placeholders:

- `docs/screenshots/workbench-overview.png` — executive dashboard and data quality.
- `docs/screenshots/workbench-financials.png` — annual trends and scorecard.
- `docs/screenshots/workbench-evidence.png` — claims, sources, and validation.

These are placeholders only; no generated screenshots are committed.

## Project structure

```text
.
├── .env.example
├── .gitignore
├── .python-version
├── .streamlit/
│   └── config.toml
├── app.py
├── financial_analyst/
│   ├── __init__.py
│   ├── analytics.py
│   ├── config.py
│   ├── documents.py
│   ├── evidence.py
│   ├── llm.py
│   ├── market.py
│   ├── models.py
│   ├── reporting.py
│   ├── sec.py
│   ├── security.py
│   ├── tickers.py
│   ├── tools.py
│   ├── transcripts.py
│   ├── valuation.py
│   └── workflow.py
├── tests/
│   ├── fixtures/
│   ├── conftest.py
│   ├── test_config_and_llm.py
│   ├── test_documents.py
│   ├── test_evaluation_suite.py
│   ├── test_security_and_smoke.py
│   ├── test_sources.py
│   ├── test_tickers.py
│   ├── test_valuation.py
│   └── test_workflow_and_reporting.py
├── pyproject.toml
├── README.md
└── uv.lock
```

`app.py` is the only UI entry point. `pyproject.toml` is the dependency and tool
configuration source; `uv.lock` contains the resolved environment.

## Requirements

- Windows 10 or later
- Python 3.11, 3.12, or 3.13
- PowerShell
- Internet access for live market/filing sources and cloud providers
- Ollama only when using a local model

Automated tests and evaluations require no network, provider key, paid data service, or
Ollama process. A test-level socket guard fails any accidental live connection.

## Windows PowerShell setup

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If activation is blocked for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

With `uv`, the equivalent reproducible setup is:

```powershell
uv sync --extra dev
```

## Provider setup

### OpenAI

Select OpenAI and paste the key in the password field, or set it for the current shell:

```powershell
$env:OPENAI_API_KEY = "your-key"
```

### Google Gemini

```powershell
$env:GOOGLE_API_KEY = "your-key"
```

### Anthropic

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
```

### Ollama

Start Ollama and pull a supported model:

```powershell
ollama serve
ollama pull llama3.1:8b
```

Then select Ollama and confirm the base URL, normally `http://localhost:11434`.

### Optional data providers

SEC EDGAR requires a descriptive User-Agent containing a contact email:

```powershell
$env:SEC_USER_AGENT = "FinancialResearch/1.0 you@example.com"
```

Actual earnings-call transcript retrieval is an optional Financial Modeling Prep
enhancement:

```powershell
$env:FMP_API_KEY = "your-key"
```

Without a valid SEC User-Agent or FMP key, those datasets return clear unavailable
statuses and the analysis continues where safe. Core market and statement collection
uses yfinance.

You may copy `.env.example` to `.env` for local environment configuration. `.env` is
ignored by Git. UI-entered keys take precedence and are not written to it.

## Run the application

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Initial page load only renders controls. A provider is contacted after **Test
connection** or **Run analysis** is selected.

## Use the workbench

1. Select OpenAI, Google Gemini, Anthropic, or Ollama.
2. Select a suggested model or enter a custom model name.
3. Paste a cloud API key in the password field, or use Ollama locally.
4. Test the connection.
5. Enter a ticker and research question.
6. Select Quick, Standard, or Detailed analysis.
7. Optionally upload a PDF and adjust visible DCF assumptions.
8. Run the analysis.
9. Inspect dashboard, financial trends, valuation, evidence, sources, and validation.
10. Download the full Markdown report.

## Tests and quality checks

Run the complete offline suite:

```powershell
python -m pytest
```

Run the compact deterministic evaluation suite:

```powershell
python -m pytest tests\test_evaluation_suite.py
```

Run static and compilation checks:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m compileall app.py financial_analyst tests
```

The evaluation suite covers ticker extraction, annual/quarterly separation, annual
alignment, revenue growth, margins, net cash, scoring, DCF sensitivity, missing inputs,
claim support, conflicts, canonical price reuse, source deduplication, news relevance,
validation retry/fallback, and secret redaction. Results are reported as test outcomes;
the project does not claim an unmeasured accuracy percentage.

## Security notes

- UI key fields use password inputs and remain in Streamlit session state only.
- Resetting the session removes UI-entered credentials, uploads, results, and per-run
  clients.
- API keys are not placed in reports, prompts shown to the user, caches, or logs.
- Provider error messages are redacted and length-bounded.
- Local logs contain event types, not full prompts, keys, or uploaded text.
- Source caches contain public response data only; API keys and provider clients are not
  cached.
- PDFs are parsed from uploaded bytes. No tool accepts an arbitrary local path.
- Uploaded filenames are sanitized, size-limited, and assigned opaque IDs.
- PDF text is untrusted evidence and cannot override system instructions.
- The application has no shell, broker, trading, or persistent-memory tool.
- `.env`, logs, reports, caches, virtual environments, archives, and generated outputs
  are ignored.

If a credential has ever appeared in version history, revoke it at the provider even
after removing the current file.

## Financial disclaimer

This software is for informational, educational, and portfolio-project use. It is not
financial, investment, legal, accounting, or tax advice. Source data may be delayed,
incomplete, or incorrect. DCF values are highly assumption-sensitive. Verify material
facts in original filings and consult a qualified professional before making decisions.

## Known limitations

- yfinance is an unofficial third-party interface and may be delayed, incomplete, or
  temporarily blocked.
- SEC Company Facts coverage and taxonomy vary by issuer. The application preserves
  filing form, date, fiscal context, unit, period, and accession when provided.
- The scorecard uses general, documented rules and is not sector-specific.
- DCF is a simplified five-year model; it is not a substitute for a complete forecast.
- News quality depends on upstream metadata. The deterministic relevance layer reduces,
  but cannot eliminate, irrelevant stories.
- Optional transcript coverage depends on the FMP account and plan.
- PDF extraction is text-only; scanned documents require OCR, which is not enabled.
- PDF retrieval uses compact lexical relevance rather than embeddings.
- The workflow analyzes one primary ticker per run and does not perform peer comparison.
- Reports download as Markdown; PDF export and persistent report history are not
  included.
- Offline mocks validate application behavior, not the availability or correctness of
  live third-party services.
