# Agentic Evidence-Grounded Financial Research Workbench

A security-conscious local Streamlit application that uses one bounded research agent
to plan public-company research, select read-only tools, reconcile official and
third-party financial facts, calculate deterministic analytics and an explicit FCFE
valuation, retrieve page-level PDF evidence, verify claims, and expose run provenance.
It is an educational research workbench, not a trading system, investment adviser,
prediction engine, or production financial terminal.

## What makes it different from a finance chatbot

The selected language model plans and synthesizes; it does not invent dashboard
numbers. Market and filing tools return typed records, SEC facts become canonical only
when concept, period, duration, unit, and currency are comparable, and calculations run
in Python. Numeric claims are linked to calculation records and recomputed. Qualitative
lines pass a compact semantic support screen. A deterministic consistency gate can
request one controlled report revision or return a clearly labelled partial result.
Every completed run exposes the selected tools, source states, latency, model-call
count, available token metadata, evidence quality, and validation outcome.

## Research workflow

```text
validated request
      |
model-directed typed plan (native tool calls or guarded JSON fallback)
      |
allowlisted deterministic executor -- tool budget: Quick 3 / Standard 6 / Detailed 8
      |
evidence-gap assessment -- Detailed may revise once
      |
SEC-first reconciliation + deterministic analytics + optional FCFE DCF
      |
claim and calculation lineage + constrained synthesis
      |
deterministic consistency gate -- one bounded revision -- final or partial report
```

This is a single-agent design. It deliberately does not create artificial specialist
agents or expose shell, arbitrary file, trading, or write-capable tools.

### Analysis modes

| Mode | Source contract | Context and behavior |
| --- | --- | --- |
| Quick | Market snapshot and annual statements | Compact context, maximum 3 tool calls, no DCF/transcript/document retrieval unless explicitly requested |
| Standard | Quick plus SEC facts, news, reconciliation, trends, scorecard, and claim verification | Maximum 6 tool calls; optional DCF when requested |
| Detailed | Standard plus deeper evidence, optional transcript and uploaded-document retrieval | Maximum 8 tool calls, semantic claim screen, expanded conflict review, and at most one evidence-gap replan |

The planner chooses from a read-only catalogue. Policy enforcement adds mandatory
mode sources, removes irrelevant conditional tools, validates all inputs, and prevents
the model from executing tools directly.

## Canonical source hierarchy

1. SEC EDGAR Company Facts for compatible reported US filing facts.
2. yfinance for market price, price history, and convenient statement fallback.
3. Optional FMP earnings transcripts for management statements.
4. yfinance news metadata for recent external developments.
5. Uploaded PDFs as user-supplied supplementary evidence.
6. The selected LLM for planning, interpretation, synthesis, and support screening
   only.

The reconciliation record preserves the canonical value and period, every comparable
alternative, absolute and percentage difference, definition compatibility, conflict
state, resolution reason, evidence IDs, and unresolved warning. The system never
averages conflicting sources and does not compare incompatible currencies, units,
frequencies, periods, durations, or accounting concepts.

## Information sources

| Source | Purpose | Authentication | Priority and fallback |
| --- | --- | --- | --- |
| SEC EDGAR Company Facts | Official annual, quarter-only, year-to-date, and instant facts | Descriptive `SEC_USER_AGENT` with contact email | Canonical for comparable reported US filing facts; otherwise returns unavailable or yfinance remains the labelled fallback |
| Yahoo Finance via yfinance | Market snapshot, six-month price history, annual statements, and news metadata | None | Canonical for market observations; statement fallback when official normalization is unavailable |
| Financial Modeling Prep | Optional earnings-call transcript | `FMP_API_KEY` | Management-language evidence only; never replaced by news |
| Uploaded PDF | User-supplied supplementary evidence | None | In-memory, page-aware, size/type checked; never overrides official structured facts |
| Selected LLM | Planning and constrained synthesis | Provider key, except local Ollama | Never a source of invented numeric facts |

Provider clients use timeouts and bounded retries where their libraries or HTTP clients
support them. Public response caches use short TTLs and retrieval timestamps. Market,
statement, news, SEC, and transcript clients are not shared with LLM credentials.
Parsed PDFs and their compact local embeddings are bounded to the current Streamlit
session and cleared on reset.

## SEC fact selection

The SEC client preserves entity, CIK, taxonomy, concept, label, description, unit,
value, form, filing date, accession, fiscal year/period, start/end date, duration,
frame, source URL, retrieval timestamp, and selection reason. Selection separates:

- annual duration facts (normally 300-430 days, preferring 10-K/10-K/A);
- quarter-only duration facts (70-120 days from 10-Q/10-Q/A);
- year-to-date duration facts (121-300 days from 10-Q/10-Q/A); and
- annual or quarterly instant facts.

Amendments and duplicate contexts are resolved by the latest valid comparable filing.
Quarter-only values are not inferred from year-to-date values.

## Financial methodology

All historical calculations use reconciled annual periods. CAGR uses the elapsed time
between fiscal period-end dates, not `observation_count - 1`, so missing fiscal years do
not inflate growth. Provider definitions are retained for revenue, cash, debt, free
cash flow, diluted shares, and market price.

The scorecard is a transparent educational heuristic. It shows every threshold and
contribution, marks missing categories as unscored, and is not calibrated confidence or
a sector-adjusted rating.

### FCFE DCF

The implemented valuation is FCFE only:

- base cash flow is provider free cash flow, defined as operating cash flow minus
  capital expenditure;
- cash flows are discounted with a user-supplied cost-of-equity assumption;
- discounted FCFE produces equity value directly;
- cash is not added and debt is not subtracted after discounting;
- equity value is divided by diluted shares;
- projection horizon is configurable from 1 to 10 years;
- bear, base, and bull assumptions and the sensitivity matrix are explicit;
- invalid terminal-growth/discount-rate combinations are rejected;
- terminal value share, dominance warnings, period, currency, and false-precision
  warnings are displayed; and
- price comparison is omitted when the canonical market price is unavailable.

FCFF is not implemented because the workbench does not silently estimate EBIT tax
normalization, depreciation, or working-capital changes. The DCF is
assumption-sensitive educational analysis, not an intrinsic truth.

## Hybrid PDF retrieval and evidence verification

Text-based PDFs are parsed from uploaded bytes with PyMuPDF. Page-aware overlapping
chunks preserve the sanitized filename, opaque document ID, page, chunk index, and
character offsets. Ranking combines BM25-style lexical relevance with an in-process
finance concept-vector embedding. Identical and heavily overlapping chunks are
deduplicated, zero-relevance passages are not used to fill a quota, and every result
retains a page citation and component scores.

The embedding is a compact deterministic local concept representation, not a
sentence-transformer or a claim of full document understanding. It avoids a model
download and paid embedding API, but its semantic vocabulary is intentionally limited.
Scanned PDFs require OCR, which is not included.

Numeric claims resolve evidence and calculation IDs, recompute allowlisted formulas,
compare output within tolerance, and check period/currency metadata. Qualitative claims
are classified as supported, not verifiable, or unsupported using local semantic
similarity and conservative language rules. Unsupported assertive wording is removed.
The evaluator's imperfect citation precision is reported rather than hidden.

## Prompt-injection and local security controls

- External news, transcript, SEC descriptions, and uploaded text are delimited as
  untrusted evidence in model prompts.
- The model is told never to follow instructions found inside evidence.
- Instruction-like document text is flagged but remains evidence content only.
- No research tool accepts an arbitrary local path or exposes shell/file writes.
- PDF type, size, filename, context, tool calls, retries, and graph recursion are
  bounded.
- UI credential inputs use password widgets and remain in Streamlit session state.
- Reset removes entered credentials, uploads, parsed documents, results, and clients.
- Provider errors are redacted and length bounded; authorization headers and prompts
  are not logged.
- `.env`, caches, coverage files, virtual environments, reports, and evaluation
  artifacts are ignored.

These are practical controls for a local portfolio project, not an enterprise-security
claim.

## LLM providers

| Provider | Package | Credential | Planning behavior |
| --- | --- | --- | --- |
| OpenAI | `langchain-openai` | `OPENAI_API_KEY` or UI password input | Native tool path attempted; strict structured fallback |
| Google Gemini | `langchain-google-genai` | `GOOGLE_API_KEY` or UI password input | Native tool path attempted; strict structured fallback |
| Anthropic | `langchain-anthropic` | `ANTHROPIC_API_KEY` or UI password input | Native tool path attempted; strict structured fallback |
| Ollama | `langchain-ollama` | None | Guarded structured-plan path suitable for local models |

Normal tests mock provider formats, tool calls, malformed output, error redaction, and
token metadata. They do not prove current live service availability. The optional
smoke command makes one explicit, minimal live call using an existing environment key
and never prints the key.

## Run manifest and downloads

Each run creates an in-memory manifest with run ID, UTC analysis time and timezone,
ticker, mode, provider/model, planning method, selected tools, status and latency for
each tool, source retrieval timestamps/statuses, LLM call count, available input/output
tokens, total runtime, completeness, evidence quality, validation status, and
application version. Estimated cost remains null because pricing is not configured.

The UI provides downloads for:

- Markdown report;
- evidence, calculation, and reconciliation JSON; and
- run manifest JSON.

Nothing is persisted automatically.

## Offline evaluation

Run:

```powershell
python -m financial_analyst.evaluation
```

The evaluator runs 53 deterministic fixture tasks and exits nonzero if a mandatory
threshold fails. It does not use an LLM judge as its sole authority and requires no
network, provider key, paid API, SEC endpoint, yfinance endpoint, FMP account, or
Ollama process.

Measured locally on 2026-07-30:

| Metric | Measured | Threshold |
| --- | ---: | ---: |
| Numeric accuracy | 1.000 | 0.950 minimum |
| SEC selection accuracy | 1.000 | 0.900 minimum |
| Tool-plan precision | 1.000 | 0.850 minimum |
| Tool-plan recall | 1.000 | 0.850 minimum |
| Retrieval Recall@K | 1.000 | 0.800 minimum |
| Citation precision | 0.833 | 0.800 minimum |
| Citation recall | 1.000 | 0.900 minimum |
| Unsupported claim rate | 0.000 | 0.050 maximum |
| Consistency pass rate | 1.000 | 0.900 minimum |
| Source-conflict detection | 1.000 | 0.900 minimum |
| Provider-format success | 1.000 | 0.900 minimum |

These are fixture-set measurements, not claims about all companies, filings, PDFs, or
live provider behavior.

## Interface

The restrained high-contrast light UI uses warm neutral surfaces, charcoal text, muted
navy actions, soft amber warnings, light borders, and modest radii. It includes:

- provider/model/key controls and explicit connection testing;
- optional SEC/FMP settings and complete session reset;
- research question, ticker, mode, PDF upload, and optional FCFE assumptions;
- Overview, Financials, Valuation, Research plan, Evidence, Sources, and Full report
  tabs;
- plan/tool outcomes, canonical source labels, reconciliation conflicts, data quality,
  run summary, validation states, and output downloads; and
- readable empty, partial, unavailable, and blocking states.

Initial page load renders controls only; it does not initialize a provider or make a
paid call.

Screenshot placeholders:

- `docs/screenshots/workbench-overview.png`
- `docs/screenshots/workbench-research-plan.png`
- `docs/screenshots/workbench-evidence.png`

No fabricated screenshots are committed.

## Project structure

```text
.
|-- .env.example
|-- .github/workflows/ci.yml
|-- .gitignore
|-- .python-version
|-- .streamlit/config.toml
|-- app.py
|-- financial_analyst/
|   |-- __init__.py
|   |-- analytics.py
|   |-- config.py
|   |-- documents.py
|   |-- evaluation.py
|   |-- evidence.py
|   |-- llm.py
|   |-- market.py
|   |-- models.py
|   |-- reporting.py
|   |-- sec.py
|   |-- security.py
|   |-- tools.py
|   |-- transcripts.py
|   |-- ui.py
|   |-- valuation.py
|   `-- workflow.py
|-- tests/
|   |-- fixtures/
|   |-- conftest.py
|   |-- test_config_and_llm.py
|   |-- test_documents.py
|   |-- test_evaluation_suite.py
|   |-- test_flagship_upgrade.py
|   |-- test_security_and_smoke.py
|   |-- test_sources.py
|   |-- test_tickers.py
|   |-- test_valuation.py
|   `-- test_workflow_and_reporting.py
|-- pyproject.toml
|-- README.md
`-- uv.lock
```

The package stays flat. `app.py` orchestrates Streamlit; `ui.py` presents results.
`pyproject.toml` and `uv.lock` are the canonical dependency inputs.

## Windows PowerShell setup

Requirements: Windows 10 or later, Python 3.11-3.13, PowerShell, and
[uv](https://docs.astral.sh/uv/) for the locked install.

```powershell
cd D:\projects\financial_analyst_agent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install uv
uv sync --extra dev --locked --active
```

If activation is blocked in the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Provider and optional-source setup

Keys can be pasted into the UI, or set for the current PowerShell process:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:GOOGLE_API_KEY = "your-key"
$env:ANTHROPIC_API_KEY = "your-key"
$env:SEC_USER_AGENT = "FinancialAnalystAgent/3.0 your-email@example.com"
$env:FMP_API_KEY = "optional-key"
```

For local Ollama:

```powershell
ollama serve
ollama pull llama3.1:8b
```

Select Ollama in the UI; the default URL is `http://localhost:11434`.
SEC and FMP remain optional. Without them, their sources return structured unavailable
states and safe analysis continues where possible.

## Run and validate

```powershell
streamlit run app.py
python -m pytest
python -m pytest --cov=financial_analyst --cov=app --cov-report=term-missing
python -m financial_analyst.evaluation
python -m ruff check .
python -m ruff format --check .
python -m compileall -q app.py financial_analyst tests
```

Optional live smoke examples:

```powershell
python -m financial_analyst.llm --provider OpenAI --model gpt-4.1-mini
python -m financial_analyst.llm --provider "Google Gemini" --model gemini-2.5-pro
python -m financial_analyst.llm --provider Anthropic --model claude-sonnet-4-6
python -m financial_analyst.llm --provider Ollama --model llama3.1:8b
```

Only run a cloud smoke test after setting the corresponding environment key.

## Resume positioning

Recommended title: **Agentic Evidence-Grounded Financial Research Workbench**

One-line description: Built a bounded single-agent financial research application that
plans read-only tool use, reconciles SEC and provider facts, performs deterministic
FCFE analysis, retrieves PDF evidence, verifies claims, and reports auditable run
provenance.

Truthful resume bullets:

- Designed a typed LangGraph workflow with model-directed tool selection, deterministic
  allowlisted execution, mode-specific budgets, evidence-gap assessment, and bounded
  replanning across four LLM providers.
- Implemented duration-aware SEC fact selection, SEC-first source reconciliation,
  elapsed-fiscal-time CAGR, a financially consistent FCFE DCF, and recomputed
  claim/calculation lineage.
- Built page-aware BM25 plus local concept-vector PDF retrieval, prompt-injection
  controls, claim/report verification, an offline 53-task benchmark, 112 network-
  isolated tests, and downloadable run provenance.

Relevant skills: Python, Pydantic, LangGraph, LangChain, Streamlit, RAG, lexical and
vector retrieval, SEC XBRL/Company Facts, yfinance, financial modelling, evaluation,
pytest, Ruff, security-conscious prompt construction, typed data modelling, and local
LLM integration.

Claims to avoid: multi-agent, autonomous trading, investment advice, production ready,
enterprise security, sentence-transformer retrieval, live provider certification,
real-time guaranteed data, calibrated confidence, research-grade valuation, complete
hallucination elimination, or perfect citation verification.

Likely interview questions:

1. How does the planner choose tools without letting the model execute them?
2. When is an SEC fact comparable enough to replace a provider statement value?
3. Why is the DCF FCFE, and why is there no post-discount cash/debt bridge?
4. How are lexical and semantic scores combined, cached, and evaluated?
5. How do calculation lineage and the consistency gate prevent unsupported claims?

## Limitations

- yfinance is an unofficial third-party interface and can be delayed, incomplete, or
  temporarily blocked.
- SEC Company Facts coverage and taxonomy vary by issuer; reconciliation currently
  canonicalizes compatible annual revenue and net income and keeps other statement
  metrics as definition-labelled provider fallbacks.
- The compact concept embedding has limited vocabulary and is not a learned
  sentence-transformer; semantic recall outside mapped finance concepts will be weaker.
- PDF parsing is text only; there is no OCR, table reconstruction, cross-encoder, or
  persistent vector index.
- Qualitative verification is a conservative local similarity heuristic, not a trained
  NLI model or proof of factual entailment.
- Provider capability flags and normal compatibility tests are mocked/configured
  assumptions; live services and specific model names can change.
- Token usage is recorded only when a provider supplies it. Cost is not estimated.
- The evaluator is small and synthetic. Its scores should not be generalized beyond
  the 53 included fixtures.
- Citation precision is 0.833 on the current fixtures, not perfect.
- The system supports one primary ticker per run and has no peer-comparison engine.
- The general scorecard is not sector adjusted or calibrated.
- FCFF, forecasting, OCR, portfolio analysis, broker connectivity, trading, and
  persistent report history are intentionally absent.
- Several mature finance/provider modules remain longer than the preferred 550-line
  target; responsibilities are kept cohesive to avoid file sprawl, but further
  extraction may improve maintainability.

## Financial disclaimer

This software is for informational, educational, and portfolio-project use only. It is
not financial, investment, legal, accounting, or tax advice, and it does not recommend
buying, holding, or selling a security. Source data may be delayed, incomplete, or
incorrect. DCF outputs are highly assumption sensitive. Verify material facts in
original filings and consult qualified professionals before making decisions.
