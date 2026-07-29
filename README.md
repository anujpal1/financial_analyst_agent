# Evidence-Aware Financial Research

Evidence-Aware Financial Research is a local Streamlit application for investigating a
public-company ticker with provider-labelled market data, official SEC filing facts,
transparent valuation scenarios, optional PDF evidence, and a user-selected language
model. Financial values are collected and calculated deterministically; the language
model is limited to qualitative synthesis and is instructed not to invent facts,
recommendations, citations, or confidence percentages.

## Key features

- Select OpenAI, Google Gemini, Anthropic, or a local Ollama model in the UI.
- Paste a cloud API key into a password field for the current session, or use an
  environment variable.
- Use Ollama without an API key.
- Retrieve market snapshots, statements, price history, and labelled news through
  `yfinance`.
- Retrieve annual and quarterly facts from SEC EDGAR with filing metadata.
- Run bear, base, and bull discounted-cash-flow scenarios only when real base free cash
  flow is available.
- Upload a PDF safely in memory, extract every page, and preserve page numbers in
  evidence.
- Produce a stable Markdown report and download it from the application.
- Continue with clearly labelled gaps when SEC, news, transcript, or optional-provider
  data is unavailable.
- Run a fully offline automated test suite with mocked provider and financial sources.

## Supported LLM providers

| Provider | LangChain package | Credential |
| --- | --- | --- |
| OpenAI | `langchain-openai` | `OPENAI_API_KEY` or UI key |
| Google Gemini | `langchain-google-genai` | `GOOGLE_API_KEY` or UI key |
| Anthropic | `langchain-anthropic` | `ANTHROPIC_API_KEY` or UI key |
| Ollama | `langchain-ollama` | No API key |

The UI includes practical model choices and a custom-model field because model
availability differs by account and changes over time. Some current reasoning models
ignore or reject sampling controls; the provider factory omits temperature for those
known model families while retaining the requested value for compatible models.

Provider documentation:

- [OpenAI models](https://platform.openai.com/docs/models)
- [Gemini models](https://ai.google.dev/gemini-api/docs/models)
- [Anthropic model IDs](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions)
- [Ollama model library](https://ollama.com/library)

## Architecture overview

The application uses one provider-independent LangGraph workflow:

1. Validate the question and resolve one ticker.
2. Invoke deterministic tools from one canonical registry.
3. Collect typed `DataResult` objects from yfinance, SEC EDGAR, optional transcript
   access, and uploaded-document retrieval.
4. Run DCF scenarios only when the request asks for valuation and real free cash flow
   exists.
5. Compare like-period SEC and provider facts and report material conflicts without
   averaging them.
6. Render all numeric sections deterministically.
7. Ask the selected chat model for constrained qualitative conclusions, risks, and
   assumptions.
8. Apply final report guardrails and attach evidence quality plus the disclaimer.

The graph receives a common LangChain chat-model interface from
`financial_analyst/llm.py`. Provider changes do not change graph code. There is no
persistent chat or vector memory; each Streamlit session has a unique identifier.

## Project structure

```text
.
├── .env.example
├── .gitignore
├── .python-version
├── app.py
├── financial_analyst/
│   ├── __init__.py
│   ├── config.py
│   ├── documents.py
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
│   └── test_*.py
├── pyproject.toml
├── README.md
└── uv.lock
```

`app.py` is the only application entry point. `pyproject.toml` is the canonical
dependency and tool configuration. `uv.lock` records the resolved dependency set.

## Requirements

- Windows 10 or later
- Python 3.11, 3.12, or 3.13 (Python 3.11 is the documented default)
- PowerShell
- Internet access for cloud LLMs and live financial sources
- Ollama only if using a local model

The automated tests do not require internet access, provider keys, paid data APIs, or a
running Ollama server.

## Windows setup

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Provider setup

### Ollama

Install [Ollama](https://ollama.com/), then start the service and pull a model:

```powershell
ollama serve
ollama pull llama3.1:8b
```

In a second PowerShell window, run the Streamlit application, select **Ollama**, confirm
`http://localhost:11434`, choose the pulled model, and test the connection.

### OpenAI

Select **OpenAI** and paste the API key into the password field. Alternatively, set it
for the current PowerShell process:

```powershell
$env:OPENAI_API_KEY = "your-key"
```

### Google Gemini

Select **Google Gemini** and paste the key, or set:

```powershell
$env:GOOGLE_API_KEY = "your-key"
```

### Anthropic

Select **Anthropic** and paste the key, or set:

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
```

UI keys take precedence over environment variables. The application never writes a UI
key to `.env` or another file.

## Optional financial-data configuration

Core analysis uses yfinance and does not require a separate financial-data key.

SEC EDGAR requires a descriptive User-Agent with contact information. Enter it under
**Advanced data sources** or set:

```powershell
$env:SEC_USER_AGENT = "FinancialAnalystAgent/1.0 you@example.com"
```

Actual earnings-call transcript retrieval is an optional enhancement through Financial
Modeling Prep:

```powershell
$env:FMP_API_KEY = "your-key"
```

Without that key, transcript requests return a labelled unavailable result. News is
never presented as a transcript.

Users who prefer a local environment file may copy `.env.example` to `.env` and fill
only the variables they intend to use. `.env` is ignored by Git. Do not commit it.

## Run the application

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Streamlit opens the local app in the browser. Page load does not initialize a model or
make a paid provider call. A provider is contacted only after **Test connection** or
**Run analysis** is selected.

## Use the application

1. Select an LLM provider.
2. Select a suggested model or enable the custom-model field.
3. Paste the cloud API key, or configure the Ollama base URL.
4. Select **Test connection**.
5. Enter a ticker and research question.
6. Optionally upload a PDF and adjust DCF assumptions.
7. Select **Run analysis**.
8. Review the report, source metadata, missing-data notices, and chart.
9. Download the Markdown report.

## Run tests and quality checks

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m compileall app.py financial_analyst tests
```

Tests use stored SEC fixtures and mocks. A test-level network guard fails the suite if a
test attempts a live socket connection.

## Security notes

- Password inputs live only in Streamlit session state.
- Resetting the session removes UI-entered credentials and the generated report.
- API keys are not written to disk, reports, prompts shown to the user, or logs.
- Displayed provider errors are redacted and length-bounded.
- Local logs are bounded and contain event types, not uploaded text or full prompts.
- PDFs are parsed from uploaded bytes; no LLM tool accepts a local filesystem path.
- Uploaded filenames are sanitized, size-limited, and assigned opaque internal IDs.
- Text inside an uploaded document is treated as untrusted evidence, not instructions.
- The application has no shell-execution tool.
- `.env`, caches, logs, reports, virtual environments, and generated databases are
  excluded by `.gitignore`.

If a credential was ever committed in an older project revision, revoke it at the
provider even after removing the file from the current tree. Git history still contains
the older revision.

## Data-source limitations

- yfinance is a convenient third-party interface and can be delayed, incomplete, or
  temporarily blocked. Its values are labelled as provider data.
- SEC Company Facts coverage varies by issuer and taxonomy. The client preserves form,
  filing date, fiscal context, unit, period, and accession metadata when present.
- SEC access is disabled without a valid User-Agent.
- DCF output is assumption-sensitive. Missing cash or debt suppresses equity value;
  missing diluted shares suppresses per-share value.
- News availability and publisher metadata vary.
- FMP transcript availability depends on the optional key, provider coverage, and plan.
- Text-only PDF extraction does not perform OCR on scanned documents.
- The workflow analyzes one primary ticker per run.

## Financial disclaimer

This software is for informational, educational, and portfolio-project use. It is not
financial, investment, legal, accounting, or tax advice. Source data may be delayed,
incomplete, or incorrect. Verify material facts in original filings and consult a
qualified professional before making financial decisions.

## Known limitations

- Reliable automatic peer discovery is not enabled. The application reports peer
  comparison as unavailable instead of substituting an ETF.
- PDF retrieval uses a compact lexical ranker rather than embeddings.
- Reports are downloaded as Markdown; PDF export is intentionally not included.
- Live provider and market behavior cannot be guaranteed by the offline tests.
- The application does not persist research sessions or uploaded documents.

## Future improvements

- Add opt-in OCR for scanned filings.
- Add an authoritative, licensed peer-classification source.
- Add explicit multi-ticker comparison with like-currency normalization.
- Add more filing concepts with issuer-specific taxonomy tests.
- Add optional user-approved local encrypted report history.
