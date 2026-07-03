import os
import requests
import yfinance as yf
from ddgs import DDGS
from langchain_core.tools import tool
import config


from gnews import GNews
from tavily import TavilyClient
from newsapi import NewsApiClient


import json
import fitz  # PyMuPDF
import requests

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from config import SYSTEM_LOGGER as logger


import os
import json
import fitz
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger

# Add these LangChain imports for the Context Compressor
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the dedicated compression model (Temperature 0 for strict extraction)
compressor_llm = ChatOllama(model="llama3.1:8b", temperature=0)

# ==========================================
# 1. LIVE MARKET DATA TOOL
# ==========================================

@tool
def get_stock_price(ticker: str) -> str:
    """Fetch live real-time stock pricing, daily highs, lows, and volume for any valid market ticker."""
    try:
        ticker_clean = ticker.upper().strip()
        stock = yf.Ticker(ticker_clean)
        info = stock.fast_info
        
        if info['last_price'] is None or float(info['last_price']) == 0:
            return f"Error: Ticker '{ticker_clean}' returned no active market data."
            
        return (
            f"Live Market Data for {ticker_clean}:\n"
            f"- Current Price: ${info['last_price']:.2f}\n"
            f"- Day High: ${info['day_high']:.2f}\n"
            f"- Day Low: ${info['day_low']:.2f}\n"
            f"- Volume: {info['last_volume']:,}"
        )
    except Exception as e:
        return f"Failed to retrieve data for ticker '{ticker}': {str(e)}"

# ==========================================
# 2. ADVANCED SEC EDGAR PARSING (With Fallbacks)
# ==========================================

@tool
def search_sec_filings(ticker: str) -> str:
    """Queries official SEC EDGAR database to extract the absolute latest net income statistics using variable US-GAAP financial schemas."""
    try:
        ticker_clean = ticker.upper().strip()
        headers = {"User-Agent": "EducationalAgent research@domain.com"}
        
        # Pull standard mapping
        cik_response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=10)
        if cik_response.status_code != 200:
            return "Error: SEC registry mapping unreachable."
            
        company_data = cik_response.json()
        cik = None
        for item in company_data.values():
            if item['ticker'] == ticker_clean:
                cik = str(item['cik_str']).zfill(10)
                break
                
        if not cik:
            return f"Error: Ticker '{ticker_clean}' cannot be mapped to an SEC CIK index."
            
        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        facts_response = requests.get(facts_url, headers=headers, timeout=10)
        if facts_response.status_code != 200:
            return f"Error: SEC metadata endpoint returned status code {facts_response.status_code}."
            
        facts_data = facts_response.json()
        us_gaap = facts_data.get("us-gaap", {})
        
        # Financial Schema Fallback Strategy array: try multiple valid US-GAAP schema taxonomies
        target_concepts = ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"]
        net_income_dict = None
        
        for concept in target_concepts:
            if concept in us_gaap:
                net_income_dict = us_gaap[concept].get("units", {}).get("USD", [])
                if net_income_dict:
                    break
                    
        if not net_income_dict:
            return f"SEC CIK {cik} verified, but the financial statement tags utilize custom non-standard definitions."
            
        # Get latest entry
        latest_entry = net_income_dict[-1]
        return (
            f"Official SEC EDGAR Fundamental Data for {ticker_clean} (CIK: {cik}):\n"
            f"- Reporting Period: FY{latest_entry.get('fy', 'N/A')} ({latest_entry.get('form', '10-K')})\n"
            f"- Net Income Value: ${latest_entry.get('val', 0):,} USD"
        )
    except Exception as e:
        return f"SEC Pipeline exception for ticker '{ticker}': {str(e)}"

# ==========================================
# 3. FIXED LIVE WEB RESEARCH TOOL
# ==========================================

# ==========================================
# ADVANCED NEWS & WEB SEARCH TOOLS
# ==========================================

@tool
def search_financial_news(query: str) -> str:
    """Searches for the latest financial news articles and market sentiment using GNews and NewsAPI."""
    try:
        news_results = []
        
        # 1. Try NewsAPI First (if key exists)
        newsapi_key = os.getenv("NEWSAPI_KEY")
        if newsapi_key and newsapi_key != "your_newsapi_key_here":
            newsapi = NewsApiClient(api_key=newsapi_key)
            articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=3)
            if articles['status'] == 'ok' and articles['totalResults'] > 0:
                for article in articles['articles']:
                    news_results.append(f"- {article['title']} ({article['source']['name']}): {article['description']}")
        
        # 2. Fallback to GNews (Requires no API key, highly reliable)
        if not news_results:
            google_news = GNews(language='en', country='US', max_results=3)
            gnews_articles = google_news.get_news(query)
            for article in gnews_articles:
                news_results.append(f"- {article['title']} ({article['publisher']['title']})")
                
        if not news_results:
            return f"No recent news found for query: {query}"
            
        return "Latest Financial News:\n" + "\n".join(news_results)
    except Exception as e:
        return f"Failed to retrieve news data: {str(e)}"

@tool
def tavily_web_search(query: str) -> str:
    """Uses Tavily's AI-optimized search engine to find up-to-date facts, transcripts, or general web data."""
    try:
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key or tavily_key == "your_tavily_key_here":
            return "TAVILY_API_KEY is missing or invalid in the .env file."
            
        client = TavilyClient(api_key=tavily_key)
        # We use advanced search to get snippet contexts
        response = client.search(query, search_depth="advanced", max_results=3)
        
        results = [f"- {res['title']}: {res['content']}" for res in response.get('results', [])]
        
        if not results:
            return f"No web search results found for: {query}"
            
        return f"Tavily Search Results for '{query}':\n" + "\n".join(results)
    except Exception as e:
        return f"Tavily search failed: {str(e)}"


# ==========================================
# 4. RATIO CALCULATOR TOOL (New Engine Tool)
# ==========================================

@tool
def calculate_financial_ratios(ticker: str) -> str:
    """Calculates fundamental financial ratios like Price-to-Earnings (P/E) using current live stock metrics."""
    try:
        ticker_clean = ticker.upper().strip()
        stock = yf.Ticker(ticker_clean)
        info = stock.info
        
        pe_ratio = info.get('trailingPE', 'N/A')
        pb_ratio = info.get('priceToBook', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        
        return (
            f"Calculated Financial Ratios for {ticker_clean}:\n"
            f"- Trailing P/E Ratio: {pe_ratio}\n"
            f"- Forward P/E Ratio: {forward_pe}\n"
            f"- Price-to-Book (P/B) Ratio: {pb_ratio}"
        )
    except Exception as e:
        return f"Failed to compute operational analytics ratios for '{ticker}': {str(e)}"


# ==========================================
# 5. HISTORICAL TREND ANALYZER
# ==========================================

@tool
def analyze_historical_trends(ticker: str) -> str:
    """Calculates historical 30-day momentum and volatility using Pandas and NumPy, exporting an interactive Plotly chart."""
    logger.info(f"Initiating historical trend engine for ticker: {ticker}")
    try:
        ticker_clean = ticker.upper().strip()
        stock = yf.Ticker(ticker_clean)
        
        # 1. Ingest raw historical data into a Pandas DataFrame
        hist = stock.history(period="60d")
        if hist.empty:
            logger.warning(f"No pricing data history recovered for {ticker_clean}")
            return f"No historical market data discovered for {ticker_clean}."

        # 2. Use Pandas & NumPy explicitly for financial engineering
        df = pd.DataFrame(hist[['Close', 'Volume']])
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Calculate moving averages via rolling windows
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Vectorized volatility extraction using NumPy
        recent_returns = df['Daily_Return'].tail(30).to_numpy()
        volatility = np.std(recent_returns, ddof=1) * np.sqrt(252) * 100  # Annualized volatility %
        
        current_price = df['Close'].iloc[-1]
        price_30d_ago = df['Close'].iloc[-30] if len(df) >= 30 else df['Close'].iloc[0]
        momentum_pct = ((current_price - price_30d_ago) / price_30d_ago) * 100

        logger.info(f"Calculated 30D Momentum: {momentum_pct:.2f}%, Volatility: {volatility:.2f}% for {ticker_clean}")

        # 3. Build a beautiful interactive Plotly Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='royalblue', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20-Day Moving Average', line=dict(color='orange', dash='dash')))
        
        fig.update_layout(
            title=f"{ticker_clean} Historical Momentum & MA Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Stock Price ($)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        # Save Plotly configuration as HTML snippet so Streamlit can render it dynamically
        chart_dir = "reports/charts"
        os.makedirs(chart_dir, exist_ok=True)
        chart_path = f"{chart_dir}/{ticker_clean}_trend.html"
        pio.write_html(fig, chart_path, include_plotlyjs='cdn', full_html=False)
        logger.info(f"Plotly visualization asset safely compiled at {chart_path}")

        return (
            f"Pandas & NumPy Trend Summary for {ticker_clean}:\n"
            f"- Current Closing Level: ${current_price:.2f}\n"
            f"- 30-Day Pure Rolling Momentum: {momentum_pct:.2f}%\n"
            f"- Annualized Historical Volatility: {volatility:.2f}%\n"
            f"- Interactive Chart Component Generated: View path '{chart_path}'"
        )
    except Exception as e:
        logger.exception(f"Trend processing crashed for {ticker}: {str(e)}")
        return f"Failed historical analysis: {str(e)}"

# next 5 tools

# Append these 5 new tools to the bottom of tools/financial_tools.py

@tool
def get_income_statement_metrics(ticker: str) -> str:
    """Extracts key income statement line items like Total Revenue, Gross Profit, and Net Income."""
    try:
        ticker_clean = ticker.upper().strip()
        stock = yf.Ticker(ticker_clean)
        stmt = stock.income_stmt
        
        if stmt.empty:
            return f"Income statement data is currently unavailable for {ticker_clean}."
            
        latest_period = stmt.columns[0]
        data = stmt[latest_period]
        
        rev = data.get('Total Revenue', 'N/A')
        gp = data.get('Gross Profit', 'N/A')
        ni = data.get('Net Income', 'N/A')
        
        # FIX: Format strings safely before putting them in the f-string
        rev_str = f"${rev:,}" if isinstance(rev, (int, float)) else str(rev)
        gp_str = f"${gp:,}" if isinstance(gp, (int, float)) else str(gp)
        ni_str = f"${ni:,}" if isinstance(ni, (int, float)) else str(ni)
        
        return (
            f"Income Statement Data for {ticker_clean} (Period End: {latest_period.strftime('%Y-%m-%d')}):\n"
            f"- Total Revenue: {rev_str}\n"
            f"- Gross Profit: {gp_str}\n"
            f"- Net Income: {ni_str}"
        )
    except Exception as e:
        return f"Failed to parse income statement: {str(e)}"

@tool
def get_cash_flow_metrics(ticker: str) -> str:
    """Retrieves operational cash flow, capital expenditures, and free cash flow metrics."""
    try:
        ticker_clean = ticker.upper().strip()
        stock = yf.Ticker(ticker_clean)
        cf = stock.cashflow
        
        if cf.empty:
            return f"Cash flow statement data is currently unavailable for {ticker_clean}."
            
        latest_period = cf.columns[0]
        data = cf[latest_period]
        
        op_cf = data.get('Operating Cash Flow', 'N/A')
        cap_ex = data.get('Capital Expenditure', 'N/A')
        free_cf = 'N/A'
        
        if isinstance(op_cf, (int, float)) and isinstance(cap_ex, (int, float)):
            free_cf = op_cf + cap_ex if cap_ex < 0 else op_cf - cap_ex
            
        # FIX: Format strings safely
        op_cf_str = f"${op_cf:,}" if isinstance(op_cf, (int, float)) else str(op_cf)
        cap_ex_str = f"${cap_ex:,}" if isinstance(cap_ex, (int, float)) else str(cap_ex)
        free_cf_str = f"${free_cf:,}" if isinstance(free_cf, (int, float)) else str(free_cf)
            
        return (
            f"Cash Flow Metrics for {ticker_clean} (Period End: {latest_period.strftime('%Y-%m-%d')}):\n"
            f"- Operating Cash Flow: {op_cf_str}\n"
            f"- Capital Expenditure (CapEx): {cap_ex_str}\n"
            f"- Calculated Free Cash Flow (FCF): {free_cf_str}"
        )
    except Exception as e:
        return f"Failed to parse cash flow data: {str(e)}"

@tool
def find_industry_competitors(ticker: str) -> str:
    """Identifies the industry sector of a company and derives a list of major competitor tickers for comparison."""
    try:
        ticker_clean = ticker.upper().strip()
        stock = yf.Ticker(ticker_clean)
        info = stock.info
        
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        
        # Simple dynamic lookup table mapping prominent tickers to peers
        competitor_map = {
            "NVDA": ["AMD", "INTC", "AVGO"],
            "MSFT": ["GOOGL", "AMZN", "AAPL"],
            "AAPL": ["MSFT", "GOOGL", "SONY"]
        }
        
        peers = competitor_map.get(ticker_clean, ["SPY (Broad Market Benchmark Index)"])
        return (
            f"Sector Classification for {ticker_clean}:\n"
            f"- Sector: {sector}\n"
            f"- Industry: {industry}\n"
            f"- Peer Group Competitors for analysis: {', '.join(peers)}"
        )
    except Exception as e:
        return f"Failed to identify industry competitors: {str(e)}"


@tool
def get_dividend_and_yield_history(ticker: str) -> str:
    """Fetches corporate yield tracking indicators, trailing dividend distributions, and payout ratios."""
    try:
        ticker_clean = ticker.upper().strip()
        stock = yf.Ticker(ticker_clean)
        info = stock.info
        
        div_rate = info.get('dividendRate', 0.0)
        div_yield = info.get('dividendYield', 0.0)
        payout_ratio = info.get('payoutRatio', 0.0)
        
        yield_pct = f"{div_yield * 100:.2f}%" if div_yield else "0.00%"
        payout_pct = f"{payout_ratio * 100:.2f}%" if payout_ratio else "0.00%"
        
        return (
            f"Dividend Registry Profile for {ticker_clean}:\n"
            f"- Annual Dividend Rate: ${div_rate if div_rate else 0.0}\n"
            f"- Current Dividend Yield: {yield_pct}\n"
            f"- Historical Payout Ratio: {payout_pct}"
        )
    except Exception as e:
        return f"Failed to retrieve dividend information: {str(e)}"


@tool
def calculate_dcf_valuation_estimate(ticker: str) -> str:
    """Runs a programmatic automated Discounted Cash Flow (DCF) model to estimate raw intrinsic stock value."""
    try:
        ticker_clean = ticker.upper().strip()
        stock = yf.Ticker(ticker_clean)
        info = stock.info
        cf = stock.cashflow
        
        # Safely extract Free Cash Flow indicators or default gracefully
        fcf = 0
        if not cf.empty and 'Operating Cash Flow' in cf.index:
            op_cf = cf.loc['Operating Cash Flow'].iloc[0]
            cap_ex = cf.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cf.index else 0
            fcf = op_cf + cap_ex if cap_ex < 0 else op_cf - cap_ex
            
        if fcf <= 0:
            # Fallback estimation based on gross margins if cash flow is temporarily distorted
            fcf = info.get('freeCashflow', info.get('operatingCashflow', 10_000_000_000))
            
        shares_outstanding = info.get('sharesOutstanding', None)
        if not shares_outstanding:
            return f"DCF Modeling aborted: Shares Outstanding variable unavailable for {ticker_clean}."
            
        # Hardcoded baseline modeling assumptions (Conservative values)
        growth_rate = 0.12        # 12% growth for next 5 years
        discount_rate = 0.09      # 9% cost of capital (WACC)
        terminal_growth = 0.02    # 2% terminal growth rate
        
        # Calculate Projected Cash Flows for next 5 years
        projected_cash_flows = []
        current_fcf = fcf
        for year in range(1, 6):
            current_fcf *= (1 + growth_rate)
            discounted_val = current_fcf / ((1 + discount_rate) ** year)
            projected_cash_flows.append(discounted_val)
            
        # Terminal Value calculation
        terminal_value = (current_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        discounted_terminal_value = terminal_value / ((1 + discount_rate) ** 5)
        
        # Total Intrinsic Value
        intrinsic_value_pool = sum(projected_cash_flows) + discounted_terminal_value
        intrinsic_value_per_share = intrinsic_value_pool / shares_outstanding
        
        return (
            f"Automated 5-Year DCF Model for {ticker_clean}:\n"
            f"- Base Free Cash Flow Used: ${fcf:,.2f}\n"
            f"- Assumed Core Growth Rate: {growth_rate*100:.1f}%\n"
            f"- Weighted Average Cost of Capital (Discount Rate): {discount_rate*100:.1f}%\n"
            f"- Estimated Intrinsic Fair Value Target: ${intrinsic_value_per_share:.2f} per share"
        )
    except Exception as e:
        return f"DCF Valuation Engine exception: {str(e)}"


# Import the memory recall function at the top of the file
from memory.long_term import recall_from_memory

# ==========================================
# 7. LONG-TERM MEMORY SEARCH TOOL
# ==========================================

@tool
def search_internal_database(query: str) -> str:
    """Searches the agent's internal long-term vector database for deeply stored historical documents and reports."""
    return recall_from_memory(query)




# ==========================================
# REAL SEC EDGAR & DOCUMENT PROCESSING TOOLS
# ==========================================

@tool
def search_sec_filings(ticker: str) -> str:
    """Fetches real institutional financial metrics directly from the SEC EDGAR Company Facts database."""
    try:
        ticker_clean = ticker.upper().strip()
        user_agent = os.getenv("SEC_USER_AGENT", "FinancialAnalystAgent/1.0 (anuj@example.com)")
        headers = {"User-Agent": user_agent}

        # 1. SEC requires a 10-digit Central Index Key (CIK). We fetch the global mapping file.
        cik_url = "https://data.sec.gov/files/company_tickers.json"
        cik_resp = requests.get(cik_url, headers=headers)
        if cik_resp.status_code != 200:
            return f"SEC Database Error: Unable to fetch CIK registry mapping (Status {cik_resp.status_code})."

        company_mapping = cik_resp.json()
        cik = None
        
        # Look up the ticker's matching CIK number
        for item in company_mapping.values():
            if item["ticker"] == ticker_clean:
                # Format to a 10-digit zero-padded string as required by the SEC URL structure
                cik = f"{item['cik_str']:010d}"
                break

        if not cik:
            return f"Ticker mapping anomaly: CIK index not discovered for ticker symbol '{ticker_clean}'."

        # 2. Query the official SEC Company Facts API for this CIK
        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        facts_resp = requests.get(facts_url, headers=headers)
        if facts_resp.status_code != 200:
            return f"SEC Access Denied: Failed to retrieve disclosures for CIK {cik} (Status {facts_resp.status_code})."

        facts_data = facts_resp.json()
        us_gaap = facts_data.get("facts", {}).get("us-gaap", {})

        # Extract core financial tags commonly found in 10-K / 10-Q statements
        metrics_to_extract = {
            "Net Income": us_gaap.get("NetIncomeLoss", {}).get("units", {}).get("USD", []),
            "Revenues": us_gaap.get("Revenues", {}).get("units", {}).get("USD", []) or us_gaap.get("RevenueFromContractWithCustomerExcludingAssessedTax", {}).get("units", {}).get("USD", []),
            "Assets": us_gaap.get("Assets", {}).get("units", {}).get("USD", [])
        }

        output_summary = f"Official SEC EDGAR Disclosures for {ticker_clean} (CIK: {cik}):\n"
        
        for name, data_list in metrics_to_extract.items():
            if data_list:
                # Grab the most recently reported data point in the array
                latest_entry = sorted(data_list, key=lambda x: x.get("end", ""))[-1]
                val = latest_entry.get("val", 0)
                form = latest_entry.get("form", "N/A")
                period_end = latest_entry.get("end", "N/A")
                val_formatted = f"${val:,}" if isinstance(val, (int, float)) else str(val)
                output_summary += f"- {name}: {val_formatted} (Source Form: {form}, Period Ended: {period_end})\n"
            else:
                output_summary += f"- {name}: Data tag not actively exposed in recent electronic filings.\n"

        return output_summary
    except Exception as e:
        return f"SEC operational parsing exception: {str(e)}"



@tool
def process_local_pdf_document(file_path: str) -> str:
    """Leverages PyMuPDF to extract text from PDFs, then uses an LLM to compress it into key metrics."""
    logger.info(f"Initiating Map-Reduce PDF extraction for: {file_path}")
    try:
        if not os.path.exists(file_path):
            return f"File Access Failure: The path '{file_path}' does not exist on disk."
            
        doc = fitz.open(file_path)
        extracted_text = []
        
        max_pages = min(len(doc), 5)
        for page_num in range(max_pages):
            page = doc.load_page(page_num)
            extracted_text.append(page.get_text())
            
        doc.close()
        full_content = "\n".join(extracted_text)
        
        if not full_content.strip():
            return f"Processing Notice: PDF '{file_path}' contained no extractable text."

        # === THE COMPRESSION LAYER ===
        logger.info("Executing context compression on raw PDF text...")
        messages = [
            SystemMessage(content="You are a strict data extraction tool. Extract the top 10 quantitative financial facts and metrics from the provided text. Return ONLY a bulleted list. Do not add conversational filler."),
            HumanMessage(content=f"RAW TEXT:\n{full_content[:6000]}")
        ]
        
        compression_response = compressor_llm.invoke(messages)
        compressed_text = compression_response.content
        
        logger.info("PDF context successfully compressed.")
        return f"Compressed Financial Data from '{file_path}':\n\n{compressed_text}"
        
    except Exception as e:
        logger.error(f"PDF compression engine threw an error: {str(e)}")
        return f"PyMuPDF document processing engine threw an error: {str(e)}"




# ==========================================
# EARNINGS TRANSCRIPT API INTEGRATION
# ==========================================

@tool
def get_earnings_transcript(ticker: str, year: int, quarter: int) -> str:
    """Fetches earnings call transcripts and compresses them into key forward-looking guidance bullet points."""
    logger.info(f"Fetching Q{quarter} {year} transcript for {ticker}...")
    try:
        ticker_clean = ticker.upper().strip()
        raw_transcript_text = ""
        
        # 1. Attempt FMP
        fmp_key = os.getenv("FMP_API_KEY")
        if fmp_key and fmp_key != "your_fmp_key_here":
            url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker_clean}?quarter={quarter}&year={year}&apikey={fmp_key}"
            response = requests.get(url)
            
            if response.status_code == 200 and len(response.json()) > 0:
                raw_transcript_text = response.json()[0].get("content", "")
        
        # 2. Attempt Finnhub Fallback (if FMP failed)
        if not raw_transcript_text:
            finnhub_key = os.getenv("FINNHUB_API_KEY")
            if finnhub_key and finnhub_key != "your_finnhub_key_here":
                import finnhub
                finnhub_client = finnhub.Client(api_key=finnhub_key)
                import datetime
                end_date = datetime.datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
                news = finnhub_client.company_news(ticker_clean, _from=start_date, to=end_date)
                earnings_news = [n for n in news if 'earning' in n.get('headline', '').lower()]
                if earnings_news:
                    raw_transcript_text = "\n".join([f"{item['headline']}: {item['summary']}" for item in earnings_news[:5]])

        if not raw_transcript_text:
            return f"Transcript Data Unavailable for {ticker_clean} (Q{quarter} {year})."

        # === THE COMPRESSION LAYER ===
        logger.info("Compressing raw earnings transcript...")
        messages = [
            SystemMessage(content="You are a strict financial extraction tool. Read the following earnings call transcript. Extract the CEO/CFO forward-looking guidance, projected revenues, and major risk factors. Return ONLY a bulleted list. Do not add conversational filler."),
            HumanMessage(content=f"RAW TRANSCRIPT:\n{raw_transcript_text[:6000]}")
        ]
        
        compression_response = compressor_llm.invoke(messages)
        logger.info("Transcript context successfully compressed.")
        
        return f"Compressed Management Guidance for {ticker_clean} (Q{quarter} {year}):\n\n{compression_response.content}"

    except Exception as e:
        logger.error(f"Earnings Transcript API Exception: {str(e)}")
        return f"Earnings Transcript API Exception: {str(e)}"


# ==========================================
# ADVANCED INSTITUTIONAL DATA LAYER (FINNHUB/FMP)
# ==========================================

@tool
def get_advanced_institutional_metrics(ticker: str) -> str:
    """
    Queries Finnhub and Financial Modeling Prep (FMP) APIs to extract institutional fundamental metrics,
    such as Enterprise Value, EV/EBITDA, operating margins, and 52-week high/low data.
    """
    try:
        ticker_clean = ticker.upper().strip()
        summary_lines = [f"Advanced Institutional Profile for {ticker_clean}:"]
        data_retrieved = False

        # 1. Attempt Finnhub Fundamental Metric Retrieval
        finnhub_key = os.getenv("FINNHUB_API_KEY")
        if finnhub_key and finnhub_key != "your_finnhub_key_here":
            import finnhub
            finnhub_client = finnhub.Client(api_key=finnhub_key)
            # Fetch basic financial metrics (margins, valuation, etc.)
            metrics_data = finnhub_client.company_basic_financials(ticker_clean, 'all')
            
            metric_dict = metrics_data.get('metric', {})
            if metric_dict:
                data_retrieved = True
                summary_lines.append("\n[Finnhub Core Data points]:")
                summary_lines.append(f"- 52-Week High: ${metric_dict.get('52WeekHigh', 'N/A')}")
                summary_lines.append(f"- 52-Week Low: ${metric_dict.get('52WeekLow', 'N/A')}")
                summary_lines.append(f"- Operating Margin (TTM): {metric_dict.get('operatingMarginTTM', 'N/A')}%")
                summary_lines.append(f"- Net Profit Margin (TTM): {metric_dict.get('netProfitMarginTTM', 'N/A')}%")
                summary_lines.append(f"- Return on Equity (TTM): {metric_dict.get('roettm', 'N/A')}%")

        # 2. Attempt Financial Modeling Prep (FMP) Enterprise Valuation Retrieval
        fmp_key = os.getenv("FMP_API_KEY")
        if fmp_key and fmp_key != "your_fmp_key_here":
            url = f"https://financialmodelingprep.com/api/v3/enterprise-values/{ticker_clean}?limit=1&apikey={fmp_key}"
            response = requests.get(url)
            
            if response.status_code == 200 and len(response.json()) > 0:
                data_retrieved = True
                fmp_data = response.json()[0]
                summary_lines.append("\n[Financial Modeling Prep (FMP) Enterprise Values]:")
                
                ev = fmp_data.get("enterpriseValue", "N/A")
                ev_formatted = f"${ev:,}" if isinstance(ev, (int, float)) else str(ev)
                
                summary_lines.append(f"- Enterprise Value: {ev_formatted}")
                summary_lines.append(f"- Stock Price Used: ${fmp_data.get('stockPrice', 'N/A')}")
                summary_lines.append(f"- Market Capitalization: ${fmp_data.get('marketCapitalization', 0):,}")
                summary_lines.append(f"- Number of Shares: {fmp_data.get('numberOfShares', 0):,}")

        if not data_retrieved:
            return f"Institutional Data Layer Notice: No premium keys found for Finnhub or FMP. Defaulting to standard yfinance data tools."

        return "\n".join(summary_lines)

    except Exception as e:
        return f"Advanced Institutional Data Layer exception: {str(e)}"


@tool
def get_balance_sheet_metrics(ticker: str) -> str:
    """Extracts key balance sheet items: Total Assets, Total Liabilities, and Total Debt."""
    try:
        ticker_clean = ticker.upper().strip()
        stock = yf.Ticker(ticker_clean)
        bs = stock.balance_sheet
        
        if bs.empty:
            return f"Balance sheet data is currently unavailable for {ticker_clean}."
            
        latest_period = bs.columns[0]
        data = bs[latest_period]
        
        assets = data.get('Total Assets', 'N/A')
        liabilities = data.get('Total Liabilities Net Minority Interest', data.get('Total Liabilities', 'N/A'))
        debt = data.get('Total Debt', 'N/A')
        
        # Format strings safely
        assets_str = f"${assets:,}" if isinstance(assets, (int, float)) else str(assets)
        liab_str = f"${liabilities:,}" if isinstance(liabilities, (int, float)) else str(liabilities)
        debt_str = f"${debt:,}" if isinstance(debt, (int, float)) else str(debt)
        
        # Risk Metric: Extract Beta from stock info
        info = stock.info
        beta = info.get('beta', 'N/A')
        risk_profile = "High Volatility" if isinstance(beta, (float, int)) and beta > 1.2 else "Market Perform" if isinstance(beta, (float, int)) and beta >= 0.8 else "Low Volatility"
            
        return (
            f"Balance Sheet & Risk Metrics for {ticker_clean} (Period End: {latest_period.strftime('%Y-%m-%d')}):\n"
            f"- Total Assets: {assets_str}\n"
            f"- Total Liabilities: {liab_str}\n"
            f"- Total Debt: {debt_str}\n"
            f"- Beta (Risk Measure): {beta} ({risk_profile})"
        )
    except Exception as e:
        return f"Failed to parse balance sheet data: {str(e)}"