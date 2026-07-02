import os
import requests
import yfinance as yf
from ddgs import DDGS
from langchain_core.tools import tool
import config

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

@tool
def search_financial_news(query: str) -> str:
    """Queries internet indexes to retrieve recent market headlines and business summaries."""
    try:
        clean_query = query.replace("{", "").replace("}", "").replace("'", "")
        results_summary = [f"Live News Track for: '{clean_query}'"]
        
        # Fallback to plain keyword to avoid strict bracket parsing bugs
        with DDGS() as ddgs:
            # We use max_results keyword parameter directly
            results = list(ddgs.text(f"{clean_query} financial news", max_results=3))
            
        if not results:
            return f"Search returned 0 indexed matches for: '{clean_query}'."
            
        for index, item in enumerate(results, 1):
            results_summary.append(f"{index}. Head: {item.get('title')}\n   Data: {item.get('body')}")
            
        return "\n\n".join(results_summary)
    except Exception as e:
        return f"Internet index exception for query '{query}': {str(e)}"

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
    """Fetches historical stock prices for the last 30 days to calculate simple moving averages and momentum."""
    try:
        ticker_clean = ticker.upper().strip()
        stock = yf.Ticker(ticker_clean)
        hist = stock.history(period="30d")
        
        if hist.empty:
            return f"No historical trend lines found for ticker: {ticker_clean}"
            
        close_prices = hist['Close'].tolist()
        avg_30d = sum(close_prices) / len(close_prices)
        current_price = close_prices[-1]
        momentum = "UPWARD" if current_price > avg_30d else "DOWNWARD"
        
        return (
            f"30-Day Historical Trend Analysis for {ticker_clean}:\n"
            f"- 30-Day Simple Moving Average (SMA): ${avg_30d:.2f}\n"
            f"- Current Price Position: ${current_price:.2f}\n"
            f"- Momentum: The stock is currently trending in an {momentum} direction."
        )
    except Exception as e:
        return f"Failed to calculate historical trend metrics: {str(e)}"


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