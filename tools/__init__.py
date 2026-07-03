from tools.financial_tools import process_local_pdf_document
from tools.financial_tools import (
    get_stock_price, 
    search_sec_filings, 
    search_financial_news, 
    calculate_financial_ratios,
    analyze_historical_trends,
    get_income_statement_metrics,
    get_cash_flow_metrics,
    find_industry_competitors,
    get_dividend_and_yield_history,
    calculate_dcf_valuation_estimate,
    search_internal_database,
    tavily_web_search,
    search_sec_filings,
    process_local_pdf_document,
    get_earnings_transcript,
    get_advanced_institutional_metrics,
    get_balance_sheet_metrics
)

FINANCIAL_TOOLS = [
    get_stock_price,
    search_sec_filings,
    search_financial_news,
    calculate_financial_ratios,
    analyze_historical_trends,
    get_income_statement_metrics,
    get_cash_flow_metrics,
    find_industry_competitors,
    get_dividend_and_yield_history,
    calculate_dcf_valuation_estimate,
    search_internal_database,
    tavily_web_search,
    search_sec_filings,
    process_local_pdf_document,
    get_earnings_transcript,
    get_advanced_institutional_metrics,
    get_balance_sheet_metrics
]

TOOL_MAPPING = {tool.name: tool for tool in FINANCIAL_TOOLS}