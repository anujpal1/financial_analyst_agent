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
    search_internal_database
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
    search_internal_database
]

TOOL_MAPPING = {tool.name: tool for tool in FINANCIAL_TOOLS}