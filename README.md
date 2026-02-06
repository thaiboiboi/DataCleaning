# DataCleaning
"""
Cafe Sales data cleaning (end-to-end)
- Loads dirty_cafe_sales.csv from CafeSale.zip
- Cleans missing/ERROR/UNKNOWN values
- Converts numeric columns
- Reconstructs Total Spent when possible
- Optionally infers missing Quantity/Price from the other fields
- Parses dates
- Removes duplicate Transaction IDs
- Saves cleaned CSV + a quick data-quality report
"""
