#CODE RUNNING ATTENTION
Before running the code, go to `Terminal`, paste to run this 
```
pip install -r requirements.txt
```
then run the code `CafeSale_code.py`

# DataCleaning
Cafe Sales data cleaning work
- Loads dirty_cafe_sales.csv from CafeSale.zip
- Cleans missing/ERROR/UNKNOWN values
- Converts numeric columns
- Reconstructs Total Spent when possible
- Optionally infers missing Quantity/Price from the other fields
- Parses dates
- Removes duplicate Transaction IDs
- Saves cleaned CSV in `cafe_sale_clean.csv`
- a quick data-quality report in `cleaning_report.csv`
