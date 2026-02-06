from __future__ import annotations
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
ZIP_PATH = Path("CafeSale.zip")          
CSV_NAME_IN_ZIP = "dirty_cafe_sales.csv" 
OUT_DIR = Path("output_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLEAN_CSV = OUT_DIR / "cafe_sales_clean.csv"
OUT_REPORT_CSV = OUT_DIR / "cleaning_report.csv"

# For Total Spent validation: allow small floating error
TOTAL_TOLERANCE = 0.01

# ----------------------------
# HELPERS
# ----------------------------
MISSING_MARKERS = {"", " ", "NA", "N/A", "NULL", "NONE", "nan", "NaN", "Unknown", "UNKNOWN", "ERROR"}

def normalize_text_cell(x):
    """Strip whitespace; keep NaN as NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return s

def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a series to numeric, coercing bad values like 'ERROR'/'UNKNOWN' to NaN."""
    s = s.map(normalize_text_cell)
    # Make common bad markers NaN
    s = s.replace(list(MISSING_MARKERS), np.nan)
    return pd.to_numeric(s, errors="coerce")

def clean_category(s: pd.Series, mapping: dict[str, str] | None = None, fill_value: str = "Unknown") -> pd.Series:
    """Clean a categorical series: strip, set markers to NaN, map values, fill."""
    s = s.map(normalize_text_cell)
    s = s.replace(list(MISSING_MARKERS), np.nan)

    if mapping:
        # Normalize case before mapping
        s_norm = s.astype("string").str.strip()
        # Map using a case-insensitive approach
        # (convert keys to lower once)
        mapping_lower = {k.lower(): v for k, v in mapping.items()}
        s = s_norm.str.lower().map(mapping_lower).fillna(s_norm)

    return s.fillna(fill_value)

def parse_date(s: pd.Series) -> pd.Series:
    """Parse transaction date safely."""
    s = s.map(normalize_text_cell)
    s = s.replace(list(MISSING_MARKERS), np.nan)
    return pd.to_datetime(s, errors="coerce")  # NaT for invalid

# ----------------------------
# LOAD
# ----------------------------
if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Could not find zip file: {ZIP_PATH.resolve()}")

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    if CSV_NAME_IN_ZIP not in z.namelist():
        raise FileNotFoundError(f"'{CSV_NAME_IN_ZIP}' not found in {ZIP_PATH.name}. Found: {z.namelist()}")
    with z.open(CSV_NAME_IN_ZIP) as f:
        df = pd.read_csv(f)

print("Loaded shape:", df.shape)
print(df.head(3))

# ----------------------------
# BASIC STANDARDIZATION
# ----------------------------
# Strip all string cells
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].map(normalize_text_cell)

# Ensure expected columns exist (adjust if dataset differs)
expected_cols = {
    "Transaction ID", "Item", "Quantity", "Price Per Unit", "Total Spent",
    "Payment Method", "Location", "Transaction Date"
}
missing_cols = expected_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}. Found columns: {list(df.columns)}")

# ----------------------------
# REMOVE DUPLICATES BY TRANSACTION ID
# ----------------------------
before_dupes = len(df)
df = df.drop_duplicates(subset=["Transaction ID"], keep="first")
print(f"Removed duplicates by Transaction ID: {before_dupes - len(df)}")

# ----------------------------
# CLEAN CATEGORICAL COLUMNS
# ----------------------------
df["Item"] = clean_category(df["Item"], fill_value="Unknown Item")

payment_map = {
    "cash": "Cash",
    "credit card": "Credit Card",
    "card": "Credit Card",
    "digital wallet": "Digital Wallet",
    "wallet": "Digital Wallet",
    "other": "Other",
    "unknown": "Other",
    "error": "Other",
}
df["Payment Method"] = clean_category(df["Payment Method"], mapping=payment_map, fill_value="Other")

location_map = {
    "in-store": "In-store",
    "instore": "In-store",
    "in store": "In-store",
    "takeaway": "Takeaway",
    "take-away": "Takeaway",
    "take away": "Takeaway",
    "unknown": "Unknown",
    "error": "Unknown",
}
df["Location"] = clean_category(df["Location"], mapping=location_map, fill_value="Unknown")

# ----------------------------
# CLEAN NUMERIC COLUMNS
# ----------------------------
df["Quantity"] = to_numeric_series(df["Quantity"])
df["Price Per Unit"] = to_numeric_series(df["Price Per Unit"])
df["Total Spent"] = to_numeric_series(df["Total Spent"])

# Quantity should be integer-ish and >= 1 (keep NaN for now)
# Round to nearest int where close to an int (optional, but nice)
qty = df["Quantity"]
df.loc[qty.notna(), "Quantity"] = np.where(np.isclose(qty, np.round(qty), atol=1e-9), np.round(qty), qty)
df["Quantity"] = df["Quantity"].astype("Float64")  # allows NA

# Remove non-sensical values
df.loc[df["Quantity"].notna() & (df["Quantity"] <= 0), "Quantity"] = np.nan
df.loc[df["Price Per Unit"].notna() & (df["Price Per Unit"] <= 0), "Price Per Unit"] = np.nan
df.loc[df["Total Spent"].notna() & (df["Total Spent"] <= 0), "Total Spent"] = np.nan

# ----------------------------
# RECONSTRUCT / IMPUTE MISSING VALUES (BUSINESS LOGIC)
# ----------------------------
# 1) If Total Spent is missing but Quantity and Price exist -> compute
mask_total_missing = df["Total Spent"].isna() & df["Quantity"].notna() & df["Price Per Unit"].notna()
df.loc[mask_total_missing, "Total Spent"] = df.loc[mask_total_missing, "Quantity"] * df.loc[mask_total_missing, "Price Per Unit"]

# 2) If Quantity is missing but Total and Price exist -> infer Quantity = Total / Price
mask_qty_missing = df["Quantity"].isna() & df["Total Spent"].notna() & df["Price Per Unit"].notna()
inferred_qty = df.loc[mask_qty_missing, "Total Spent"] / df.loc[mask_qty_missing, "Price Per Unit"]
# Keep only plausible quantities (>=1); round if close to integer
inferred_qty = np.where(np.isfinite(inferred_qty), inferred_qty, np.nan)
df.loc[mask_qty_missing, "Quantity"] = inferred_qty
# Re-apply integer rounding + invalid cleanup
qty = df["Quantity"]
df.loc[qty.notna(), "Quantity"] = np.where(np.isclose(qty, np.round(qty), atol=1e-9), np.round(qty), qty)
df.loc[df["Quantity"].notna() & (df["Quantity"] <= 0), "Quantity"] = np.nan

# 3) If Price is missing but Total and Quantity exist -> infer Price = Total / Quantity
mask_price_missing = df["Price Per Unit"].isna() & df["Total Spent"].notna() & df["Quantity"].notna()
inferred_price = df.loc[mask_price_missing, "Total Spent"] / df.loc[mask_price_missing, "Quantity"]
inferred_price = np.where(np.isfinite(inferred_price), inferred_price, np.nan)
df.loc[mask_price_missing, "Price Per Unit"] = inferred_price
df.loc[df["Price Per Unit"].notna() & (df["Price Per Unit"] <= 0), "Price Per Unit"] = np.nan

# After inference, recompute Total Spent again if still missing and now we have qty+price
mask_total_missing = df["Total Spent"].isna() & df["Quantity"].notna() & df["Price Per Unit"].notna()
df.loc[mask_total_missing, "Total Spent"] = df.loc[mask_total_missing, "Quantity"] * df.loc[mask_total_missing, "Price Per Unit"]

# ----------------------------
# VALIDATE TOTALS + FLAG ISSUES
# ----------------------------
df["Computed Total"] = np.where(
    df["Quantity"].notna() & df["Price Per Unit"].notna(),
    df["Quantity"] * df["Price Per Unit"],
    np.nan
)

df["Total Mismatch Flag"] = False
mismatch_mask = (
    df["Total Spent"].notna()
    & pd.notna(df["Computed Total"])
    & (np.abs(df["Total Spent"] - df["Computed Total"]) > TOTAL_TOLERANCE)
)
df.loc[mismatch_mask, "Total Mismatch Flag"] = True

# ----------------------------
# CLEAN DATE
# ----------------------------
df["Transaction Date"] = parse_date(df["Transaction Date"])
df["Year"] = df["Transaction Date"].dt.year
df["Month"] = df["Transaction Date"].dt.month
df["Day"] = df["Transaction Date"].dt.day

# ----------------------------
# FINAL TOUCHES
# ----------------------------
df["Quantity"] = df["Quantity"].round().astype("Int64")
# Reorder columns nicely
ordered_cols = [
    "Transaction ID", "Transaction Date", "Year", "Month", "Day",
    "Item", "Quantity", "Price Per Unit", "Total Spent",
    "Payment Method", "Location",
    "Computed Total", "Total Mismatch Flag"
]
df = df[[c for c in ordered_cols if c in df.columns]]

# ----------------------------
# REPORT + SAVE
# ----------------------------
report = pd.DataFrame({
    "rows": [len(df)],
    "missing_transaction_date": [df["Transaction Date"].isna().sum()],
    "missing_item": [(df["Item"] == "Unknown Item").sum()],
    "missing_quantity": [df["Quantity"].isna().sum()],
    "missing_price_per_unit": [df["Price Per Unit"].isna().sum()],
    "missing_total_spent": [df["Total Spent"].isna().sum()],
    "total_mismatch_flags": [df["Total Mismatch Flag"].sum()],
})
df.to_csv(OUT_CLEAN_CSV, index=False)
report.to_csv(OUT_REPORT_CSV, index=False)
print("\n Saved cleaned data to:", OUT_CLEAN_CSV.resolve())
print(" Saved cleaning report to:", OUT_REPORT_CSV.resolve())
print("\nCleaning report:")
print(report.to_string(index=False))
