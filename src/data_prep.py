"""Load and clean UCI Online Retail II transactions with an auditable log."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

def load_raw(path=None):
    """Load xlsx or csv and normalize source column names."""
    path = Path(path) if path else next(iter(RAW.glob("*.xlsx")), None)
    if path is None:
        path = next(iter(RAW.glob("*.csv")), None)
    if path is None:
        raise FileNotFoundError("Place online_retail_II.xlsx or a transaction CSV in data/raw/.")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)
    aliases = {"InvoiceNo":"Invoice", "UnitPrice":"Price", "CustomerID":"Customer ID"}
    return df.rename(columns=aliases)

def clean_transactions(df, output_dir=OUT):
    """Apply deliberate exclusions and save a row-count audit and clean file.

    Cancelled invoices and negative-quantity returns are excluded from purchase
    modeling; their counts remain visible in the cleaning audit.
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    audit = [{"step":"raw rows", "input_rows":len(df), "removed_rows":0, "rows":len(df)}]
    # Keep customer-linked negative-quantity transactions as a separate input
    # for the historical return-rate feature; exclude them from purchases.
    returns = df[df["Customer ID"].notna() & df["Quantity"].lt(0) & df["InvoiceDate"].notna()].copy()
    returns["Customer ID"] = returns["Customer ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    returns.to_csv(output_dir / "transactions_returns.csv", index=False)
    def keep(mask, name):
        nonlocal df
        before=len(df); df = df.loc[mask].copy()
        audit.append({"step":name,"input_rows":before,"removed_rows":before-len(df),"rows":len(df)})
    keep(df["Customer ID"].notna(), "remove missing Customer ID")
    keep(~df.duplicated(), "remove exact duplicate rows")
    keep(~df["Invoice"].astype(str).str.startswith("C", na=False), "remove cancelled invoices (C prefix)")
    keep(df["Quantity"].gt(0), "remove returns and non-positive quantities")
    keep(df["Price"].gt(0), "remove non-positive prices")
    keep(df["InvoiceDate"].notna(), "remove invalid invoice dates")
    df["Customer ID"] = df["Customer ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["Revenue"] = df["Quantity"] * df["Price"]
    df.to_csv(output_dir / "transactions_clean.csv", index=False)
    pd.DataFrame(audit).to_csv(output_dir / "cleaning_audit.csv", index=False)
    return df, pd.DataFrame(audit)

if __name__ == "__main__":
    clean_transactions(load_raw())
