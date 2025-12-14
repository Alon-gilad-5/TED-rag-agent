import pandas as pd
from pathlib import Path

# Get the path relative to this script's location
script_dir = Path(__file__).parent
csv_path = script_dir.parent / "data" / "ted_talks_en.csv"

df = pd.read_csv(csv_path)

print(f"Total talks: {len(df)}\n")
print(f"Columns: {list(df.columns)}\n")
print("=" * 60)

for col in df.columns:
    unique_count = df[col].nunique()
    print(f"\n{col}: {unique_count} unique values")

    # Show sample values for columns with reasonable unique counts
    if unique_count <= 20:
        print(f"  Values: {df[col].unique().tolist()}")
    else:
        print(f"  Sample: {df[col].dropna().head(5).tolist()}")