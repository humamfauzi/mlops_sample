import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv("dataset/cfs_2017.csv", dtype={"SCTG": str, "MODE": str})
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# Rename for convenience
val_col = "SHIPMT_VALUE"
mode_col = "MODE"
sctg_col = "SCTG"
dist_col = "SHIPMT_DIST_ROUTED"

# ─────────────────────────────────────────────
# 1. SHIPMT_VALUE distribution
# ─────────────────────────────────────────────
print("=" * 60)
print("1. SHIPMT_VALUE distribution")
print("=" * 60)
v = df[val_col].dropna()
pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]
pct_vals = np.percentile(v, pcts)
print(f"  Count   : {len(v):,}")
print(f"  Mean    : {v.mean():,.2f}")
print(f"  Median  : {v.median():,.2f}")
print(f"  Std     : {v.std():,.2f}")
print(f"  Min     : {v.min():,.2f}")
print(f"  Max     : {v.max():,.2f}")
print()
print("  Percentiles:")
for p, val in zip(pcts, pct_vals):
    print(f"    p{str(p).rjust(5)} : {val:>15,.2f}")

# Log distribution buckets
print()
print("  Value bucket counts (raw $):")
buckets = [
    0,
    10,
    100,
    500,
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    np.inf,
]
labels = [
    "<10",
    "10-100",
    "100-500",
    "500-1k",
    "1k-5k",
    "5k-10k",
    "10k-50k",
    "50k-100k",
    "100k-500k",
    "500k-1M",
    ">1M",
]
cuts = pd.cut(v, bins=buckets, labels=labels, right=False)
bkt_counts = cuts.value_counts().sort_index()
for lbl, cnt in bkt_counts.items():
    pct = 100 * cnt / len(v)
    print(f"    {lbl:>12s} : {cnt:>10,}  ({pct:5.1f}%)")

# ─────────────────────────────────────────────
# 2. MODE distribution
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("2. MODE distribution")
print("=" * 60)

# Mode code descriptions (CFS codebook)
mode_labels = {
    "01": "Truck",
    "02": "For-hire Truck",
    "03": "Private Truck",
    "04": "Air",
    "05": "Air (incl courier)",
    "06": "Rail",
    "07": "Water",
    "08": "Pipeline",
    "09": "Other/Unknown",
    "11": "Truck",
    "12": "Truck & Rail",
    "13": "Truck & Water",
    "14": "Other intermodal",
    "15": "Unknown",
    "19": "Other/Multiple",
    "20": "Parcel/USPS/Courier",
}

mode_grp = df.groupby(mode_col)[val_col]
mode_stats = mode_grp.agg(
    count="count",
    mean_value="mean",
    median_value="median",
    total_value="sum",
).sort_values("count", ascending=False)

print(
    f"  {'Mode':>6}  {'Label':<22}  {'Count':>10}  {'%Rows':>6}  {'Mean $':>12}  {'Median $':>10}  {'Total $':>16}"
)
print(
    f"  {'-' * 6}  {'-' * 22}  {'-' * 10}  {'-' * 6}  {'-' * 12}  {'-' * 10}  {'-' * 16}"
)
for mode, row in mode_stats.iterrows():
    lbl = mode_labels.get(str(mode).zfill(2), "?")
    pct = 100 * row["count"] / len(df)
    print(
        f"  {str(mode).zfill(2):>6}  {lbl:<22}  {row['count']:>10,}  {pct:>5.1f}%  "
        f"{row['mean_value']:>12,.0f}  {row['median_value']:>10,.0f}  {row['total_value']:>16,.0f}"
    )

# ─────────────────────────────────────────────
# 3. SCTG distribution
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("3. SCTG distribution  (top 20 by count, then by mean value)")
print("=" * 60)

# SCTG 2-digit group labels (selected)
sctg_labels = {
    "01": "Live animals/fish",
    "02": "Cereal grains",
    "03": "Other ag products",
    "04": "Animal feed",
    "05": "Meat/seafood",
    "06": "Milled grain products",
    "07": "Other foodstuffs",
    "08": "Alcoholic beverages",
    "09": "Tobacco",
    "10": "Building stone",
    "11": "Natural sands",
    "12": "Gravel/crushed stone",
    "13": "Nonmetallic minerals",
    "14": "Metallic ores",
    "15": "Coal",
    "16": "Crude petroleum",
    "17": "Gasoline",
    "18": "Fuel oils",
    "19": "Natural gas/other coal",
    "20": "Basic chemicals",
    "21": "Pharmaceutical products",
    "22": "Fertilizers",
    "23": "Chemical products",
    "24": "Plastics/rubber",
    "25": "Logs/other wood",
    "26": "Wood products",
    "27": "Pulp/newsprint",
    "28": "Paper articles",
    "29": "Printed products",
    "30": "Textiles/leather",
    "31": "Nonmetal mineral products",
    "32": "Base metals",
    "33": "Articles of base metal",
    "34": "Machinery",
    "35": "Electronics",
    "36": "Motorized vehicles",
    "37": "Transportation equipment",
    "38": "Precision instruments",
    "39": "Furniture",
    "40": "Misc manufactured",
    "41": "Waste/scrap",
    "43": "Mixed freight",
}

df["sctg2"] = df[sctg_col].str[:2]
sctg_grp = df.groupby("sctg2")[val_col]
sctg_stats = sctg_grp.agg(
    count="count",
    mean_value="mean",
    median_value="median",
    total_value="sum",
)

print("\n  Top 20 SCTG by ROW COUNT:")
print(
    f"  {'SCTG':>5}  {'Label':<28}  {'Count':>10}  {'%Rows':>6}  {'Mean $':>10}  {'Median $':>10}"
)
print(f"  {'-' * 5}  {'-' * 28}  {'-' * 10}  {'-' * 6}  {'-' * 10}  {'-' * 10}")
top_count = sctg_stats.sort_values("count", ascending=False).head(20)
for sctg, row in top_count.iterrows():
    lbl = sctg_labels.get(str(sctg), "?")
    pct = 100 * row["count"] / len(df)
    print(
        f"  {sctg:>5}  {lbl:<28}  {row['count']:>10,}  {pct:>5.1f}%  "
        f"{row['mean_value']:>10,.0f}  {row['median_value']:>10,.0f}"
    )

print("\n  Top 20 SCTG by MEAN SHIPMENT VALUE:")
print(f"  {'SCTG':>5}  {'Label':<28}  {'Count':>10}  {'Mean $':>12}  {'Median $':>10}")
print(f"  {'-' * 5}  {'-' * 28}  {'-' * 10}  {'-' * 12}  {'-' * 10}")
top_val = sctg_stats.sort_values("mean_value", ascending=False).head(20)
for sctg, row in top_val.iterrows():
    lbl = sctg_labels.get(str(sctg), "?")
    print(
        f"  {sctg:>5}  {lbl:<28}  {row['count']:>10,}  "
        f"{row['mean_value']:>12,.0f}  {row['median_value']:>10,.0f}"
    )

# ─────────────────────────────────────────────
# 4. SHIPMT_DIST_ROUTED distribution
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("4. SHIPMT_DIST_ROUTED distribution (miles)")
print("=" * 60)
d = df[dist_col].dropna()
print(f"  Count  : {len(d):,}")
print(f"  Mean   : {d.mean():,.1f}")
print(f"  Median : {d.median():,.1f}")
print(f"  Std    : {d.std():,.1f}")
pcts2 = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]
pct_vals2 = np.percentile(d, pcts2)
print()
print("  Percentiles:")
for p, val in zip(pcts2, pct_vals2):
    print(f"    p{str(p).rjust(5)} : {val:>10,.1f}  miles")

print()
print("  Distance bucket counts:")
dbuckets = [0, 50, 100, 250, 500, 1000, 2000, 3000, np.inf]
dlabels = ["<50", "50-100", "100-250", "250-500", "500-1k", "1k-2k", "2k-3k", ">3k"]
dcuts = pd.cut(d, bins=dbuckets, labels=dlabels, right=False)
dbc = dcuts.value_counts().sort_index()
for lbl, cnt in dbc.items():
    pct = 100 * cnt / len(d)
    print(f"    {lbl:>8s} : {cnt:>10,}  ({pct:5.1f}%)")

# ─────────────────────────────────────────────
# 5. Segmentation signals
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("5. Segmentation signals")
print("=" * 60)

# Value skewness
from scipy.stats import skew, kurtosis

print(f"\n  SHIPMT_VALUE skewness : {skew(v):.2f}")
print(f"  SHIPMT_VALUE kurtosis : {kurtosis(v):.2f}")
log_v = np.log1p(v[v > 0])
print(f"  log(SHIPMT_VALUE) skewness : {skew(log_v):.2f}")
print(f"  log(SHIPMT_VALUE) kurtosis : {kurtosis(log_v):.2f}")

# Cross: value by distance band
print()
print("  Mean SHIPMT_VALUE by distance band:")
df["dist_band"] = pd.cut(df[dist_col], bins=dbuckets, labels=dlabels, right=False)
xval = df.groupby("dist_band")[val_col].agg(["count", "mean", "median"])
for band, row in xval.iterrows():
    print(
        f"    {band:>8s} : count={row['count']:>9,}  mean=${row['mean']:>10,.0f}  median=${row['median']:>8,.0f}"
    )

# Cross: value by top modes
print()
print("  Median SHIPMT_VALUE by mode (sorted by median desc):")
mode_med = df.groupby(mode_col)[val_col].median().sort_values(ascending=False)
for mode, med in mode_med.items():
    lbl = mode_labels.get(str(mode).zfill(2), "?")
    print(f"    Mode {str(mode).zfill(2)} ({lbl:<22}): median ${med:>10,.0f}")

# Zero/near-zero values
print()
zero_count = (v == 0).sum()
lt1_count = (v < 1).sum()
lt10_count = (v < 10).sum()
print(f"  Rows with SHIPMT_VALUE == 0  : {zero_count:,}")
print(f"  Rows with SHIPMT_VALUE < 1   : {lt1_count:,}")
print(f"  Rows with SHIPMT_VALUE < 10  : {lt10_count:,}")

# Weighted factor spread (proxy for survey design)
print()
wf = df["WGT_FACTOR"].dropna()
print(
    f"  WGT_FACTOR  min={wf.min():.1f}  max={wf.max():.1f}  mean={wf.mean():.1f}  median={wf.median():.1f}"
)

print()
print("Done.")
