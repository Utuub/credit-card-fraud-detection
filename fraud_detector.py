"""
Credit card fraud detector.

Scores every transaction in the dataset with three simple risk rules, flags
the risky ones, sorts them so the likeliest fraud is reviewed first, and
checks the flags against the dataset's real fraud labels.

Usage:
    python fraud_detector.py [path/to/data]      (default: ./data)
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
OUT = Path("output")
SPLIT_DATE = "2017-01-01"   # learn risky categories before this date, evaluate on or after it
FLAG_AT = 2                 # risk score that sends a transaction to review


def step(msg):
    print(f"\n{msg}", flush=True)


start = time.time()

# ---- 1. load ---------------------------------------------------------------
step("[1/6] Loading transactions...")
tx = pd.read_csv(
    DATA / "transactions_data.csv",
    usecols=["id", "date", "client_id", "amount", "mcc"],
    dtype={"id": "int64", "client_id": "int32", "mcc": "int32", "amount": "string"},
)
tx["amount"] = pd.to_numeric(tx["amount"].str.replace("$", "", regex=False), errors="coerce").astype("float32")
tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
print(f"  {len(tx):,} transactions")

with open(DATA / "mcc_codes.json") as f:
    mcc_names = {int(k): v for k, v in json.load(f).items()}

with open(DATA / "train_fraud_labels.json") as f:
    labels = json.load(f)["target"]
lab = pd.Series(labels, name="label")
del labels
lab.index = lab.index.astype("int64")
tx["is_fraud"] = tx["id"].map(lab.map({"Yes": 1, "No": 0})).astype("float32")  # NaN = unlabeled
del lab
labeled = tx["is_fraud"].notna()
print(f"  {labeled.sum():,} have a fraud label ({int(tx['is_fraud'].sum()):,} confirmed fraud)")

# ---- 2. clean --------------------------------------------------------------
step("[2/6] Cleaning...")
before = len(tx)
tx = tx.drop_duplicates(subset="id").dropna(subset=["date", "amount"])
print(f"  dropped {before - len(tx):,} duplicate or unreadable rows")

# ---- 3. features -----------------------------------------------------------
step("[3/6] Building features...")
tx["hour"] = tx["date"].dt.hour.astype("int8")
tx["user_avg"] = tx.groupby("client_id")["amount"].transform("mean").astype("float32")
tx["category"] = tx["mcc"].map(mcc_names).fillna("Unknown")

# risky categories are learned only from the earlier period, so the evaluation is honest
train = tx[(tx["date"] < SPLIT_DATE) & tx["is_fraud"].notna()]
cat_rate = train.groupby("mcc")["is_fraud"].mean()
risky_mcc = set(cat_rate[cat_rate > cat_rate.quantile(0.75)].index)
print(f"  {len(risky_mcc)} high-risk merchant categories (top quarter by fraud rate before {SPLIT_DATE[:4]})")

# ---- 4. score --------------------------------------------------------------
step("[4/6] Scoring...")
score = np.zeros(len(tx), dtype="int8")
score += 2 * (tx["amount"] > 2 * tx["user_avg"]).to_numpy()   # more than 2x this customer's average
score += 1 * tx["hour"].between(0, 5).to_numpy()               # midnight to 5 AM
score += 1 * tx["mcc"].isin(risky_mcc).to_numpy()              # high-risk merchant category
tx["risk_score"] = score
tx["flagged"] = tx["risk_score"] >= FLAG_AT
print(f"  flagged {tx['flagged'].sum():,} of {len(tx):,} transactions ({tx['flagged'].mean():.1%})")

# ---- 5. evaluate on the held-out period ------------------------------------
step(f"[5/6] Checking flags against real labels ({SPLIT_DATE[:4]} onward)...")
test = tx[(tx["date"] >= SPLIT_DATE) & tx["is_fraud"].notna()]
fraud = test["is_fraud"] == 1
tp = int((test["flagged"] & fraud).sum())
recall = tp / max(int(fraud.sum()), 1)
precision = tp / max(int(test["flagged"].sum()), 1)
base = fraud.mean()
print(f"  labeled transactions: {len(test):,}, fraud: {int(fraud.sum()):,} ({base:.3%})")
print(f"  recall:    {recall:.1%} of fraud gets flagged")
print(f"  precision: {precision:.3%} of flags are fraud ({precision / base:.1f}x the base rate)")
by_score = test.groupby("risk_score")["is_fraud"].agg(["count", "mean"])
by_score.columns = ["transactions", "fraud_rate"]
print("\n  fraud rate by risk score:")
for s, row in by_score.iterrows():
    print(f"    score {s}: {row.fraud_rate:.3%} of {int(row.transactions):,}")

# ---- 6. export -------------------------------------------------------------
step("[6/6] Saving results...")
OUT.mkdir(exist_ok=True)
review = (tx[tx["flagged"]]
          .sort_values(["risk_score", "amount"], ascending=[False, False])
          [["id", "date", "client_id", "amount", "category", "hour", "risk_score", "is_fraud"]])
review.to_csv(OUT / "review_queue.csv", index=False)
print(f"  {OUT / 'review_queue.csv'}: {len(review):,} flagged transactions, riskiest first")

summary = {
    "transactions_scored": int(len(tx)),
    "flagged": int(tx["flagged"].sum()),
    "eval_period": f"{SPLIT_DATE} onward",
    "eval_labeled": int(len(test)),
    "eval_fraud": int(fraud.sum()),
    "recall": round(recall, 4),
    "precision": round(precision, 5),
    "base_fraud_rate": round(float(base), 5),
    "lift": round(precision / base, 2),
    "fraud_rate_by_score": {int(k): round(float(v), 5) for k, v in by_score["fraud_rate"].items()},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
counts = tx["risk_score"].value_counts().sort_index()
ax[0].bar(counts.index, counts.values, color="#e08a2c")
ax[0].set(title="Transactions by risk score", xlabel="risk score", ylabel="transactions")
ax[1].bar(by_score.index, by_score["fraud_rate"] * 100, color="#c0392b")
ax[1].set(title=f"Fraud rate by risk score ({SPLIT_DATE[:4]}+)", xlabel="risk score", ylabel="fraud rate (%)")
hr = test.groupby("hour")["is_fraud"].mean() * 100
ax[2].plot(hr.index, hr.values, marker="o", color="#c0392b")
ax[2].set(title="Fraud rate by hour of day", xlabel="hour", ylabel="fraud rate (%)")
ax[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fraud_overview.png", dpi=110)
print("  fraud_overview.png")

print(f"\nDone in {time.time() - start:.0f}s")
