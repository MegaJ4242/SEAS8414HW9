#!/usr/bin/env python3
"""
HW 9 — Part 1
1_train_and_export.py

- Generates a simple training dataset (length, entropy, label)
- Trains an H2O AutoML classifier
- Exports the leader (or next best) as a MOJO zip into ./model/DGA_Leader.zip
- Saves artifacts/leaderboard.csv and data/dga_dataset_train.csv
"""
import argparse
from pathlib import Path
import math
import random
import string
import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML


def shannon_entropy(s: str) -> float:
    """Compute Shannon entropy (base-2)."""
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def random_domain(length: int, tld: str = "com") -> str:
    """Generate a random-looking domain token of given length."""
    alphabet = string.ascii_lowercase + string.digits
    token = "".join(random.choice(alphabet) for _ in range(length))
    return f"{token}.{tld}"


def synthesize_dataset(n_legit: int = 1500, n_dga: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Create a small synthetic dataset with simple features:
      - length: len of the first label (before first dot)
      - entropy: Shannon entropy of that first label
      - label: 'legit' or 'dga'
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    rows = []

    # Legit: shorter, lower entropy, pronounceable patterns
    syllables = [
        "ba","be","bi","bo","bu","ca","ce","ci","co","cu","da","de","di","do","du",
        "fa","fe","fi","fo","fu","la","le","li","lo","lu","ma","me","mi","mo","mu"
    ]
    legit_tlds = ["com", "org", "net", "info"]
    for _ in range(n_legit):
        token_len = int(rng.integers(4, 12))
        token = "".join(rng.choice(syllables) for _ in range(max(2, token_len // 2)))
        token = token[:token_len]
        dom = f"{token}.{rng.choice(legit_tlds)}"
        first = dom.split(".")[0]
        rows.append({"domain": dom, "length": len(first), "entropy": shannon_entropy(first), "label": "legit"})

    # DGA: longer, higher entropy, more digits
    dga_tlds = ["xyz", "top", "site", "info"]
    for _ in range(n_dga):
        token_len = int(rng.integers(10, 24))
        dom = random_domain(token_len, tld=rng.choice(dga_tlds))
        first = dom.split(".")[0]
        rows.append({"domain": dom, "length": len(first), "entropy": shannon_entropy(first), "label": "dga"})

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="model", help="Directory to write the exported MOJO")
    ap.add_argument("--runtime", type=int, default=90, help="Max runtime (seconds) for AutoML")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--rows", type=int, default=3000, help="Total rows to synthesize")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # 1) Build dataset
    n_legit = args.rows // 2
    n_dga = args.rows - n_legit
    df = synthesize_dataset(n_legit=n_legit, n_dga=n_dga, seed=args.seed)
    csv_path = data_dir / "dga_dataset_train.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Wrote training data -> {csv_path} ({len(df)} rows)")

    # 2) Start H2O and run AutoML
    print("[*] Initializing H2O...")
    h2o.init(max_mem_size="2G", nthreads=-1)

    hf = h2o.H2OFrame(df[["length", "entropy", "label"]])
    hf["label"] = hf["label"].asfactor()
    train, test = hf.split_frame(ratios=[0.8], seed=args.seed)

    x = ["length", "entropy"]
    y = "label"
    aml = H2OAutoML(
        max_runtime_secs=args.runtime,
        seed=args.seed,
        sort_metric="AUC",
        verbosity="info"
    )
    print(f"[*] Running AutoML for ~{args.runtime}s...")
    aml.train(x=x, y=y, training_frame=train, leaderboard_frame=test)

    leader = aml.leader
    print(f"[OK] Leader model: {leader.algo}  id={leader.model_id}")

    # Save leaderboard
    lb_df = aml.leaderboard.as_data_frame()
    lb_csv = artifacts_dir / "leaderboard.csv"
    lb_df.to_csv(lb_csv, index=False)
    print(f"[OK] Saved leaderboard -> {lb_csv}")

    # 3) Export MOJO (leader first, then fall back through leaderboard)
    candidates = [leader.model_id] + [m for m in lb_df["model_id"].tolist() if m != leader.model_id]
    mojo_path = None
    for mid in candidates:
        try:
            mdl = h2o.get_model(mid)
            mojo_path = mdl.download_mojo(path=str(outdir), get_genmodel_jar=True)
            print(f"[OK] Exported MOJO from model: {mid}")
            break
        except Exception as e:
            print(f"[!] Could not export MOJO for {mid}: {e}")

    if not mojo_path:
        raise RuntimeError("Failed to export MOJO from any leaderboard model.")

    mp = Path(mojo_path)
    target = outdir / "DGA_Leader.zip"
    if mp.resolve() != target.resolve():
        try:
            if target.exists():
                target.unlink()
            mp.rename(target)
            print(f"[OK] Renamed MOJO to -> {target}")
        except Exception:
            import shutil as _sh
            _sh.copy2(mp, target)
            print(f"[OK] Copied MOJO to -> {target}")

    print("[DONE] Training complete.")
    print(f"    MOJO: {target}")
    print(f"    Data: {csv_path}")
    print(f"    Leaderboard: {lb_csv}")


if __name__ == "__main__":
    main()
