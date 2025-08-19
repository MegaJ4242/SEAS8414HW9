#!/usr/bin/env python3
"""
HW 9 — Part 1
2_analyze_domain.py
- Loads the exported MOJO (./model/DGA_Leader.zip)
- Computes features for an input domain (length, entropy)
- Predicts class using H2O Generic Estimator from MOJO
- If predicted DGA: builds a local SHAP explanation (KernelExplainer)
- Bridges XAI -> GenAI to create a prescriptive incident-response playbook
"""
import argparse
import os
import numpy as np
import pandas as pd
import h2o
from h2o.estimators import H2OGenericEstimator

USE_GENAI = True
try:
    import google.generativeai as genai
except Exception:
    USE_GENAI = False

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c)/len(s) for c in set(s)]
    import math as _m
    return -sum(p*_m.log2(p) for p in probs if p > 0)

def extract_token(domain: str) -> str:
    d = domain.lower().strip()
    for prefix in ("http://", "https://"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    d = d.split("/")[0]
    token = d.split(".")[0]
    return "".join(ch for ch in token if ch.isalnum())

def calc_features(domain: str) -> dict:
    token = extract_token(domain)
    return {"length": float(len(token)), "entropy": float(shannon_entropy(token))}

def load_mojo(path: str) -> H2OGenericEstimator:
    h2o.init(max_mem_size="1G", nthreads=-1)
    from pathlib import Path
    mojo_zip = Path(path)
    if not mojo_zip.exists():
        raise FileNotFoundError(f"MOJO not found at {path}. Run 1_train_and_export.py first.")
    model = H2OGenericEstimator.from_file(file=str(mojo_zip))
    return model

def predict_proba_dga(model: H2OGenericEstimator, feats_df: pd.DataFrame):
    hf = h2o.H2OFrame(feats_df)
    pred = model.predict(hf).as_data_frame()
    label = str(pred.loc[0, "predict"]) if "predict" in pred.columns else "unknown"
    if "dga" in pred.columns:
        proba = float(pred.loc[0, "dga"])
    elif "p1" in pred.columns:
        proba = float(pred.loc[0, "p1"])
    elif label in pred.columns:
        proba = float(pred.loc[0, label])
    else:
        prob_cols = [c for c in pred.columns if c != "predict"]
        proba = float(pred.loc[0, prob_cols].max()) if prob_cols else float("nan")
    return label, proba, pred

def predict_proba_vector(model, feats_df: pd.DataFrame, target: str = "dga") -> np.ndarray:
    """Return a 1-D numpy array of P(target) for all rows in feats_df."""
    hf = h2o.H2OFrame(feats_df)
    pred = model.predict(hf).as_data_frame()

    # Identify the probability column robustly
    if target in pred.columns:
        probs = pred[target].to_numpy()
    elif "p1" in pred.columns:
        probs = pred["p1"].to_numpy()
    elif "dga" in pred.columns and "legit" in pred.columns:
        probs = pred["dga"].to_numpy()
    else:
        prob_cols = [c for c in pred.columns if c != "predict"]
        # fallback: use the last prob column
        probs = pred[prob_cols[-1]].to_numpy()
    return probs

def kernel_shap_local_explanation(model, background_X: np.ndarray, instance_X: np.ndarray, feature_names: list[str]) -> dict:
    import shap

    def f(X: np.ndarray) -> np.ndarray:
        # X is (n_samples, n_features). Return (n_samples,) of P(dga).
        df = pd.DataFrame(X, columns=feature_names)
        return predict_proba_vector(model, df, target="dga")

    explainer = shap.KernelExplainer(model=f, data=background_X)
    # Keep samples modest for speed; bump if you want more precision
    shap_vals = explainer.shap_values(instance_X, nsamples=150)

    # Handle SHAP return shape variants
    if isinstance(shap_vals, list):
        shap_vec = np.array(shap_vals[-1][0])
    else:
        shap_vec = np.array(shap_vals[0])

    base_value = getattr(explainer, "expected_value", None)
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_value = float(base_value[-1])
    else:
        base_value = float(base_value) if base_value is not None else None

    contrib = dict(zip(feature_names, shap_vec.tolist()))
    ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {"base_value": base_value, "contrib": contrib, "ranked": ranked}

def generate_playbook_with_genai(xai_findings: str) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not (USE_GENAI and api_key):
        return (
            "GenAI disabled or GOOGLE_API_KEY not set.\n\n"
            "=== Analyst Playbook (Template) ===\n"
            "1) Block domain at DNS/email gateways.\n"
            "2) Hunt historical DNS for domain/variants.\n"
            "3) Isolate contacting hosts; acquire triage.\n"
            "4) Add Sigma/Suricata/YARA detections.\n"
            "5) Open incident; notify stakeholders.\n"
            "6) Monitor for recurrence; document KB."
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are a SOC TL. Based on the model decision summary below, produce a concise, step-by-step, context-aware incident response playbook.
Group by phase (Containment, Eradication, Detection, Monitoring, Comms). Keep under 25 bullets.

{xai_findings}
"""
    resp = model.generate_content(prompt)
    return getattr(resp, "text", "").strip() or "GenAI returned empty text."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--mojo", default="model/DGA_Leader.zip")
    ap.add_argument("--background", default="data/dga_dataset_train.csv")
    args = ap.parse_args()

    model = load_mojo(args.mojo)
    feats = calc_features(args.domain)
    feats_df = pd.DataFrame([feats])

    label, p_dga, pred_df = predict_proba_dga(model, feats_df)
    print("=== Prediction ===")
    print({"domain": args.domain, "predicted_label": label, "prob_dga": p_dga})
    if label != "dga":
        print("\nVerdict is LEGIT (or below threshold). No playbook generated.")
        return

    bg = pd.read_csv(args.background)[["length","entropy"]].sample(n=100, random_state=42)
    instance = np.array([[feats["length"], feats["entropy"]]])
    xai = kernel_shap_local_explanation(model, bg.to_numpy(), instance, ["length","entropy"])

    top = xai["ranked"]
    top_txt = "\n".join([f"  - {k}: SHAP={v:+.4f}" for k, v in top])
    xai_findings = (
        f"- Alert: Potential DGA domain detected.\n"
        f"- Domain: '{args.domain}'\n"
        f"- Predicted probability (dga): {p_dga:.4f}\n"
        f"- Local explanation (Kernel SHAP over [length, entropy]):\n{top_txt}\n"
    )
    playbook = generate_playbook_with_genai(xai_findings)

    print("\n=== XAI Findings ===")
    print(xai_findings)
    print("\n=== Prescriptive Playbook ===")
    print(playbook)

if __name__ == "__main__":
    main()
