# Prescriptive DGA Detector (HW 9)
An end-to-end, command-line pipeline that:
1) **Trains** a high-performing DGA detector with **H2O AutoML** and exports a production **MOJO**.
2) **Explains** per-alert decisions with **SHAP** (Kernel SHAP, local explanation).
3) **Prescribes** next steps by converting the SHAP findings into a tailored **incident-response playbook** via **Generative AI**.

---

## Architecture (3 stages)

- **Model (AutoML + MOJO)**  
  `1_train_and_export.py` synthesizes a small dataset (domain **length**, **entropy**) → runs AutoML → exports the leader as `model/DGA_Leader.zip` (MOJO).

- **Explainability (SHAP)**  
  `2_analyze_domain.py` loads the MOJO, scores a single domain, and—if classified **DGA**—computes a **local** explanation (Kernel SHAP) over `[length, entropy]`.

- **Prescriptive GenAI**  
  The script formats a compact `xai_findings` summary (domain, probability, feature values, ranked SHAP contributions) and prompts a Generative AI model to return a concise, phase-grouped **playbook**.  
  If no API key is present, it auto-falls back to a **template** playbook.

---

## Quick Start

### Prerequisites
- Python 3.11+ recommended (works with 3.12)
- (Windows) Git Bash or PowerShell
- Internet for AutoML packages and (optional) GenAI

### Create & activate a venv (Windows Git Bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
