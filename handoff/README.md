# Stakeholder handoff package

This folder contains a self-contained copy of the **Jugo IBCS Analysis web app** — everything needed to run the upload-and-analyze UI. No notebooks, training scripts, or dev tooling.

## Deliverable

| Item | Description |
|------|-------------|
| `jugo-ibcs-web/` | Full runnable app (share this folder or the zip) |
| `jugo-ibcs-web.zip` | Same contents, ready to email or upload |

## Recipient instructions

1. Unzip (if needed) and open `jugo-ibcs-web/README.md`
2. **macOS/Linux:** `./setup.sh` → edit `.env` → `./start.sh`  
   **Windows:** double-click `setup.bat` → edit `.env` → double-click `start.bat`
3. Open http://127.0.0.1:8000

## What's included vs excluded

**Included:** web server, frontend, ML pipeline, optional live LLM analysis (user OpenAI key in UI), compliance report JSON, setup scripts.

**Excluded:** YOLO training, synthetic data generation, Jupyter notebooks, test suite, `outputs/` history from development, scaling-detection dev tools (Streamlit, benchmarks).
