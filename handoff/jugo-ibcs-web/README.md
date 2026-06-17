# Jugo IBCS Analysis — Web App Handoff

Upload a dashboard image and get an automated IBCS compliance check: chart detection, axis scaling checks, data labels, and bar analysis.

This tool is built for **explainability**, **interpretability**, and **transparency** — so stakeholders can see not just *what* was flagged, but *why*, with evidence they can verify.

---

## Explainability, interpretability & transparency

Automated chart review only earns trust when people can follow the reasoning. This app is designed around three ideas:

### Interpretability — *How does the AI think?*

Results are broken down so non-technical reviewers can **trace the pipeline step by step**:

- **Per-chart cards** — each detected chart is analyzed separately with its own crops, checks, and overlays.
- **Visual overlays** — label detection and bar geometry are drawn on the chart image so you can compare the machine’s reading to what you see.
- **Structured statuses** — clear labels like `pass`, `fail`, `unknown`, and `skipped` instead of opaque model outputs.
- **Rule score ring** — a quick visual summary of which IBCS rules passed, with each segment linked to its evidence.

The goal is that a finance or design reviewer can open the results and say: *“I see the chart it picked, I see the rule it applied, I see the proof.”*

### Explainability — *why did the system say that?*

Every result is meant to be **auditable in plain language**, not just a score.

- **IBCS rule checks** show pass, fail, or unknown for each rule (axis baseline, consistent scaling, zoom requirement, labelling), each with a written **evidence** note explaining what was observed.
- **Violations** are listed explicitly when a dashboard is non-compliant.
- **Per-chart ML outputs** describe what was detected: whether the value axis starts at zero, which OCR readings were used, which data labels were found, and how many bars were identified.
- A **summary explanation** from live LLM analysis or a matching bundled report.

You should never have to guess why a chart was marked compliant or not.

### Transparency — *what is the system doing, and what are its limits?*

The pipeline is **open about its methods and boundaries**:

- **Documented steps** — chart detection (Roboflow), axis OCR (EasyOCR / Tesseract), label reading, and OpenCV bar detection are separate, inspectable stages — not a single black box.
- **Optional live LLM audit** — turn on LLM analysis in the UI with **your own OpenAI API key** to generate compliance explanations for any image (key is sent per request only, never stored).
- **Honest uncertainty** — when the system cannot read an axis or compare scales reliably, it reports **unknown** rather than guessing.
- **Known scope** — bar geometry analysis applies to **vertical** bar charts; horizontal charts are flagged but bar detection is skipped with a stated reason.

Transparency here means showing what was measured, what was inferred, and where human judgment may still be needed.

---

## What you need

1. **A computer running macOS, Linux, or Windows**
2. **Python 3.10, 3.11, or 3.12** — not 3.13 or newer
  Download: [python.org/downloads](https://www.python.org/downloads/)  
   On Windows, check **“Add python.exe to PATH”** during install.
3. **A Roboflow API key** — used to detect bar charts in your image
  Get one at [Roboflow API settings](https://app.roboflow.com/settings/api)
4. **(Optional) An OpenAI API key** — only if you want live LLM compliance analysis in the UI
  Get one at [OpenAI API keys](https://platform.openai.com/api-keys)

Internet is required for chart detection (Roboflow), optional LLM analysis (OpenAI), and the first EasyOCR model download.

---

## Quick start (3 steps)

### 1. Install

Pick your operating system:

#### macOS / Linux

Open Terminal, go to this folder, and run:

```bash
chmod +x setup.sh start.sh
./setup.sh
```

#### Windows (easiest — double-click)

1. Double-click `**setup.bat**`
2. Wait for install to finish (first run can take several minutes)

#### Windows (PowerShell)

If double-click is blocked, open PowerShell in this folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

This creates a virtual environment and installs dependencies. EasyOCR downloads models on first use.

### 2. Add your Roboflow API key

Open the `**.env**` file in Notepad (Windows) or any text editor and replace `your_api_key_here` with your real Roboflow key:

```
ROBOFLOW_API_KEY=paste_your_key_here
```

Save the file. This key stays on your machine in `.env` — it is required for chart detection.

### 3. Start the app

#### macOS / Linux

```bash
./start.sh
```

#### Windows

Double-click `**start.bat**`, or in PowerShell:

```powershell
.\start.ps1
```

Open your browser to **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

Drag and drop a PNG, JPG, or WebP dashboard image (max 15 MB), then click **Analyze Dashboard**.

---

## Live LLM compliance analysis (optional)

To analyze **any** dashboard with fresh explanations:

1. In the web UI, turn on **Run live LLM compliance analysis**
2. Paste your **OpenAI API key** in the field that appears
3. Upload your image and click **Analyze Dashboard**

What happens:

- ML overlays (chart detection, labels, bars) still run as usual
- OpenAI reviews the full dashboard image against the four IBCS rules
- You get live rule checks, violations, per-chart notes, and a summary — labeled **Live LLM analysis** in the results

**Privacy:** your OpenAI key is sent only with that one request and is **not saved** on the server or written to disk. LLM results are saved as `llm_compliance.json` inside the run folder under `outputs/web_runs/`.

**Timing:** LLM analysis can take 1–3 minutes depending on image size and OpenAI response time.

---

## System overview

When you upload an image, the server runs a **transparent, multi-step pipeline** and returns **explainable, per-chart results**:


| Step                   | What happens                                           | What you see                                 |
| ---------------------- | ------------------------------------------------------ | -------------------------------------------- |
| 1. Chart detection     | Finds bar charts in the dashboard (Roboflow)           | Chart crops and bounding boxes               |
| 2. Start-at-zero check | OCR reads axis ticks and tests for a zero baseline     | Pass/fail with OCR evidence                  |
| 3. Data labels         | Reads numeric labels on bars                           | Detected values and label overlay            |
| 4. Bar geometry        | OpenCV pipeline on vertical charts                     | Bar count, axes, bar overlay                 |
| 5. IBCS rules          | Live LLM audit **or** bundled report match by filename | Rule checks, violations, written explanation |


All of this is designed so reviewers can **interpret** each finding and **explain** it to colleagues without opening code.

---

## Troubleshooting


| Problem                                                   | What to try                                                                                            |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `ROBOFLOW_API_KEY is not configured`                      | Set the key in `.env`, restart the start script                                                        |
| `OpenAI API key is required when LLM analysis is enabled` | Turn off LLM toggle, or paste a valid key in the UI field                                              |
| `LLM analysis failed`                                     | Check internet, API key, and billing; try a smaller image                                              |
| `ML dependencies are not installed`                       | Run setup again with Python 3.10–3.12                                                                  |
| Page loads but analysis fails                             | Check internet and API key; visit [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health) |
| First analysis is very slow                               | Normal — EasyOCR loads on first use; LLM adds 1–3 min                                                  |
| No compliance explanations                                | Enable LLM in the UI, or upload with a filename that matches the bundled report                        |
| **Windows:** `python is not recognized`                   | Reinstall Python with “Add to PATH”, or use `py -3.12`                                                 |
| **Windows:** PowerShell blocks scripts                    | Use `.bat` files, or `Set-ExecutionPolicy -Scope Process Bypass`                                       |
| **Windows:** Window closes immediately                    | Open Command Prompt, `cd` to this folder, run `setup.bat` or `start.bat`                               |


---

## Folder contents

```
jugo-ibcs-web/
├── README.md              ← you are here
├── setup.sh / start.sh    ← macOS & Linux
├── setup.bat / start.bat  ← Windows (double-click)
├── setup.ps1 / start.ps1  ← Windows (PowerShell)
├── .env.example           ← Roboflow API key template
├── requirements.txt       ← Python packages
├── main.py                ← ML analysis pipeline
├── llm_audit.py           ← LLM wrapper for the web app
├── llm-test-openai.py     ← OpenAI IBCS audit logic
├── label_detection.py     ← data label reading
├── bar_detection.py       ← bar geometry
├── scaling-detection/
│   └── pipeline.py        ← OpenCV bar detection
├── web/
│   ├── server.py          ← web server
│   ├── compliance.py      ← bundled report matching
│   └── static/            ← browser UI
├── ibcs_compliance_report_openai.json   ← demo reports (filename match)
└── outputs/web_runs/        ← analysis results (created per upload)
```

---

## Stopping the server

Press **Ctrl+C** in the Terminal or Command Prompt window where the server is running.

---

## Support notes for IT

- **Port:** 8000 (local only by default)
- **Env vars:** `ROBOFLOW_API_KEY` (required on server), `ROBOFLOW_WORKSPACE`, `ROBOFLOW_WORKFLOW_ID` (optional)
- **LLM:** user-supplied OpenAI key per request via UI; not persisted server-side
- **Manual start (macOS/Linux):** `source .venv/bin/activate && uvicorn web.server:app --host 127.0.0.1 --port 8000`
- **Manual start (Windows):** `.venv\Scripts\activate` then `uvicorn web.server:app --host 127.0.0.1 --port 8000`
- **Tesseract** is optional; improves some axis OCR paths if installed (`brew install tesseract` on Mac, [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki))

