"""FastAPI server for the Jugo IBCS analysis frontend."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from web.compliance import (
    enrich_chart_with_compliance,
    load_compliance_index,
    lookup_compliance_report,
    sanitize_compliance_report,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STATIC_DIR = Path(__file__).resolve().parent / "static"
WEB_RUNS_DIR = REPO_ROOT / "outputs" / "web_runs"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

DEFAULT_WORKSPACE = "khas-workspace-3cwa2"
DEFAULT_WORKFLOW_ID = "bar-chart-detection-and-crop-1779799226718"


def _import_llm_audit():
    try:
        from llm_audit import audit_dashboard_image, llm_audit_available
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM dependencies are not installed. Run "
                "`pip install openai tqdm` and ensure llm-test-openai.py is present."
            ),
        ) from exc
    if not llm_audit_available():
        raise HTTPException(
            status_code=503,
            detail="LLM audit is not available (missing openai package or llm-test-openai.py).",
        )
    return audit_dashboard_image


def _run_llm_audit(
    image_path: Path,
    api_key: str,
    source_filename: str,
    run_dir: Path,
) -> Dict[str, Any]:
    audit_dashboard_image = _import_llm_audit()
    result = audit_dashboard_image(
        image_path,
        api_key,
        filename=Path(source_filename).name,
    )
    (run_dir / "llm_compliance.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _import_pipeline():
    """Import ML pipeline lazily so the static UI can start without all deps."""
    try:
        from main import analyze_image, get_easyocr_reader
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "ML dependencies are not installed. Use Python 3.10–3.12, recreate "
                "the venv with `python3.12 -m venv .venv`, then run "
                "`pip install -r requirements.txt`."
            ),
        ) from exc
    return analyze_image, get_easyocr_reader


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm up EasyOCR on startup so the first upload is faster."""
    load_compliance_index()
    if os.getenv("ROBOFLOW_API_KEY"):
        try:
            _, get_easyocr_reader = _import_pipeline()
            get_easyocr_reader()
        except HTTPException:
            pass
        except Exception:
            pass
    yield


app = FastAPI(title="Jugo IBCS Analysis", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _artifact_url(run_id: str, relative_path: str) -> str:
    return f"/api/artifacts/{run_id}/{relative_path.replace(os.sep, '/')}"


def _rewrite_path(run_id: str, run_dir: Path, value: Any) -> Any:
    if isinstance(value, str):
        try:
            path = Path(value).resolve()
            rel = path.relative_to(run_dir.resolve())
            return _artifact_url(run_id, str(rel))
        except ValueError:
            return value
    if isinstance(value, dict):
        return {k: _rewrite_path(run_id, run_dir, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_path(run_id, run_dir, item) for item in value]
    return value


def _overlay_url(run_id: str, run_dir: Path, relative_path: str) -> str | None:
    if (run_dir / relative_path).is_file():
        return _artifact_url(run_id, relative_path)
    return None


def _attach_visual_urls(run_id: str, run_dir: Path, chart: Dict[str, Any]) -> Dict[str, Any]:
    chart_id = int(chart.get("chart_id", 0))
    chart["label_overlay_url"] = _overlay_url(
        run_id, run_dir, f"label_overlays/chart_{chart_id}.png"
    )
    bar_result = chart.get("bar_detection_result") or {}
    bar_path = bar_result.get("overlay_path")
    if isinstance(bar_path, str):
        try:
            rel = Path(bar_path).resolve().relative_to(run_dir.resolve())
            chart["bar_overlay_url"] = _artifact_url(run_id, str(rel))
        except ValueError:
            chart["bar_overlay_url"] = None
    else:
        chart["bar_overlay_url"] = _overlay_url(
            run_id, run_dir, f"bar_overlays/chart_{chart_id}.png"
        )
    return chart


def _build_response(
    run_id: str,
    run_dir: Path,
    results: List[Dict[str, Any]],
    source_filename: str,
    compliance_override: Optional[Dict[str, Any]] = None,
    compliance_source: Optional[str] = None,
) -> Dict[str, Any]:
    rewritten = _rewrite_path(run_id, run_dir, results)
    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = _rewrite_path(run_id, run_dir, summary)

    raw_report = compliance_override or lookup_compliance_report(source_filename)
    compliance = sanitize_compliance_report(raw_report) if raw_report else None
    if compliance_source is None:
        compliance_source = "llm" if compliance_override else ("bundled" if compliance else None)
    charts_detected = (compliance or {}).get("charts_detected", [])

    enriched_results = []
    for chart in rewritten:
        chart = _attach_visual_urls(run_id, run_dir, chart)
        enriched_results.append(enrich_chart_with_compliance(chart, charts_detected))

    response: Dict[str, Any] = {
        "run_id": run_id,
        "source_filename": Path(source_filename).name,
        "num_charts": len(enriched_results),
        "labeled_output_url": _overlay_url(run_id, run_dir, "labeled_output.png"),
        "results_json_url": _artifact_url(run_id, "results.json"),
        "results": enriched_results,
        "summary": summary,
        "compliance": compliance,
        "compliance_matched": compliance is not None,
        "compliance_source": compliance_source,
    }

    if compliance:
        response["compliant"] = compliance.get("compliant")
        response["noncompliance_score"] = compliance.get("noncompliance_score")
    else:
        response["compliant"] = None
        response["noncompliance_score"] = None

    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    pipeline_ready = True
    pipeline_error = None
    try:
        _import_pipeline()
    except HTTPException as exc:
        pipeline_ready = False
        pipeline_error = exc.detail

    index = load_compliance_index()
    llm_ready = False
    llm_error = None
    try:
        from llm_audit import llm_audit_available

        llm_ready = llm_audit_available()
        if not llm_ready:
            llm_error = "openai package or llm-test-openai.py not available"
    except ImportError as exc:
        llm_error = str(exc)

    return {
        "status": "ok",
        "api_key_configured": bool(os.getenv("ROBOFLOW_API_KEY")),
        "python_version": sys.version.split()[0],
        "pipeline_ready": pipeline_ready,
        "pipeline_error": pipeline_error,
        "llm_ready": llm_ready,
        "llm_error": llm_error,
        "compliance_reports_loaded": len(index),
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    use_llm: str = Form("false"),
    openai_api_key: str = Form(""),
) -> JSONResponse:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ROBOFLOW_API_KEY is not configured on the server.",
        )

    use_llm_enabled = use_llm.strip().lower() in {"1", "true", "yes", "on"}
    llm_key = openai_api_key.strip()
    if use_llm_enabled and not llm_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key is required when LLM analysis is enabled.",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use PNG, JPG, or WebP.",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 15 MB limit.")

    run_id = str(uuid.uuid4())
    run_dir = WEB_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    upload_path = run_dir / f"upload{ext}"
    upload_path.write_bytes(content)

    analyze_image, _ = _import_pipeline()

    try:
        results = analyze_image(
            image_path=upload_path,
            workspace_name=os.getenv("ROBOFLOW_WORKSPACE", DEFAULT_WORKSPACE),
            workflow_id=os.getenv("ROBOFLOW_WORKFLOW_ID", DEFAULT_WORKFLOW_ID),
            api_key=api_key,
            api_url=os.getenv("ROBOFLOW_API_URL", "https://detect.roboflow.com"),
            image_input_name=os.getenv("ROBOFLOW_IMAGE_INPUT", "image"),
            output_dir=run_dir,
            output_json=run_dir / "results.json",
            crop_padding_ratio=0.08,
            assume_compliant_if_axis_missing=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    compliance_override = None
    compliance_source = None
    if use_llm_enabled:
        try:
            compliance_override = await asyncio.to_thread(
                _run_llm_audit,
                upload_path,
                llm_key,
                file.filename,
                run_dir,
            )
            compliance_source = "llm"
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM analysis failed: {exc}") from exc

    return JSONResponse(
        _build_response(
            run_id,
            run_dir,
            results,
            file.filename,
            compliance_override=compliance_override,
            compliance_source=compliance_source,
        )
    )


@app.get("/api/artifacts/{run_id}/{file_path:path}")
async def get_artifact(run_id: str, file_path: str) -> FileResponse:
    safe_run = Path(run_id).name
    artifact = (WEB_RUNS_DIR / safe_run / file_path).resolve()
    run_root = (WEB_RUNS_DIR / safe_run).resolve()

    if not str(artifact).startswith(str(run_root)) or not artifact.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")

    return FileResponse(artifact)
