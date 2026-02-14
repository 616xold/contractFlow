"""FastAPI app for interactive ContractFlow extraction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from contractflow.core.extractor import (
    DEFAULT_MODEL,
    ExtractionResult,
    extract_fields_field_agents,
    extract_fields_naive,
    extract_fields_orchestrated,
    extract_fields_retrieval,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "contractflow" / "schemas" / "contract_schema.json"
RISK_POLICY_PATH = REPO_ROOT / "docs" / "risk_policy.json"
EMBEDDING_CACHE_DIR = REPO_ROOT / "data" / "embeddings"
UI_DIR = Path(__file__).resolve().parent

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_RAW_TEXT_CHARS = 30_000

ExtractionMode = Literal["naive", "retrieval", "field_agents", "orchestrated"]
RetrievalBackend = Literal["bm25", "embeddings", "hybrid"]


@dataclass
class UIExtractionOptions:
    mode: ExtractionMode
    model: str
    retrieval_backend: RetrievalBackend
    top_k: int
    max_chunk_chars: int
    chunk_max_chars: int
    use_ocr: bool
    ocr_min_chars: int
    ocr_lang: str
    ocr_dpi: int
    enable_verifier: bool
    verifier_confidence_threshold: float
    verifier_max_repairs: int
    verifier_model: Optional[str]
    enable_risk_judge: bool
    risk_judge_model: Optional[str]
    enable_risk_review: bool
    risk_review_model: Optional[str]
    risk_review_top_k: Optional[int]
    embedding_model: str
    embedding_batch_size: int
    reranker_model: Optional[str]
    reranker_top_n: int
    strict: bool
    structured_outputs: bool


def create_app() -> FastAPI:
    app = FastAPI(
        title="ContractFlow UI",
        version="1.0.0",
        description="Interactive UI for agentic contract extraction and explainable risk analysis.",
    )
    templates = Jinja2Templates(directory=str(UI_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")

    @app.get("/api/health")
    async def health() -> Dict[str, bool]:
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "default_model": DEFAULT_MODEL,
                "default_top_k": 2,
                "default_max_chunk_chars": 900,
                "default_chunk_max_chars": 1400,
            },
        )

    @app.post("/api/extract")
    async def extract_api(
        pdf: UploadFile = File(...),
        mode: str = Form("orchestrated"),
        model: str = Form(DEFAULT_MODEL),
        retrieval_backend: str = Form("bm25"),
        top_k: int = Form(2),
        max_chunk_chars: int = Form(900),
        chunk_max_chars: int = Form(1400),
        use_ocr: bool = Form(False),
        ocr_min_chars: int = Form(40),
        ocr_lang: str = Form("eng"),
        ocr_dpi: int = Form(200),
        enable_verifier: bool = Form(True),
        verifier_confidence_threshold: float = Form(0.62),
        verifier_max_repairs: int = Form(3),
        verifier_model: str = Form(""),
        enable_risk_judge: bool = Form(True),
        risk_judge_model: str = Form(""),
        enable_risk_review: bool = Form(True),
        risk_review_model: str = Form(""),
        risk_review_top_k: int = Form(2),
        embedding_model: str = Form("text-embedding-3-small"),
        embedding_batch_size: int = Form(64),
        reranker_model: str = Form(""),
        reranker_top_n: int = Form(20),
        strict: bool = Form(False),
        structured_outputs: bool = Form(True),
    ) -> JSONResponse:
        run_id = uuid4().hex[:10]
        selected_mode = _validate_mode(mode)
        selected_backend = _validate_backend(retrieval_backend)
        clean_model = (model or "").strip()
        if not clean_model:
            raise HTTPException(status_code=400, detail="model must not be empty")

        if top_k < 1:
            raise HTTPException(status_code=400, detail="top_k must be >= 1")
        if max_chunk_chars < 200 or chunk_max_chars < 400:
            raise HTTPException(
                status_code=400,
                detail="chunk sizes are too small. Use max_chunk_chars >= 200 and chunk_max_chars >= 400",
            )
        if ocr_min_chars < 1:
            raise HTTPException(status_code=400, detail="ocr_min_chars must be >= 1")
        if verifier_max_repairs < 0:
            raise HTTPException(status_code=400, detail="verifier_max_repairs must be >= 0")
        if not (0.0 <= verifier_confidence_threshold <= 1.0):
            raise HTTPException(
                status_code=400,
                detail="verifier_confidence_threshold must be between 0 and 1",
            )

        _validate_pdf_upload(pdf)
        temp_pdf_path = await _save_upload_pdf(pdf, max_bytes=MAX_UPLOAD_BYTES)
        options = UIExtractionOptions(
            mode=selected_mode,
            model=clean_model,
            retrieval_backend=selected_backend,
            top_k=top_k,
            max_chunk_chars=max_chunk_chars,
            chunk_max_chars=chunk_max_chars,
            use_ocr=use_ocr,
            ocr_min_chars=ocr_min_chars,
            ocr_lang=(ocr_lang or "eng").strip() or "eng",
            ocr_dpi=max(72, int(ocr_dpi)),
            enable_verifier=enable_verifier,
            verifier_confidence_threshold=verifier_confidence_threshold,
            verifier_max_repairs=verifier_max_repairs,
            verifier_model=(verifier_model or "").strip() or None,
            enable_risk_judge=enable_risk_judge,
            risk_judge_model=(risk_judge_model or "").strip() or None,
            enable_risk_review=enable_risk_review,
            risk_review_model=(risk_review_model or "").strip() or None,
            risk_review_top_k=(risk_review_top_k if risk_review_top_k >= 1 else None),
            embedding_model=(embedding_model or "text-embedding-3-small").strip() or "text-embedding-3-small",
            embedding_batch_size=max(1, int(embedding_batch_size)),
            reranker_model=(reranker_model or "").strip() or None,
            reranker_top_n=max(1, int(reranker_top_n)),
            strict=strict,
            structured_outputs=structured_outputs,
        )

        start = time.perf_counter()
        try:
            result = await run_in_threadpool(_run_extraction, temp_pdf_path, options)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
        finally:
            temp_pdf_path.unlink(missing_ok=True)

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        payload = _build_response_payload(
            run_id=run_id,
            file_name=pdf.filename or temp_pdf_path.name,
            result=result,
            options=options,
            latency_ms=latency_ms,
        )
        return JSONResponse(content=_json_safe(payload))

    return app


def _validate_mode(mode: str) -> ExtractionMode:
    normalized = (mode or "").strip().lower()
    allowed = {"naive", "retrieval", "field_agents", "orchestrated"}
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")
    return normalized  # type: ignore[return-value]


def _validate_backend(backend: str) -> RetrievalBackend:
    normalized = (backend or "").strip().lower()
    allowed = {"bm25", "embeddings", "hybrid"}
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported retrieval backend: {backend}")
    return normalized  # type: ignore[return-value]


def _validate_pdf_upload(pdf: UploadFile) -> None:
    file_name = (pdf.filename or "").strip()
    if not file_name:
        raise HTTPException(status_code=400, detail="No file name was provided.")
    if not file_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")


async def _save_upload_pdf(pdf: UploadFile, *, max_bytes: int) -> Path:
    with NamedTemporaryFile(delete=False, suffix=".pdf", prefix="contractflow_ui_") as tmp_file:
        temp_path = Path(tmp_file.name)

    total = 0
    try:
        with temp_path.open("wb") as out:
            while True:
                chunk = await pdf.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF too large. Max size is {max_bytes // (1024 * 1024)} MB.",
                    )
                out.write(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        await pdf.close()

    return temp_path


def _run_extraction(pdf_path: Path, options: UIExtractionOptions) -> ExtractionResult:
    common_kwargs: Dict[str, Any] = {
        "model": options.model,
        "validate": True,
        "strict": options.strict,
        "coerce": True,
        "structured_outputs": options.structured_outputs,
        "use_ocr": options.use_ocr,
        "ocr_min_chars": options.ocr_min_chars,
        "ocr_lang": options.ocr_lang,
        "ocr_dpi": options.ocr_dpi,
        "enable_risk_judge": options.enable_risk_judge,
        "enable_risk_review": options.enable_risk_review,
        "risk_judge_model": options.risk_judge_model,
        "risk_review_model": options.risk_review_model,
        "risk_review_top_k": options.risk_review_top_k,
        "risk_policy_path": RISK_POLICY_PATH,
    }
    retrieval_kwargs: Dict[str, Any] = {
        "retrieval_backend": options.retrieval_backend,
        "embedding_model": options.embedding_model,
        "embedding_batch_size": options.embedding_batch_size,
        "embedding_cache_dir": EMBEDDING_CACHE_DIR,
        "reranker_model": options.reranker_model,
        "reranker_top_n": options.reranker_top_n,
        "top_k": options.top_k,
        "max_chunk_chars": options.max_chunk_chars,
        "chunk_max_chars": options.chunk_max_chars,
    }

    if options.mode == "naive":
        return extract_fields_naive(pdf_path, SCHEMA_PATH, **common_kwargs)

    if options.mode == "retrieval":
        return extract_fields_retrieval(
            pdf_path,
            SCHEMA_PATH,
            **common_kwargs,
            **retrieval_kwargs,
        )

    if options.mode == "field_agents":
        return extract_fields_field_agents(
            pdf_path,
            SCHEMA_PATH,
            **common_kwargs,
            **retrieval_kwargs,
        )

    return extract_fields_orchestrated(
        pdf_path,
        SCHEMA_PATH,
        **common_kwargs,
        **retrieval_kwargs,
        enable_verifier=options.enable_verifier,
        verifier_confidence_threshold=options.verifier_confidence_threshold,
        verifier_max_repairs=options.verifier_max_repairs,
        verifier_model=options.verifier_model,
    )


def _build_response_payload(
    *,
    run_id: str,
    file_name: str,
    result: ExtractionResult,
    options: UIExtractionOptions,
    latency_ms: float,
) -> Dict[str, Any]:
    retrieval_meta = result.retrieval or {"enabled": False}
    risk_meta: Dict[str, Any] = {}
    if isinstance(retrieval_meta, dict):
        maybe_risk = retrieval_meta.get("risk")
        if isinstance(maybe_risk, dict):
            risk_meta = maybe_risk

    raw_text = result.raw_text or ""
    raw_truncated = False
    if len(raw_text) > MAX_RAW_TEXT_CHARS:
        raw_text = raw_text[:MAX_RAW_TEXT_CHARS].rstrip() + "\n...[truncated]"
        raw_truncated = True

    prompt_tokens = result.prompt_tokens
    completion_tokens = result.completion_tokens
    total_tokens = None
    if isinstance(prompt_tokens, int) or isinstance(completion_tokens, int):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    return {
        "ok": True,
        "run_id": run_id,
        "file_name": file_name,
        "fields": result.json_result,
        "issues": result.issues or [],
        "meta": {
            "mode": options.mode,
            "retrieval_backend": options.retrieval_backend if options.mode != "naive" else None,
            "model": options.model,
            "latency_ms": latency_ms,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "risk": _build_risk_summary(result.json_result, risk_meta),
        "retrieval_trace": retrieval_meta,
        "raw_text": raw_text,
        "raw_text_truncated": raw_truncated,
    }


def _build_risk_summary(fields: Dict[str, Any], risk_meta: Dict[str, Any]) -> Dict[str, Any]:
    level = fields.get("risk_level")
    explanation = fields.get("risk_explanation")
    if not isinstance(risk_meta, dict) or not risk_meta:
        return {
            "available": False,
            "risk_level": level,
            "risk_explanation": explanation,
            "drivers": [],
            "protectors": [],
            "uncertainty": {},
            "hard_triggers": [],
            "arbitration": None,
            "orchestration": {},
        }

    factors = risk_meta.get("factors")
    factor_list = factors if isinstance(factors, list) else []
    drivers = sorted(
        [f for f in factor_list if isinstance(f, dict) and _safe_float(f.get("contribution")) > 0],
        key=lambda f: _safe_float(f.get("contribution")),
        reverse=True,
    )[:5]
    protectors = sorted(
        [f for f in factor_list if isinstance(f, dict) and _safe_float(f.get("contribution")) < 0],
        key=lambda f: _safe_float(f.get("contribution")),
    )[:5]

    orchestration = risk_meta.get("orchestration")
    return {
        "available": True,
        "risk_level": risk_meta.get("risk_level", level),
        "risk_explanation": risk_meta.get("risk_explanation", explanation),
        "confidence": risk_meta.get("confidence"),
        "score": risk_meta.get("score"),
        "rule_level": risk_meta.get("rule_level"),
        "rule_score": risk_meta.get("rule_score"),
        "arbitration": risk_meta.get("arbitration"),
        "hard_triggers": risk_meta.get("hard_triggers") if isinstance(risk_meta.get("hard_triggers"), list) else [],
        "uncertainty": risk_meta.get("uncertainty") if isinstance(risk_meta.get("uncertainty"), dict) else {},
        "drivers": drivers,
        "protectors": protectors,
        "orchestration": orchestration if isinstance(orchestration, dict) else {},
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


app = create_app()
