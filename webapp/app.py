#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import ssl
import subprocess
import threading
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import urllib.request
from urllib.error import HTTPError

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for
from PIL import Image
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parents[1]
WEBAPP_DIR = Path(__file__).resolve().parent
JOBS_DIR = Path(os.environ.get("JOBS_DIR", str(WEBAPP_DIR / "jobs"))).resolve()
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(JOBS_DIR.parent / "cache"))).resolve()
PIPELINE_SCRIPT = BASE_DIR / "pipeline.py"
WAAMD_SCRIPT = BASE_DIR / "add_waamd_author_ids.py"
WAAMD_CSV = BASE_DIR / "waamd.csv"
PROMPTS_DIR = BASE_DIR / "prompts"
ALLOWED_EXTENSIONS = {"pdf"}
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
THUMBNAIL_WIDTH = 280
PIPELINE_FIELDS_PER_PAGE = 15
PIPELINE_START_PROGRESS = 0
PIPELINE_END_PROGRESS = 92
WAAMD_START_PROGRESS = 94
WAAMD_END_PROGRESS = 99
EDITABLE_PROMPTS = [
    "collectionno.txt",
    "title.txt",
    "author.txt",
    "authordeath.txt",
    "catalog_header_page_prompt.txt",
    "dimensions_condition_prompt.txt",
    "firstline.txt",
    "lastline.txt",
    "copyist.txt",
    "datecopied.txt",
    "form.txt",
    "subject.txt",
]

app = Flask(__name__, template_folder=str(WEBAPP_DIR / "templates"), static_folder=str(WEBAPP_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
JOBS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

JOB_LOCK = threading.Lock()
JOB_STATE = {}
PROCESS_LOCK = threading.Lock()
ACTIVE_PROCESSES = {}
RUNNING_JOB_GRACE_PERIOD = timedelta(seconds=15)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_anthropic_api_key(api_key: str) -> tuple[bool, str]:
    if not api_key:
        return False, "Anthropic API key is required for each run."

    body = json.dumps(
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
        },
        ensure_ascii=False,
    ).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=body,
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as resp:
            if 200 <= resp.status < 300:
                return True, ""
            return False, "Anthropic rejected the API key. Enter a valid key and try again."
    except HTTPError as exc:
        if exc.code in (401, 403):
            return False, "Anthropic rejected the API key. Enter a valid key and try again."
        details = ""
        if exc.fp:
            details = exc.read().decode("utf-8", errors="ignore")[:300]
        return False, f"Could not validate the API key right now (HTTP {exc.code}). {details}".strip()
    except Exception as exc:
        return False, f"Could not validate the API key right now. {exc}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def parse_iso_z(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def update_job(job_id: str, **fields) -> None:
    with JOB_LOCK:
        state = JOB_STATE.setdefault(job_id, {})
        state.update(fields)
        state["updated_at"] = now_iso()
        meta_path = JOBS_DIR / job_id / "job.json"
        disk_state = read_json(meta_path) if meta_path.exists() else {}
        disk_state.update(state)
        write_json(meta_path, disk_state)


def stage_progress(stage: str, status: str) -> int:
    if status in {"completed", "failed", "stopped"}:
        return 100
    mapping = {
        "conversion": 5,
        "select_range": 0,
        "pipeline": 0,
        "waamd": WAAMD_START_PROGRESS,
        "done": 100,
        "stopping": 99,
        "error": 100,
        "stopped": 100,
    }
    return mapping.get(stage or "", 5)


def parse_pipeline_progress(line: str, state: dict) -> tuple[Optional[int], Optional[str]]:
    if not line:
        return None, None
    raw_line = line.rstrip()
    page_match = re.match(r"^\[(\d+)\/(\d+)\]\s+\S+\.(?:jpg|jpeg|png)\b", raw_line, re.IGNORECASE)
    if page_match:
        state["page_index"] = int(page_match.group(1))
        state["page_total"] = int(page_match.group(2))
        detail = f"Page {state['page_index']} of {state['page_total']}"
        return _pipeline_percent(state["page_index"] - 1, state["page_total"]), detail

    step_match = re.match(r"^\s*\[(\d+)([a-z]?)\/15\]", raw_line, re.IGNORECASE)
    if not step_match:
        return None, None
    page_index = state.get("page_index")
    page_total = state.get("page_total")
    if not page_index or not page_total:
        return None, None
    step_num = int(step_match.group(1))
    suffix = (step_match.group(2) or "").lower()
    field_order = {
        (1, ""): 1,
        (2, ""): 2,
        (3, ""): 3,
        (3, "b"): 4,
        (3, "c"): 5,
        (3, "d"): 6,
        (4, ""): 7,
        (5, ""): 8,
        (6, ""): 9,
        (7, ""): 10,
        (8, ""): 11,
        (8, "b"): 12,
        (9, ""): 13,
        (10, ""): 14,
        (10, "b"): 14,
        (11, ""): 15,
    }
    field_index = field_order.get((step_num, suffix))
    if field_index is None:
        return None, None
    units_done = ((page_index - 1) * PIPELINE_FIELDS_PER_PAGE) + field_index
    total_units = page_total * PIPELINE_FIELDS_PER_PAGE
    step_label = f"{step_num}{suffix}" if suffix else str(step_num)
    detail = f"Page {page_index} of {page_total} · Step {step_label} of 15"
    return _pipeline_percent(units_done, total_units), detail


def _pipeline_percent(units_done: int, total_units: int) -> int:
    if total_units <= 0:
        return PIPELINE_START_PROGRESS
    span = PIPELINE_END_PROGRESS - PIPELINE_START_PROGRESS
    raw = PIPELINE_START_PROGRESS + int((units_done / total_units) * span)
    return min(PIPELINE_END_PROGRESS, max(PIPELINE_START_PROGRESS, raw))


def parse_waamd_progress(line: str, state: dict) -> tuple[Optional[int], Optional[str]]:
    raw_line = (line or "").rstrip()
    if not raw_line:
        return None, None
    if raw_line.startswith("Author column:"):
        return WAAMD_START_PROGRESS, "Matching authors against WAAMD"
    if raw_line.startswith("Rows processed:"):
        m = re.search(r"(\d+)", raw_line)
        if m:
            state["rows_total"] = int(m.group(1))
        return WAAMD_START_PROGRESS + 2, raw_line
    if raw_line.startswith("WAAMD IDs matched:"):
        return WAAMD_START_PROGRESS + 4, raw_line
    if raw_line.startswith("Saved:"):
        return WAAMD_END_PROGRESS, "WAAMD matching finished"
    return None, None


def get_job(job_id: str) -> dict:
    meta_path = JOBS_DIR / job_id / "job.json"
    if not meta_path.exists():
        raise FileNotFoundError(job_id)
    disk_state = read_json(meta_path)
    with JOB_LOCK:
        mem_state = JOB_STATE.get(job_id, {})
    if not disk_state and not mem_state:
        raise FileNotFoundError(job_id)
    if mem_state:
        disk_state.update(mem_state)
    if disk_state.get("status") == "running":
        if disk_state.get("stage") == "done":
            disk_state["status"] = "completed"
            disk_state["stop_available"] = False
            disk_state["progress"] = 100
            disk_state["progress_detail"] = disk_state.get("progress_detail") or "Run finished"
            if disk_state.get("error") == "Worker state was lost. The server likely restarted or the worker exited unexpectedly.":
                disk_state["error"] = ""
            update_job(
                job_id,
                status="completed",
                stop_available=False,
                progress=100,
                progress_detail=disk_state["progress_detail"],
                error=disk_state.get("error", ""),
            )
            return disk_state
        with PROCESS_LOCK:
            proc = ACTIVE_PROCESSES.get(job_id)
        updated_at = parse_iso_z(disk_state.get("updated_at", ""))
        grace_deadline = (updated_at + RUNNING_JOB_GRACE_PERIOD) if updated_at else None
        if proc is None and grace_deadline and datetime.now(timezone.utc) >= grace_deadline:
            append_job_log(
                JOBS_DIR / job_id,
                "[job] Worker state lost; marking run as failed. This usually means the server restarted or the worker exited unexpectedly.",
            )
            update_job(
                job_id,
                status="failed",
                stage="error",
                error="Worker state was lost. The server likely restarted or the worker exited unexpectedly.",
                live_log=read_log_tail(JOBS_DIR / job_id),
                progress=100,
                stop_available=False,
            )
            disk_state = read_json(meta_path)
    return disk_state


def list_jobs() -> list[dict]:
    jobs = []
    for meta_path in JOBS_DIR.glob("*/job.json"):
        try:
            job = get_job(meta_path.parent.name)
            job.setdefault("id", meta_path.parent.name)
            if not job.get("pdf_name"):
                job["pdf_name"] = "Untitled Upload"
            if not job.get("stage"):
                job["stage"] = "unknown"
            if not job.get("status"):
                job["status"] = "ready"
            jobs.append(job)
        except Exception:
            continue
    jobs.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
    return jobs


def run_command(cmd, cwd: Path, env=None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def append_job_log(job_dir: Path, text: str) -> None:
    if not text:
        return
    log_path = job_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def read_log_tail(job_dir: Path, max_lines: int = 120) -> str:
    log_path = job_dir / "run.log"
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-max_lines:])


def run_streaming_command(cmd, cwd: Path, job_id: str, stage: str, env=None) -> str:
    job_dir = JOBS_DIR / job_id
    append_job_log(job_dir, "$ " + " ".join(str(x) for x in cmd))
    progress_state = {"page_index": 0, "page_total": 0}
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    with PROCESS_LOCK:
        ACTIVE_PROCESSES[job_id] = proc
    captured = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            captured.append(line)
            append_job_log(job_dir, line.rstrip("\n"))
            parsed_progress = None
            progress_detail = None
            if stage == "pipeline":
                parsed_progress, progress_detail = parse_pipeline_progress(line, progress_state)
            elif stage == "waamd":
                parsed_progress, progress_detail = parse_waamd_progress(line, progress_state)
            with JOB_LOCK:
                current_progress = (JOB_STATE.get(job_id, {}) or {}).get("progress")
            update_job(
                job_id,
                stage=stage,
                live_log=read_log_tail(job_dir),
                progress=max(int(current_progress or 0), parsed_progress) if parsed_progress is not None else (current_progress if current_progress is not None else stage_progress(stage, "running")),
                progress_detail=progress_detail if progress_detail is not None else (JOB_STATE.get(job_id, {}) or {}).get("progress_detail", ""),
                current_output=line.rstrip("\n"),
            )
        return_code = proc.wait()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        with PROCESS_LOCK:
            ACTIVE_PROCESSES.pop(job_id, None)
    if return_code != 0:
        raise RuntimeError(f"Command failed ({return_code}): {' '.join(str(x) for x in cmd)}")
    return "".join(captured)


def get_pdf_page_count(pdf_path: Path) -> int:
    out = run_command(["pdfinfo", str(pdf_path)], cwd=pdf_path.parent)
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError("Could not determine PDF page count")


def cache_dir_for_pdf(pdf_hash: str) -> Path:
    return CACHE_DIR / pdf_hash


def cache_images_dir(pdf_hash: str) -> Path:
    return cache_dir_for_pdf(pdf_hash) / "images"


def cache_meta_path(pdf_hash: str) -> Path:
    return cache_dir_for_pdf(pdf_hash) / "meta.json"


def load_cached_conversion(pdf_hash: str) -> dict:
    return read_json(cache_meta_path(pdf_hash))


def save_cached_conversion(pdf_hash: str, *, page_count: int, stem: str) -> None:
    write_json(
        cache_meta_path(pdf_hash),
        {
            "pdf_hash": pdf_hash,
            "page_count": page_count,
            "stem": stem,
            "cached_at": now_iso(),
        },
    )


def populate_job_images_from_cache(pdf_hash: str, target_dir: Path) -> int:
    source_dir = cache_images_dir(pdf_hash)
    if not source_dir.exists():
        raise RuntimeError("Cached image directory is missing")
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(source_dir.glob("*.jpg")):
        shutil.copy2(src, target_dir / src.name)
        count += 1
    if count == 0:
        raise RuntimeError("Cached image directory is empty")
    return count


def cache_converted_images(pdf_hash: str, source_dir: Path) -> int:
    target_dir = cache_images_dir(pdf_hash)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(source_dir.glob("*.jpg")):
        shutil.copy2(src, target_dir / src.name)
        count += 1
    if count == 0:
        raise RuntimeError("No converted images were available to cache")
    return count


def convert_pdf_to_images(pdf_path: Path, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / stem
    run_command(
        ["pdftoppm", "-jpeg", "-r", "220", str(pdf_path), str(prefix)],
        cwd=output_dir,
    )
    converted = sorted(output_dir.glob(f"{stem}-*.jpg"))
    renamed = []
    for src in converted:
        page_no = int(src.stem.rsplit("-", 1)[1])
        dst = output_dir / f"{stem}_page_{page_no:03d}.jpg"
        src.rename(dst)
        renamed.append(dst)
    if not renamed:
        raise RuntimeError("PDF conversion produced no images")
    return renamed


def create_thumbnail(image_path: Path) -> Path:
    thumb_path = image_path.parent / f"{image_path.stem}_thumb.jpg"
    with Image.open(image_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        width, height = img.size
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid image dimensions for thumbnail: {image_path}")
        thumb_height = max(1, int((THUMBNAIL_WIDTH / width) * height))
        resized = img.resize((THUMBNAIL_WIDTH, thumb_height), Image.Resampling.LANCZOS)
        resized.save(thumb_path, format="JPEG", quality=85, optimize=True)
    return thumb_path


def build_page_items(job_id: str) -> list[dict]:
    job_dir = JOBS_DIR / job_id
    image_dir = job_dir / "images"
    thumbs_dir = job_dir / "thumbs"
    thumbs_dir.mkdir(exist_ok=True)
    items = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        thumb_path = thumbs_dir / image_path.name
        if not thumb_path.exists():
            generated = create_thumbnail(image_path)
            generated.replace(thumb_path)
        page_num = int(image_path.stem.rsplit("_", 1)[1])
        items.append(
            {
                "page_num": page_num,
                "image_name": image_path.name,
                "thumb_name": thumb_path.name,
            }
        )
    return items


def find_latest(path_glob: str, cwd: Path) -> Optional[Path]:
    matches = sorted(cwd.glob(path_glob), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def load_base_prompts() -> list:
    prompts = []
    for name in EDITABLE_PROMPTS:
        path = PROMPTS_DIR / name
        prompts.append({
            "key": name,
            "label": name,
            "content": path.read_text(encoding="utf-8"),
        })
    return prompts


def collect_prompt_overrides(form_data) -> dict:
    overrides = {}
    enabled = form_data.get("use_custom_prompts") == "on"
    if not enabled:
        return overrides
    for name in EDITABLE_PROMPTS:
        field = f"prompt__{name}"
        if field in form_data:
            overrides[name] = form_data.get(field, "")
    return overrides


def build_job_workspace(job_id: str, prompt_overrides: dict) -> Path:
    job_dir = JOBS_DIR / job_id
    workspace = job_dir / "workspace"
    prompts_target = workspace / "prompts"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PIPELINE_SCRIPT, workspace / "pipeline.py")
    shutil.copy2(WAAMD_SCRIPT, workspace / "add_waamd_author_ids.py")
    shutil.copy2(WAAMD_CSV, workspace / "waamd.csv")
    shutil.copytree(PROMPTS_DIR, prompts_target)
    for name, content in prompt_overrides.items():
        (prompts_target / name).write_text(content, encoding="utf-8")
    return workspace


def run_pipeline_job(job_id: str, start_page: int, end_page: int, prompt_overrides: dict, api_key: str) -> None:
    job_dir = JOBS_DIR / job_id
    image_dir = job_dir / "images"
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = api_key
    env.pop("CLAUDE_API_KEY", None)
    try:
        append_job_log(job_dir, f"[job] Started at {now_iso()}")
        update_job(
            job_id,
            status="running",
            stage="pipeline",
            range_start=start_page,
            range_end=end_page,
            live_log=read_log_tail(job_dir),
            progress=stage_progress("pipeline", "running"),
            progress_detail="Waiting for pipeline output...",
            stop_available=True,
        )
        workspace = build_job_workspace(job_id, prompt_overrides)
        # Per-page cache: keyed by PDF hash so the same page is never re-billed across jobs.
        job_meta = read_json(job_dir / "job.json")
        pdf_hash = job_meta.get("pdf_hash", "")
        page_cache_dir = (CACHE_DIR / pdf_hash / "page_results") if pdf_hash else None
        if page_cache_dir:
            page_cache_dir.mkdir(parents=True, exist_ok=True)
        pipeline_cmd = [
            "python3",
            str(workspace / "pipeline.py"),
            "--range",
            str(start_page),
            str(end_page),
            "--folder",
            str(image_dir),
        ]
        if page_cache_dir:
            pipeline_cmd += ["--page-cache-dir", str(page_cache_dir)]
        env["PYTHONUNBUFFERED"] = "1"
        run_streaming_command(pipeline_cmd, cwd=job_dir, job_id=job_id, stage="pipeline", env=env)

        pipeline_csv = find_latest("pipeline_responses_*.csv", job_dir)
        if pipeline_csv is None:
            raise RuntimeError("Pipeline finished but did not produce a CSV")

        update_job(
            job_id,
            stage="waamd",
            pipeline_csv=pipeline_csv.name,
            live_log=read_log_tail(job_dir),
            progress=stage_progress("waamd", "running"),
            progress_detail="Matching authors against WAAMD",
        )
        waamd_output = job_dir / f"{pipeline_csv.stem}_with_waamd_ids.csv"
        waamd_cmd = [
            "python3",
            str(workspace / "add_waamd_author_ids.py"),
            str(pipeline_csv),
            "--output",
            str(waamd_output),
            "--waamd",
            str(workspace / "waamd.csv"),
        ]
        run_streaming_command(waamd_cmd, cwd=job_dir, job_id=job_id, stage="waamd", env=env)

        matched = ""
        total_rows = ""
        with waamd_output.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        total_rows = len(rows)
        matched = sum(1 for row in rows if (row.get("WAAMD Author ID") or "").strip())
        append_job_log(job_dir, f"[job] Completed at {now_iso()}")
        update_job(
            job_id,
            status="completed",
            stage="done",
            output_csv=waamd_output.name,
            raw_pipeline_csv=pipeline_csv.name,
            total_rows=total_rows,
            matched_rows=matched,
            live_log=read_log_tail(job_dir),
            progress=100,
            progress_detail="Run finished",
            current_output="Results ready for download",
            error="",
            stop_available=False,
        )
    except RuntimeError as exc:
        if "Command failed (-15)" in str(exc) or "Command failed (-9)" in str(exc):
            append_job_log(job_dir, f"[job] Stopped at {now_iso()}")
            update_job(
                job_id,
                status="stopped",
                stage="stopped",
                error="Run stopped by user",
                live_log=read_log_tail(job_dir),
                progress=100,
                progress_detail="Run stopped",
                stop_available=False,
            )
            return
        append_job_log(job_dir, f"ERROR: {exc}")
        update_job(job_id, status="failed", stage="error", error=str(exc), live_log=read_log_tail(job_dir), progress=100, progress_detail="Run failed", stop_available=False)
    except Exception as exc:
        append_job_log(job_dir, f"ERROR: {exc}")
        update_job(job_id, status="failed", stage="error", error=str(exc), live_log=read_log_tail(job_dir), progress=100, progress_detail="Run failed", stop_available=False)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", jobs=list_jobs())


def run_conversion_job(job_id: str, pdf_path: Path, stem: str) -> None:
    """Background thread: convert PDF to images and update job state."""
    job_dir = JOBS_DIR / job_id
    image_dir = job_dir / "images"
    try:
        pdf_hash = sha256_file(pdf_path)
        update_job(job_id, progress_detail="Checking conversion cache...")
        cached = load_cached_conversion(pdf_hash)
        page_count = cached.get("page_count")
        cache_hit = False
        if cached and isinstance(page_count, int) and page_count > 0 and cache_images_dir(pdf_hash).exists():
            update_job(job_id, progress_detail="Restoring pages from cache...")
            populate_job_images_from_cache(pdf_hash, image_dir)
            cache_hit = True
        else:
            update_job(job_id, progress_detail="Counting pages...")
            page_count = get_pdf_page_count(pdf_path)
            update_job(job_id, progress_detail=f"Converting {page_count} pages to images (this may take a few minutes)...")
            convert_pdf_to_images(pdf_path, image_dir, stem)
            update_job(job_id, progress_detail="Saving pages to cache...")
            cache_converted_images(pdf_hash, image_dir)
            save_cached_conversion(pdf_hash, page_count=page_count, stem=stem)
        update_job(job_id, progress_detail="Building page previews...")
        page_items = build_page_items(job_id)
        update_job(
            job_id,
            status="ready",
            stage="select_range",
            page_count=page_count,
            page_items=page_items,
            prompt_overrides={},
            editable_prompts=load_base_prompts(),
            progress=stage_progress("select_range", "ready"),
            progress_detail="Reused cached page images" if cache_hit else "Converted PDF and saved page images to cache",
            cache_hit=cache_hit,
            pdf_hash=pdf_hash,
        )
    except Exception as exc:
        update_job(job_id, status="failed", stage="error", error=str(exc), progress=100, progress_detail="Conversion failed")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf")
    if not file or not file.filename:
        abort(400, "No PDF uploaded")
    if not allowed_file(file.filename):
        abort(400, "Only PDF uploads are supported")

    job_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    uploads_dir = job_dir / "uploads"
    image_dir = job_dir / "images"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(file.filename) or "upload.pdf"
    pdf_path = uploads_dir / safe_name
    file.save(pdf_path)

    update_job(
        job_id,
        id=job_id,
        status="preparing",
        stage="conversion",
        pdf_name=safe_name,
        created_at=now_iso(),
        progress=stage_progress("conversion", "preparing"),
        progress_detail="Starting PDF conversion...",
        stop_available=False,
    )

    thread = threading.Thread(
        target=run_conversion_job,
        args=(job_id, pdf_path, pdf_path.stem),
        daemon=True,
    )
    thread.start()
    return redirect(url_for("job_view", job_id=job_id))


@app.route("/jobs/<job_id>", methods=["GET"])
def job_view(job_id: str):
    try:
        job = get_job(job_id)
    except FileNotFoundError:
        abort(404)
    return render_template("job.html", job=job)


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})


@app.route("/jobs/<job_id>/run", methods=["POST"])
def run_job(job_id: str):
    try:
        job = get_job(job_id)
    except FileNotFoundError:
        abort(404)

    start_page = int(request.form.get("start_page", "0"))
    end_page = int(request.form.get("end_page", "0"))
    if start_page < 1 or end_page < 1 or end_page < start_page:
        abort(400, "Invalid page range")
    if end_page > int(job.get("page_count", 0)):
        abort(400, "Page range exceeds PDF length")
    if job.get("status") == "running":
        return redirect(url_for("job_view", job_id=job_id))
    api_key = (request.form.get("anthropic_api_key") or "").strip()
    if not api_key:
        abort(400, "Anthropic API key is required for each run")
    key_ok, key_message = validate_anthropic_api_key(api_key)
    if not key_ok:
        prompt_overrides = collect_prompt_overrides(request.form)
        editable_prompts = []
        base_prompts = load_base_prompts()
        for prompt in base_prompts:
            editable_prompts.append({
                "key": prompt["key"],
                "label": prompt["label"],
                "content": prompt_overrides.get(prompt["key"], prompt["content"]),
            })
        update_job(
            job_id,
            status="ready",
            stage="select_range",
            prompt_overrides=prompt_overrides,
            editable_prompts=editable_prompts,
            custom_prompts_enabled=bool(prompt_overrides),
            error=key_message,
            current_output="API key validation failed",
            progress=stage_progress("select_range", "ready"),
            progress_detail="Enter a valid Anthropic API key to start the run",
            stop_available=False,
        )
        return redirect(url_for("job_view", job_id=job_id))

    prompt_overrides = collect_prompt_overrides(request.form)
    editable_prompts = []
    base_prompts = load_base_prompts()
    for prompt in base_prompts:
        editable_prompts.append({
            "key": prompt["key"],
            "label": prompt["label"],
            "content": prompt_overrides.get(prompt["key"], prompt["content"]),
        })
    update_job(
        job_id,
        prompt_overrides=prompt_overrides,
        editable_prompts=editable_prompts,
        custom_prompts_enabled=bool(prompt_overrides),
        error="",
        current_output="Starting pipeline...",
    )
    thread = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, start_page, end_page, prompt_overrides, api_key),
        daemon=True,
    )
    thread.start()
    return redirect(url_for("job_view", job_id=job_id))


@app.route("/jobs/<job_id>/stop", methods=["POST"])
def stop_job(job_id: str):
    try:
        job = get_job(job_id)
    except FileNotFoundError:
        abort(404)
    if job.get("status") != "running":
        return redirect(url_for("job_view", job_id=job_id))
    with PROCESS_LOCK:
        proc = ACTIVE_PROCESSES.get(job_id)
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    append_job_log(JOBS_DIR / job_id, "[job] Stop requested by user")
    update_job(
        job_id,
        stage="stopping",
        current_output="Stopping process...",
        live_log=read_log_tail(JOBS_DIR / job_id),
        progress=stage_progress("stopping", "running"),
        stop_available=False,
    )
    return redirect(url_for("job_view", job_id=job_id))


@app.route("/jobs/<job_id>/status", methods=["GET"])
def job_status(job_id: str):
    try:
        return jsonify(get_job(job_id))
    except FileNotFoundError:
        abort(404)


@app.route("/jobs/<job_id>/files/<path:filename>", methods=["GET"])
def download_job_file(job_id: str, filename: str):
    job_dir = JOBS_DIR / job_id
    path = (job_dir / filename).resolve()
    if not str(path).startswith(str(job_dir.resolve())) or not path.exists():
        abort(404)
    return send_file(path, as_attachment=True)


@app.route("/jobs/<job_id>/thumbs/<path:filename>", methods=["GET"])
def job_thumb(job_id: str, filename: str):
    path = (JOBS_DIR / job_id / "thumbs" / filename).resolve()
    if not str(path).startswith(str((JOBS_DIR / job_id).resolve())) or not path.exists():
        abort(404)
    return send_file(path)


@app.route("/jobs/<job_id>/log", methods=["GET"])
def job_log(job_id: str):
    path = JOBS_DIR / job_id / "run.log"
    if not path.exists():
        abort(404)
    return send_file(path)


@app.route("/jobs/<job_id>/download-all", methods=["GET"])
def download_job_bundle(job_id: str):
    try:
        job = get_job(job_id)
    except FileNotFoundError:
        abort(404)

    job_dir = JOBS_DIR / job_id
    bundle_path = job_dir / f"{job_id}_results_bundle.zip"
    wanted = []
    for key in ("output_csv", "raw_pipeline_csv"):
        name = job.get(key)
        if name:
            wanted.append(job_dir / name)
    log_path = job_dir / "run.log"
    if log_path.exists():
        wanted.append(log_path)

    if not wanted:
        abort(404)

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in wanted:
            if path.exists():
                zf.write(path, arcname=path.name)

    return send_file(bundle_path, as_attachment=True, download_name=f"{job_id}_results_bundle.zip")


@app.route("/jobs/<job_id>/delete", methods=["POST"])
def delete_job(job_id: str):
    job_dir = JOBS_DIR / job_id
    with PROCESS_LOCK:
        proc = ACTIVE_PROCESSES.pop(job_id, None)
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    with JOB_LOCK:
        JOB_STATE.pop(job_id, None)
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    return redirect(url_for("index"))


if __name__ == "__main__":
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
