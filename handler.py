import os
import uuid
import json
import base64
import shutil
from pathlib import Path

import requests
import runpod
from filelock import FileLock
from paddleocr import PaddleOCRVL


# -------------------------------------------------------
# Persistent cache
# -------------------------------------------------------

RUNPOD_VOLUME = Path("/runpod-volume")

if RUNPOD_VOLUME.exists():
    CACHE_ROOT = RUNPOD_VOLUME / "paddleocr-vl-cache"
else:
    CACHE_ROOT = Path("/tmp/paddleocr-vl-cache")

CACHE_ROOT.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_ROOT / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_ROOT / "huggingface" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "huggingface" / "transformers")
os.environ["PADDLE_HOME"] = str(CACHE_ROOT / "paddle")
os.environ["PADDLEOCR_HOME"] = str(CACHE_ROOT / "paddleocr")
os.environ["XDG_CACHE_HOME"] = str(CACHE_ROOT / "xdg-cache")

for key in [
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "PADDLE_HOME",
    "PADDLEOCR_HOME",
    "XDG_CACHE_HOME",
]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# Load full PaddleOCR-VL pipeline once per worker
# -------------------------------------------------------

MODEL_LOCK_PATH = str(CACHE_ROOT / "model_init.lock")

with FileLock(MODEL_LOCK_PATH, timeout=1800):
    print("Loading PaddleOCR-VL full parsing pipeline...")
    pipeline = PaddleOCRVL(pipeline_version="v1.6")
    print("PaddleOCR-VL full parsing pipeline loaded.")


def download_file(url: str, path: str):
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def save_base64_file(b64_data: str, path: str):
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]

    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_data))


def read_markdown_files(output_dir: Path):
    md_files = sorted(output_dir.rglob("*.md"))
    parts = []

    for idx, file in enumerate(md_files, start=1):
        text = file.read_text(encoding="utf-8", errors="ignore")
        parts.append(f"\n\n<!-- page_or_part {idx}: {file.name} -->\n\n{text}")

    return "\n".join(parts).strip()


def read_json_files(output_dir: Path):
    json_files = sorted(output_dir.rglob("*.json"))
    items = []

    for file in json_files:
        raw = file.read_text(encoding="utf-8", errors="ignore")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = raw

        items.append({
            "filename": file.name,
            "data": parsed,
        })

    return items


def handler(job):
    job_input = job.get("input", {}) or {}

    file_url = (
        job_input.get("file_url")
        or job_input.get("pdf_url")
        or job_input.get("image_url")
    )

    file_base64 = job_input.get("file_base64")
    filename = job_input.get("filename", "input.pdf")

    return_markdown = bool(job_input.get("return_markdown", True))
    return_json = bool(job_input.get("return_json", True))

    restructure_pages = bool(job_input.get("restructure_pages", True))
    merge_tables = bool(job_input.get("merge_tables", True))
    relevel_titles = bool(job_input.get("relevel_titles", True))
    concatenate_pages = bool(job_input.get("concatenate_pages", False))

    if not file_url and not file_base64:
        return {
            "status": "error",
            "error": "Provide input.file_url, input.pdf_url, input.image_url, or input.file_base64"
        }

    request_id = str(uuid.uuid4())
    workdir = Path("/tmp") / f"paddleocr-{request_id}"
    output_dir = workdir / "output"

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = workdir / filename

    try:
        if file_url:
            download_file(file_url, str(input_path))
        else:
            save_base64_file(file_base64, str(input_path))

        # Full PaddleOCR-VL pipeline.
        # Supports image paths and PDF paths.
        raw_output = pipeline.predict(input=str(input_path))
        pages_res = list(raw_output)

        if restructure_pages:
            final_output = pipeline.restructure_pages(
                pages_res,
                merge_tables=merge_tables,
                relevel_titles=relevel_titles,
                concatenate_pages=concatenate_pages,
            )
        else:
            final_output = pages_res

        for res in final_output:
            if return_markdown:
                res.save_to_markdown(save_path=str(output_dir))

            if return_json:
                res.save_to_json(save_path=str(output_dir))

        markdown = read_markdown_files(output_dir) if return_markdown else None
        json_data = read_json_files(output_dir) if return_json else None

        return {
            "status": "success",
            "filename": filename,
            "markdown": markdown,
            "json": json_data,
            "metadata": {
                "restructure_pages": restructure_pages,
                "merge_tables": merge_tables,
                "relevel_titles": relevel_titles,
                "concatenate_pages": concatenate_pages,
                "output_file_count": len(list(output_dir.rglob("*"))),
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


runpod.serverless.start({"handler": handler})