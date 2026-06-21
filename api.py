import os
import uuid
import json
import base64
import shutil
from pathlib import Path
from typing import Optional

import requests
import paddle
from fastapi import FastAPI
from pydantic import BaseModel
from filelock import FileLock
from paddleocr import PaddleOCRVL


# -----------------------------
# Cache setup
# -----------------------------

RUNPOD_VOLUME = Path("/runpod-volume")
WORKSPACE = Path("/workspace")

if RUNPOD_VOLUME.exists() and os.access(str(RUNPOD_VOLUME), os.W_OK):
    CACHE_ROOT = RUNPOD_VOLUME / "paddleocr-vl-cache"
elif WORKSPACE.exists() and os.access(str(WORKSPACE), os.W_OK):
    CACHE_ROOT = WORKSPACE / "paddleocr-vl-cache"
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


# -----------------------------
# Force GPU
# -----------------------------

paddle.set_device("gpu:0")
print("PADDLE DEVICE:", paddle.device.get_device())


# -----------------------------
# Load pipeline once
# -----------------------------

MODEL_LOCK_PATH = str(CACHE_ROOT / "model_init.lock")

with FileLock(MODEL_LOCK_PATH, timeout=1800):
    print("Loading PaddleOCR-VL full parsing pipeline...")
    pipeline = PaddleOCRVL(pipeline_version="v1.6")
    print("PaddleOCR-VL full parsing pipeline loaded.")


app = FastAPI()


class ParseRequest(BaseModel):
    file_url: Optional[str] = None
    pdf_url: Optional[str] = None
    image_url: Optional[str] = None
    file_base64: Optional[str] = None
    filename: str = "input.pdf"

    return_markdown: bool = True
    return_json: bool = True

    restructure_pages: bool = True
    merge_tables: bool = True
    relevel_titles: bool = True
    concatenate_pages: bool = False

    # Speed knobs
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_chart_recognition: bool = False
    use_seal_recognition: bool = False
    use_ocr_for_image_block: bool = False
    max_new_tokens: int = 2048
    max_pixels: Optional[int] = 1280000


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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": paddle.device.get_device(),
        "model": "PaddleOCR-VL-1.6",
    }


@app.post("/parse")
def parse(req: ParseRequest):
    file_url = req.file_url or req.pdf_url or req.image_url

    if not file_url and not req.file_base64:
        return {
            "status": "error",
            "error": "Provide file_url, pdf_url, image_url, or file_base64"
        }

    request_id = str(uuid.uuid4())
    workdir = Path("/tmp") / f"paddleocr-{request_id}"
    output_dir = workdir / "output"

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = workdir / req.filename

    try:
        if file_url:
            download_file(file_url, str(input_path))
        else:
            save_base64_file(req.file_base64, str(input_path))

        predict_kwargs = {
            "input": str(input_path),
            "use_doc_orientation_classify": req.use_doc_orientation_classify,
            "use_doc_unwarping": req.use_doc_unwarping,
            "use_chart_recognition": req.use_chart_recognition,
            "use_seal_recognition": req.use_seal_recognition,
            "use_ocr_for_image_block": req.use_ocr_for_image_block,
            "max_new_tokens": req.max_new_tokens,
            "max_pixels": req.max_pixels,
        }

        predict_kwargs = {k: v for k, v in predict_kwargs.items() if v is not None}

        raw_output = pipeline.predict(**predict_kwargs)
        pages_res = list(raw_output)

        if req.restructure_pages:
            final_output = pipeline.restructure_pages(
                pages_res,
                merge_tables=req.merge_tables,
                relevel_titles=req.relevel_titles,
                concatenate_pages=req.concatenate_pages,
            )
        else:
            final_output = pages_res

        for res in final_output:
            if req.return_markdown:
                res.save_to_markdown(save_path=str(output_dir))

            if req.return_json:
                res.save_to_json(save_path=str(output_dir))

        markdown = read_markdown_files(output_dir) if req.return_markdown else None
        json_data = read_json_files(output_dir) if req.return_json else None

        return {
            "status": "success",
            "filename": req.filename,
            "markdown": markdown,
            "json": json_data,
            "metadata": {
                "restructure_pages": req.restructure_pages,
                "merge_tables": req.merge_tables,
                "relevel_titles": req.relevel_titles,
                "concatenate_pages": req.concatenate_pages,
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