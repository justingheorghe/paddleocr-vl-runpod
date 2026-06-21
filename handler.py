import os
import uuid
import base64
import requests
from pathlib import Path

# -------------------------------------------------------------------
# Cache setup must happen before importing PaddleOCR
# -------------------------------------------------------------------

RUNPOD_VOLUME = Path("/runpod-volume")

if RUNPOD_VOLUME.exists():
    CACHE_ROOT = RUNPOD_VOLUME / "paddleocr-vl-cache"
else:
    CACHE_ROOT = Path("/tmp/paddleocr-vl-cache")

CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Hugging Face cache
os.environ["HF_HOME"] = str(CACHE_ROOT / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_ROOT / "huggingface" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "huggingface" / "transformers")

# Paddle / PaddleOCR cache
os.environ["PADDLE_HOME"] = str(CACHE_ROOT / "paddle")
os.environ["PADDLEOCR_HOME"] = str(CACHE_ROOT / "paddleocr")
os.environ["XDG_CACHE_HOME"] = str(CACHE_ROOT / "xdg-cache")

for cache_dir in [
    os.environ["HF_HOME"],
    os.environ["HUGGINGFACE_HUB_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["PADDLE_HOME"],
    os.environ["PADDLEOCR_HOME"],
    os.environ["XDG_CACHE_HOME"],
]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)


import runpod
from filelock import FileLock
from paddleocr import PaddleOCRVL


# -------------------------------------------------------------------
# Load the OCR pipeline once at worker startup
# -------------------------------------------------------------------

MODEL_LOCK_PATH = str(CACHE_ROOT / "model_init.lock")

with FileLock(MODEL_LOCK_PATH, timeout=900):
    print("Loading PaddleOCR-VL 1.6 pipeline...")
    pipeline = PaddleOCRVL(pipeline_version="v1.6")
    print("PaddleOCR-VL 1.6 pipeline loaded.")


def download_file(url: str, path: str):
    response = requests.get(url, timeout=180)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)


def save_base64_file(b64_data: str, path: str):
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]

    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_data))


def handler(job):
    job_input = job.get("input", {}) or {}

    image_url = job_input.get("image_url")
    file_base64 = job_input.get("file_base64")
    filename = job_input.get("filename", "input.png")

    if not image_url and not file_base64:
        return {
            "status": "error",
            "error": "Provide either input.image_url or input.file_base64"
        }

    request_id = str(uuid.uuid4())
    workdir = Path("/tmp") / request_id
    output_dir = workdir / "output"

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = workdir / filename

    try:
        if image_url:
            download_file(image_url, str(input_path))
        else:
            save_base64_file(file_base64, str(input_path))

        results = pipeline.predict(str(input_path))

        response_pages = []

        for page_index, res in enumerate(results):
            page_dir = output_dir / f"page_{page_index}"
            page_dir.mkdir(parents=True, exist_ok=True)

            res.save_to_json(save_path=str(page_dir))
            res.save_to_markdown(save_path=str(page_dir))

            page_item = {
                "page_index": page_index,
            }

            for file in page_dir.iterdir():
                if file.suffix == ".md":
                    page_item["markdown"] = file.read_text(encoding="utf-8")

                if file.suffix == ".json":
                    page_item["json"] = file.read_text(encoding="utf-8")

            response_pages.append(page_item)

        return {
            "status": "success",
            "pages": response_pages,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


runpod.serverless.start({"handler": handler})