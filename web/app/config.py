import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _int(name, default):
    return int(os.environ.get(name, default))


HOST = os.environ.get("TANGUT_OCR_HOST", "0.0.0.0")
PORT = _int("TANGUT_OCR_PORT", 7777)
DATA_DIR = Path(os.environ.get("TANGUT_OCR_DATA", ROOT / "data"))
MODEL_DIR = Path(os.environ.get("TANGUT_OCR_MODELS", ROOT / "models"))
STATIC_DIR = ROOT / "static"

ADMIN_TOKEN = os.environ.get("TANGUT_OCR_ADMIN_TOKEN", "")

THREADS = _int("TANGUT_OCR_THREADS", 2)

MAX_UPLOAD_MB = _int("TANGUT_OCR_MAX_UPLOAD_MB", 120)
MAX_PDF_PAGES = _int("TANGUT_OCR_MAX_PDF_PAGES", 2000)
MAX_SIDE = _int("TANGUT_OCR_MAX_SIDE", 5000) 
MAX_QUEUE = _int("TANGUT_OCR_MAX_QUEUE", 20) 
MAX_JOBS_PER_IP = _int("TANGUT_OCR_MAX_JOBS_PER_IP", 3) 
UPLOADS_PER_HOUR = _int("TANGUT_OCR_UPLOADS_PER_HOUR", 60)
FEEDBACK_PER_HOUR = _int("TANGUT_OCR_FEEDBACK_PER_HOUR", 1200)

RETENTION_HOURS = _int("TANGUT_OCR_RETENTION_HOURS", 72)
