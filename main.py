import os
import tempfile
import logging
from pathlib import Path

import requests
import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STORAGE_BASE_URL = os.getenv("STORAGE_BASE_URL", "https://files.showdepremios.cloud")

app = FastAPI(
    title="QR Code Reader API",
    description="Lê QR codes de cartelas a partir de um image_id.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# QR decoding helpers
# ---------------------------------------------------------------------------

def _decode_from_array(arr: np.ndarray) -> list[str]:
    """Try pyzbar on a numpy/PIL-compatible image array."""
    pil = Image.fromarray(arr)
    return [r.data.decode("utf-8") for r in decode(pil)]


def _try_all_strategies(img_cv: np.ndarray) -> list[str]:
    """
    Apply several preprocessing strategies to maximise detection on
    low-resolution or badly-lit photos.
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    strategies = [
        # 1. Raw grayscale
        lambda g: g,
        # 2. Gaussian blur to reduce noise
        lambda g: cv2.GaussianBlur(g, (3, 3), 0),
        # 3. Otsu binarisation
        lambda g: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        # 4. Adaptive threshold
        lambda g: cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        ),
        # 5. Sharpen then Otsu
        lambda g: cv2.threshold(
            cv2.filter2D(
                g, -1,
                np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            ),
            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1],
        # 6. Upscale 2× then Otsu (helps very small QR codes)
        lambda g: cv2.threshold(
            cv2.resize(g, (g.shape[1] * 2, g.shape[0] * 2), interpolation=cv2.INTER_CUBIC),
            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1],
    ]

    for fn in strategies:
        processed = fn(gray)
        found = _decode_from_array(processed)
        if found:
            return found

    return []


def extract_qr_codes(image_path: str) -> list[str]:
    """Return a list of decoded QR code strings from an image file."""
    # --- attempt 1: direct PIL (fastest, works on clean images) ---
    try:
        pil_img = Image.open(image_path)
        results = decode(pil_img)
        if results:
            return [r.data.decode("utf-8") for r in results]
    except Exception:
        pass

    # --- attempt 2: OpenCV preprocessing pipeline ---
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return []

    return _try_all_strategies(img_cv)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_image(image_id: str) -> str:
    """
    Download the image from storage and save to a temp file.
    Returns the local temp file path.
    Raises HTTPException on failure.
    """
    url = f"{STORAGE_BASE_URL.rstrip('/')}/{image_id}"
    logger.info("Downloading image from %s", url)

    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Erro ao acessar storage: {exc}")

    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Imagem não encontrada no storage.")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Storage retornou status {resp.status_code}.",
        )

    content_type = resp.headers.get("Content-Type", "")
    ext = ".jpg"
    if "png" in content_type:
        ext = ".png"
    elif "webp" in content_type:
        ext = ".webp"
    elif "gif" in content_type:
        ext = ".gif"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@app.get("/read-qr/{image_id}")
def read_qr(image_id: str):
    """
    Baixa a imagem identificada por `image_id` do storage,
    lê o QR code e retorna o ticket_id.
    A imagem temporária é deletada ao final, com sucesso ou erro.
    """
    tmp_path: str | None = None

    try:
        tmp_path = download_image(image_id)
        codes = extract_qr_codes(tmp_path)

        if not codes:
            return JSONResponse(
                content={"success": False, "error": "NO_QRCODE"},
                status_code=200,
            )

        if len(codes) > 1:
            return JSONResponse(
                content={"success": False, "error": "MULTIPLE_QRCODES"},
                status_code=200,
            )

        ticket_id = codes[0].strip()
        if not ticket_id:
            return JSONResponse(
                content={"success": False, "error": "EMPTY_QRCODE"},
                status_code=200,
            )

        return JSONResponse(
            content={"success": True, "ticket_id": ticket_id},
            status_code=200,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Erro inesperado ao processar imagem %s", image_id)
        return JSONResponse(
            content={"success": False, "error": "INTERNAL_ERROR", "detail": str(exc)},
            status_code=500,
        )

    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                os.remove(tmp_path)
                logger.info("Arquivo temporário removido: %s", tmp_path)
            except Exception:
                logger.warning("Não foi possível remover temp file: %s", tmp_path)


@app.get("/health")
def health():
    return {"status": "ok"}
