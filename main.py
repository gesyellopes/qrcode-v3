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


def _smart_resize(img: np.ndarray, max_dim: int = 600) -> np.ndarray:
    """
    Downscale image so its longest side is at most max_dim pixels.
    pyzbar is O(pixels) — keeping images small is the #1 speed lever.
    QR codes are robust to downscaling; 600px is enough for any readable QR.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _decode_regions(img: np.ndarray) -> set[str]:
    """
    Decode QR codes from an image using overlapping horizontal slices.
    Each slice is tried with grayscale and Otsu binarisation.
    Results are deduplicated by value.
    """
    h = img.shape[0]
    seen: set[str] = set()

    regions = [
        img,                        # full image
        img[:h // 2, :],            # top half
        img[h // 2:, :],            # bottom half
        img[:int(h * 0.55), :],     # top 55% (overlap)
        img[int(h * 0.45):, :],     # bottom 55% (overlap)
    ]

    for region in regions:
        if region.size == 0:
            continue
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for proc in [gray, otsu]:
                for r in decode(Image.fromarray(proc)):
                    seen.add(r.data.decode("utf-8"))
        except Exception:
            pass

    return seen


def _try_all_strategies(img_cv: np.ndarray) -> list[str]:
    """Kept for compatibility — delegates to _decode_regions."""
    return list(_decode_regions(img_cv))


def extract_qr_codes(image_path: str) -> list[str]:
    """
    Return a deduplicated list of QR code values found in the image.

    Strategy (cascade — stops as soon as enough info is available):
    1. Load and downscale to max 600px (massive speed gain, QRs survive it).
    2. Try PIL direct decode on the full image (instant for clean images).
    3. Run overlapping-slice pipeline with grayscale + Otsu.
    """
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        # Fallback: let PIL try (handles formats OpenCV may not)
        try:
            pil_img = Image.open(image_path)
            return [r.data.decode("utf-8") for r in decode(pil_img)]
        except Exception:
            return []

    img_cv = _smart_resize(img_cv)

    # Fast path: try PIL on the resized full image first
    seen: set[str] = set()
    try:
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        for r in decode(Image.fromarray(rgb)):
            seen.add(r.data.decode("utf-8"))
    except Exception:
        pass

    # Full pipeline on all regions
    seen |= _decode_regions(img_cv)

    return list(seen)


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
                content={"success": False, "reason": "NO_QRCODE"},
                status_code=200,
            )

        if len(codes) > 1:
            return JSONResponse(
                content={"success": False, "reason": "MULTIPLE_QRCODES"},
                status_code=200,
            )

        ticket_id = codes[0].strip()
        if not ticket_id:
            return JSONResponse(
                content={"success": False, "reason": "EMPTY_QRCODE"},
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