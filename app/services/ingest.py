"""文档接入：多格式解析 + 扫描件 OCR + 父子块 + 增量入索引。"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

from app.config import DATA_DIR, settings
from app.services.chunker import chunk_plain_document
from app.services.retriever import get_retriever

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ocr_engine = None


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_image(raw: bytes) -> tuple[str, float]:
    from PIL import Image
    import numpy as np

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img)
    res, elapse = _get_ocr()(arr)
    if not res:
        return "", 0.0
    texts = [r[1] for r in res if r and len(r) > 1]
    confs = [float(r[2]) for r in res if r and len(r) > 2]
    return "\n".join(texts), (sum(confs) / len(confs) if confs else 0.0)


def _pdf_text(raw: bytes) -> tuple[str, int, bool]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    pages, empty = [], 0
    for p in reader.pages:
        t = (p.extract_text() or "").strip()
        if len(t) < 20:
            empty += 1
        pages.append(t)
    text = "\n".join(pages).strip()
    return text, len(reader.pages), bool(empty)


def _docx_text(raw: bytes) -> str:
    import docx

    d = docx.Document(io.BytesIO(raw))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for tb in d.tables:
        for row in tb.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def extract_text(filename: str, raw: bytes) -> tuple[str, bool, dict]:
    """返回 (文本, 是否用了 OCR, 元信息)"""
    ext = Path(filename).suffix.lower()
    meta: dict = {"format": ext.lstrip(".")}
    ocr_used = False

    if ext in (".txt", ".md", ".csv"):
        text = raw.decode("utf-8", errors="ignore")
        if not text.strip():
            text = raw.decode("gbk", errors="ignore")
        return text, False, meta

    if ext == ".pdf":
        text, pages, has_empty = _pdf_text(raw)
        meta["pages"] = pages
        if len(text) < 30 * max(1, pages) * 0.3:
            # 扫描件：尝试整页 OCR（把 PDF 首页当图片处理需要额外依赖，这里给出明确提示）
            meta["scanned"] = True
        return text, False, meta

    if ext == ".docx":
        return _docx_text(raw), False, meta

    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"):
        text, conf = _ocr_image(raw)
        meta["ocr_conf"] = round(conf, 3)
        return text, True, meta

    raise ValueError(f"不支持的文件类型：{ext}")


def _slug(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]+", "_", Path(name).stem)[:40]
    return f"up_{s}_{int(time.time()) % 100000}"


def ingest_document(filename: str, raw: bytes, category: str = "未分类") -> dict:
    t0 = time.perf_counter()
    text, ocr_used, meta = extract_text(filename, raw)
    if not text.strip():
        raise ValueError("未解析到文本内容（可能是扫描件，请上传图片格式或先做 OCR）")

    doc_id = _slug(filename)
    save_path = UPLOAD_DIR / f"{doc_id}{Path(filename).suffix.lower()}"
    save_path.write_bytes(raw)

    children, parents = chunk_plain_document(text, doc_id, Path(filename).stem, category)

    # 入索引
    retr = get_retriever()
    retr.add_chunks(children, parents)

    # 元数据落盘
    import json

    manifest = DATA_DIR / "uploads.json"
    items = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    items.append(
        {
            "doc_id": doc_id,
            "title": Path(filename).stem,
            "filename": filename,
            "category": category,
            "chars": len(text),
            "chunks": len(children),
            "parents": len(parents),
            "ocr_used": ocr_used,
            "meta": meta,
            "path": str(save_path),
            "created_at": time.time(),
        }
    )
    manifest.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    return {
        "doc_id": doc_id,
        "title": Path(filename).stem,
        "category": category,
        "chars": len(text),
        "chunks": len(children),
        "ocr_used": ocr_used,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
    }
