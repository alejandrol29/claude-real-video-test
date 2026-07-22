import json
import time
from pathlib import Path
from typing import Any


def _extract_texts(result: Any, minimum_score: float) -> list[str]:
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)

    if texts is None and isinstance(result, tuple) and result:
        lines = result[0] or []
        texts = [line[1] for line in lines if len(line) >= 2]
        scores = [line[2] for line in lines if len(line) >= 3]

    if not texts:
        return []

    if not scores or len(scores) != len(texts):
        scores = [1.0] * len(texts)

    normalized = []
    seen = set()

    for text, score in zip(texts, scores):
        value = str(text).strip()

        if not value or float(score) < minimum_score or value in seen:
            continue

        seen.add(value)
        normalized.append(value)

    return normalized


def process_ocr_frames(
    frames_json_path: Path,
    frames_dir: Path,
    output_path: Path,
    minimum_score: float = 0.55,
) -> dict[str, Any]:
    """Run lightweight OCR over every extracted frame and save indexable segments."""
    try:
        from rapidocr import RapidOCR
    except ImportError as error:
        raise RuntimeError(
            "RapidOCR no está instalado. Ejecutá: pip install rapidocr"
        ) from error

    frames_json_path = Path(frames_json_path)
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    frames_data = json.loads(frames_json_path.read_text(encoding="utf-8"))
    frames = frames_data.get("frames")

    if not isinstance(frames, list):
        raise ValueError("frames.json no contiene una lista válida de frames.")

    print("=" * 64)
    print("OCR DE TODOS LOS FRAMES")
    print("=" * 64)
    print(f"Frames        : {len(frames)}")
    print(f"Salida        : {output_path}")
    print()

    engine = RapidOCR()
    segments = []
    errors = []
    start_time = time.perf_counter()

    for position, frame_data in enumerate(frames, start=1):
        filename = str(frame_data.get("file", ""))
        image_path = frames_dir / filename

        try:
            texts = _extract_texts(engine(str(image_path)), minimum_score)
        except Exception as error:
            errors.append(
                {
                    "frame": filename,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        if texts:
            timestamp_sec = frame_data.get("timestamp_sec")
            segments.append(
                {
                    "id": f"ocr-{image_path.stem}",
                    "type": "ocr",
                    "source": "rapidocr",
                    "start": timestamp_sec,
                    "end": timestamp_sec,
                    "timestamp": frame_data.get("timestamp"),
                    "timestamp_sec": timestamp_sec,
                    "frame": filename,
                    "frame_path": str(image_path),
                    "visible_text": texts,
                    "text": f"Texto visible: {', '.join(texts)}.",
                }
            )

        print(
            f"[{position:03d}/{len(frames):03d}] {filename}: "
            f"{len(texts)} textos",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "frames_processed": len(frames),
        "frames_with_text": len(segments),
        "frames_failed": len(errors),
        "processing_time_seconds": round(elapsed, 3),
        "errors": errors,
    }
    print(f"Frames con texto : {len(segments)}")
    print(f"Tiempo OCR       : {elapsed:.2f} s")
    return summary
