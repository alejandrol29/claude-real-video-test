import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

from app.embed import DEFAULT_COLLECTION, DEFAULT_MODEL
from app.pipeline import process_source
from app.siglip import DEFAULT_SIGLIP_COLLECTION, DEFAULT_SIGLIP_MODEL
from app.vision import FAST_VISION_MODEL


def default_video_id(source: str) -> str:
    parsed = urlparse(source)
    candidate = Path(parsed.path).stem if parsed.scheme else Path(source).stem
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", candidate).strip("-")
    return normalized or "video"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Procesa un archivo de video o URL de punta a punta: audio, "
            "transcripción, frames, análisis visual e indexación en ChromaDB."
        )
    )
    parser.add_argument("source", help="Archivo de video local o URL.")
    parser.add_argument("--video-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--chroma")
    parser.add_argument("--vision-model", default=FAST_VISION_MODEL)
    parser.add_argument(
        "--visual-frames",
        type=int,
        default=None,
        help=(
            "Cantidad fija de keyframes para el modelo visual. Si se omite, "
            "se calcula según la duración del video."
        ),
    )
    parser.add_argument(
        "--visual-interval",
        type=float,
        default=90.0,
        help="Cobertura visual automática aproximada en segundos.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_false",
        dest="run_ocr",
        help="Desactiva OCR sobre el conjunto completo de frames.",
    )
    parser.add_argument("--ocr-min-score", type=float, default=0.55)
    parser.add_argument(
        "--no-siglip",
        action="store_false",
        dest="run_siglip",
        help="Usa muestreo temporal y no crea el índice visual SigLIP.",
    )
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--siglip-collection", default=DEFAULT_SIGLIP_COLLECTION)
    parser.add_argument("--siglip-batch-size", type=int, default=16)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--scene", type=float, default=0.30)
    parser.add_argument("--fps-floor", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument(
        "--no-text-anchors",
        action="store_false",
        dest="text_anchors",
        help="No fuerza frames adicionales en timestamps de subtítulos.",
    )
    parser.add_argument("--lang", default="auto")
    parser.add_argument(
        "--whisper-model",
        choices=("tiny", "base", "small", "medium", "large", "turbo"),
        default="base",
    )
    parser.add_argument("--dedup-threshold", type=float, default=8)
    parser.add_argument("--dedup-window", type=int, default=4)
    parser.add_argument(
        "--no-keep-audio",
        action="store_false",
        dest="keep_audio",
        help="No conserva una copia separada del audio completo.",
    )
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--why")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    video_id = args.video_id or default_video_id(args.source)
    output_dir = Path(args.output_dir or f"output/{video_id}")
    chroma_path = Path(args.chroma or f"data/chroma-{video_id}")

    process_source(
        source=args.source,
        output_dir=output_dir,
        chroma_path=chroma_path,
        video_id=video_id,
        vision_model=args.vision_model,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        batch_size=args.batch_size,
        replace=args.replace,
        limit=args.limit,
        frame=args.frame,
        visual_frames=args.visual_frames,
        visual_interval=args.visual_interval,
        run_ocr=args.run_ocr,
        ocr_minimum_score=args.ocr_min_score,
        run_siglip=args.run_siglip,
        siglip_model=args.siglip_model,
        siglip_collection=args.siglip_collection,
        siglip_batch_size=args.siglip_batch_size,
        scene=args.scene,
        fps_floor=args.fps_floor,
        max_frames=args.max_frames,
        adaptive=args.adaptive,
        text_anchors=args.text_anchors,
        language=args.lang,
        whisper_model=args.whisper_model,
        dedup_threshold=args.dedup_threshold,
        dedup_window=args.dedup_window,
        keep_audio=args.keep_audio,
        report=args.report,
        why=args.why,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
