import argparse
from pathlib import Path

from app.embed import DEFAULT_COLLECTION, DEFAULT_MODEL
from app.pipeline import process_video


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Procesa un video: parsea su manifest, analiza sus frames e "
            "indexa los segmentos de audio y visión en ChromaDB."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frames-json", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chroma", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video-path", default="")
    parser.add_argument("--vision-model", default="qwen2.5vl:3b")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--frame", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    process_video(
        manifest_path=Path(args.manifest),
        frames_json_path=Path(args.frames_json),
        frames_dir=Path(args.frames_dir),
        output_dir=Path(args.output_dir),
        chroma_path=Path(args.chroma),
        video_id=args.video_id,
        video_path=args.video_path,
        vision_model=args.vision_model,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        batch_size=args.batch_size,
        replace=args.replace,
        limit=args.limit,
        frame=args.frame,
    )


if __name__ == "__main__":
    main()
