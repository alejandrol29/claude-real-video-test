from pathlib import Path
from typing import Any

from app.embed import DEFAULT_COLLECTION, DEFAULT_MODEL, index_segments
from app.vision import process_frames
from scripts.parse_manifest import parse_manifest, save_segments


def process_video(
    *,
    manifest_path: Path,
    frames_json_path: Path,
    frames_dir: Path,
    output_dir: Path,
    chroma_path: Path,
    video_id: str,
    video_path: str = "",
    vision_model: str = "qwen2.5vl:3b",
    embedding_model: str = DEFAULT_MODEL,
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 32,
    replace: bool = False,
    limit: int | None = None,
    frame: int | None = None,
) -> dict[str, Any]:
    """Run manifest parsing, visual analysis, and multimodal indexing."""
    manifest_path = Path(manifest_path)
    frames_json_path = Path(frames_json_path)
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    chroma_path = Path(chroma_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments_path = output_dir / "segments.json"
    visual_segments_path = output_dir / "visual_segments.json"

    print("=" * 64)
    print("PARSEO DEL MANIFEST")
    print("=" * 64)
    audio_segments = parse_manifest(manifest_path)
    save_segments(audio_segments, segments_path)
    print(f"Segmentos encontrados : {len(audio_segments)}")
    print(f"Archivo generado      : {segments_path}")
    print()

    vision_summary = process_frames(
        frames_json_path=frames_json_path,
        frames_dir=frames_dir,
        output_path=visual_segments_path,
        model=vision_model,
        limit=limit,
        frame=frame,
    )
    print()

    audio_index = index_segments(
        segments_path=segments_path,
        chroma_path=chroma_path,
        video_id=video_id,
        video_path=video_path,
        model_name=embedding_model,
        collection_name=collection_name,
        batch_size=batch_size,
        replace=replace,
    )
    print()
    visual_index = index_segments(
        segments_path=visual_segments_path,
        chroma_path=chroma_path,
        video_id=video_id,
        video_path=video_path,
        model_name=embedding_model,
        collection_name=collection_name,
        batch_size=batch_size,
        replace=False,
    )

    return {
        "segments_path": segments_path,
        "visual_segments_path": visual_segments_path,
        "vision": vision_summary,
        "audio_index": audio_index,
        "visual_index": visual_index,
    }
