from pathlib import Path
from typing import Any

from claude_real_video import process as extract_video

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


def process_source(
    *,
    source: str,
    output_dir: Path,
    chroma_path: Path,
    video_id: str,
    vision_model: str = "qwen2.5vl:3b",
    embedding_model: str = DEFAULT_MODEL,
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 32,
    replace: bool = False,
    limit: int | None = None,
    frame: int | None = None,
    scene: float = 0.30,
    fps_floor: float = 1.0,
    max_frames: int = 150,
    adaptive: bool = False,
    text_anchors: bool = True,
    language: str = "auto",
    whisper_model: str = "base",
    dedup_threshold: float = 8,
    dedup_window: int = 4,
    keep_audio: bool = True,
    report: bool = False,
    why: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Process a local video or URL from extraction through indexing."""
    output_dir = Path(output_dir)

    print("=" * 64)
    print("EXTRACCIÓN DE VIDEO, AUDIO, TEXTO Y FRAMES")
    print("=" * 64)
    extraction = extract_video(
        source,
        str(output_dir),
        scene=scene,
        adaptive=adaptive,
        text_anchors=text_anchors,
        fps_floor=fps_floor,
        max_frames=max_frames,
        lang=language,
        do_transcribe=True,
        whisper_model=whisper_model,
        dedup_threshold=dedup_threshold,
        dedup_window=dedup_window,
        keep_audio=keep_audio,
        report=report,
        why=why,
        overwrite=overwrite,
    )

    if extraction.frames_json_path is None:
        raise RuntimeError("La extracción no generó frames.json.")

    print(f"Video local          : {extraction.video}")
    print(f"Transcripción        : {extraction.transcript_path or extraction.transcript_note}")
    print(f"Audio                : {extraction.audio_path or '(no conservado)'}")
    print(f"Frames seleccionados : {extraction.frame_count}")
    print()

    result = process_video(
        manifest_path=Path(extraction.manifest_path),
        frames_json_path=Path(extraction.frames_json_path),
        frames_dir=Path(extraction.frames_dir),
        output_dir=output_dir,
        chroma_path=Path(chroma_path),
        video_id=video_id,
        video_path=extraction.video,
        vision_model=vision_model,
        embedding_model=embedding_model,
        collection_name=collection_name,
        batch_size=batch_size,
        replace=replace,
        limit=limit,
        frame=frame,
    )
    result["source"] = source
    result["video_path"] = extraction.video
    result["transcript_path"] = extraction.transcript_path
    result["audio_path"] = extraction.audio_path
    result["manifest_path"] = Path(extraction.manifest_path)
    result["frames_json_path"] = Path(extraction.frames_json_path)
    return result
