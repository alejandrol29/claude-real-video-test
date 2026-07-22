import json
from pathlib import Path
from typing import Any

import numpy as np

from app.embed import build_metadata
from app.vision import build_index_text, describe_frame


def modality_weights(query: str) -> tuple[float, float]:
    """Favor the modality explicitly requested while keeping hybrid recall."""
    normalized = query.casefold()
    audio_markers = (
        "qué dijeron",
        "que dijeron",
        "dijo",
        "habló",
        "hablaron",
        "mencionó",
        "comentó",
        "escuchar",
        "audio",
    )
    text_markers = (
        "texto",
        "escrito",
        "cartel",
        "subtítulo",
        "marcador",
        "logo",
        "nombre visible",
    )
    if any(marker in normalized for marker in audio_markers):
        return 0.8, 0.2
    if any(marker in normalized for marker in text_markers):
        return 0.7, 0.3
    return 0.5, 0.5


def _collection_results(
    collection: Any,
    embedding: np.ndarray,
    result_count: int,
    modality: str,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if collection is None or collection.count() == 0:
        return []

    available = collection.count()
    if where is not None:
        available = len(collection.get(where=where, include=[])["ids"])
    if available == 0:
        return []

    query_arguments = {
        "query_embeddings": [embedding.tolist()],
        "n_results": min(result_count, available),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_arguments["where"] = where
    response = collection.query(
        **query_arguments,
    )
    results = [
        {
            "document": document,
            "metadata": metadata,
            "similarity": 1 - distance,
            "modalities": [modality],
        }
        for document, metadata, distance in zip(
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        )
    ]
    if modality == "imagen":
        for result in results:
            result["image_metadata"] = result["metadata"]
    return results


def _same_moment(
    left: dict[str, Any],
    right: dict[str, Any],
    temporal_window: float,
) -> bool:
    left_meta = left["metadata"]
    right_meta = right["metadata"]
    if left_meta.get("video_id") != right_meta.get("video_id"):
        return False
    left_frame = left_meta.get("frame")
    right_frame = right_meta.get("frame")
    if left_frame and right_frame and left_frame == right_frame:
        return True
    return abs(float(left_meta.get("start", 0)) - float(right_meta.get("start", 0))) <= temporal_window


def hybrid_search(
    *,
    text_collection: Any,
    text_embedding: np.ndarray,
    siglip_collection: Any | None = None,
    siglip_embedding: np.ndarray | None = None,
    results: int = 5,
    candidates: int = 20,
    temporal_window: float = 6.0,
    text_weight: float = 0.5,
    siglip_weight: float = 0.5,
    use_text: bool = True,
    use_siglip: bool = True,
    text_where: dict[str, Any] | None = None,
    text_modality: str = "texto",
) -> list[dict[str, Any]]:
    """Fuse text and direct-image retrieval using weighted reciprocal ranks."""
    ranked_lists = []
    if use_text:
        ranked_lists.append((
            _collection_results(
                text_collection,
                text_embedding,
                candidates,
                text_modality,
                text_where,
            ),
            text_weight,
        ))
    if use_siglip and siglip_collection is not None and siglip_embedding is not None:
        ranked_lists.append(
            (
                _collection_results(
                    siglip_collection,
                    siglip_embedding,
                    candidates,
                    "imagen",
                ),
                siglip_weight,
            )
        )

    fused: list[dict[str, Any]] = []
    for ranked, weight in ranked_lists:
        for rank, candidate in enumerate(ranked, start=1):
            match = next(
                (
                    current
                    for current in fused
                    if _same_moment(current, candidate, temporal_window)
                ),
                None,
            )
            score = weight / (60 + rank)
            if match is None:
                candidate["fusion_score"] = score
                fused.append(candidate)
                continue

            adds_new_modality = any(
                modality not in match["modalities"]
                for modality in candidate["modalities"]
            )
            if adds_new_modality:
                match["fusion_score"] += score
            else:
                match["fusion_score"] = max(match["fusion_score"], score)
            match["similarity"] = max(match["similarity"], candidate["similarity"])
            match["modalities"] = sorted(
                set(match["modalities"] + candidate["modalities"])
            )
            if "image_metadata" in candidate and "image_metadata" not in match:
                match["image_metadata"] = candidate["image_metadata"]
            current_type = str(match["metadata"].get("type", ""))
            candidate_type = str(candidate["metadata"].get("type", ""))
            if current_type == "siglip" and candidate_type != "siglip":
                match["document"] = candidate["document"]
                match["metadata"] = candidate["metadata"]

    fused.sort(key=lambda item: item["fusion_score"], reverse=True)
    deduplicated: list[dict[str, Any]] = []
    for candidate in fused:
        if any(
            _same_moment(existing, candidate, temporal_window)
            for existing in deduplicated
        ):
            continue
        deduplicated.append(candidate)
        if len(deduplicated) == results:
            break
    return deduplicated


def enrichment_path(frame_path: str) -> Path:
    return Path(frame_path).parent.parent / "on_demand_visual_segments.json"


def load_enrichment(frame_path: str) -> dict[str, Any] | None:
    path = enrichment_path(frame_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        segments = json.load(file)
    frame = Path(frame_path).name
    return next((item for item in segments if item.get("frame") == frame), None)


def enrich_frame(
    metadata: dict[str, Any],
    vision_model: str,
) -> dict[str, Any]:
    """Describe a retrieved frame once and persist it for future searches."""
    frame_path = str(metadata["frame_path"])
    cached = load_enrichment(frame_path)
    if cached is not None:
        return cached

    analysis, processing_time = describe_frame(Path(frame_path), vision_model)
    segment = {
        "id": f"on_demand_{Path(frame_path).stem}",
        "type": "vision_on_demand",
        "source": vision_model,
        "start": float(metadata.get("start", 0.0)),
        "end": float(metadata.get("end", metadata.get("start", 0.0))),
        "timestamp": str(metadata.get("timestamp", "")),
        "timestamp_sec": float(metadata.get("timestamp_sec", metadata.get("start", 0.0))),
        "frame": str(metadata.get("frame", Path(frame_path).name)),
        "frame_path": frame_path,
        "selection_reason": "query_enrichment",
        "processing_time_seconds": round(processing_time, 3),
        **analysis,
        "text": build_index_text(analysis),
    }
    path = enrichment_path(frame_path)
    segments = []
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            segments = json.load(file)
    segments = [item for item in segments if item.get("frame") != segment["frame"]]
    segments.append(segment)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(segments, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)
    return segment


def index_enrichment(
    *,
    segment: dict[str, Any],
    collection: Any,
    embedding_model: Any,
    video_id: str,
    video_path: str,
) -> None:
    embedding = embedding_model.encode(
        [segment["text"]],
        normalize_embeddings=True,
    )[0]
    metadata = build_metadata(
        segment,
        video_id,
        video_path,
        str(enrichment_path(segment["frame_path"])),
        0,
    )
    collection.upsert(
        ids=[f"{video_id}:{segment['id']}"],
        documents=[segment["text"]],
        embeddings=[embedding.tolist()],
        metadatas=[metadata],
    )
