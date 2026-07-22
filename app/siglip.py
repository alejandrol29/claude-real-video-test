import argparse
import json
import time
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoProcessor


DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_SIGLIP_COLLECTION = "video_frames_siglip"


def _pooled_features(output: Any) -> torch.Tensor:
    """Support both current and older Transformers feature return types."""
    return output.pooler_output if hasattr(output, "pooler_output") else output


def load_siglip_model(model_name: str = DEFAULT_SIGLIP_MODEL) -> tuple[Any, Any, str]:
    """Load SigLIP once and prefer Apple's GPU when it is available."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
        )
    except OSError:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

    model = model.to(device)
    model.eval()
    return model, processor, device


def load_frame_metadata(frames_json_path: Path) -> list[dict[str, Any]]:
    with Path(frames_json_path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames.json no contiene una lista de frames válida.")
    return frames


def embed_frames(
    frames: list[dict[str, Any]],
    frames_dir: Path,
    model: Any,
    processor: Any,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Generate normalized image embeddings for every extracted frame."""
    if batch_size <= 0:
        raise ValueError("El batch size de SigLIP debe ser mayor que cero.")

    batches = []
    frames_dir = Path(frames_dir)
    with torch.inference_mode():
        for offset in range(0, len(frames), batch_size):
            frame_batch = frames[offset : offset + batch_size]
            images = []
            for frame in frame_batch:
                image_path = frames_dir / str(frame.get("file", ""))
                if not image_path.exists():
                    raise FileNotFoundError(f"No existe el frame: {image_path}")
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))

            inputs = {
                key: value.to(device)
                for key, value in processor(images=images, return_tensors="pt").items()
            }
            output = model.get_image_features(**inputs)
            embeddings = _pooled_features(output)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            batches.append(embeddings.cpu())
            processed = min(offset + batch_size, len(frames))
            print(f"Frames SigLIP       : {processed}/{len(frames)}", flush=True)

    if device == "mps":
        torch.mps.synchronize()
    return torch.cat(batches).numpy()


def select_diverse_frames(embeddings: np.ndarray, frame_count: int) -> list[int]:
    """Select actual frames nearest to visual cluster centroids."""
    total_frames = len(embeddings)
    if frame_count <= 0:
        raise ValueError("La cantidad de frames visuales debe ser mayor que cero.")
    if frame_count >= total_frames:
        return list(range(total_frames))

    boundary_indices = [0, total_frames - 1]
    selected = list(dict.fromkeys(boundary_indices))
    remaining_count = frame_count - len(selected)
    if remaining_count <= 0:
        return selected[:frame_count]

    clustering = KMeans(
        n_clusters=remaining_count,
        random_state=42,
        n_init=10,
    ).fit(embeddings)
    centers = clustering.cluster_centers_
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    for center in centers:
        candidates = np.argsort(-(embeddings @ center))
        selected.append(
            next(int(index) for index in candidates if int(index) not in selected)
        )
    return sorted(selected)


def embed_queries(
    queries: list[str],
    model: Any,
    processor: Any,
    device: str,
) -> np.ndarray:
    """Encode natural-language visual searches in SigLIP's vector space."""
    inputs = {
        key: value.to(device)
        for key, value in processor(
            text=queries,
            padding="max_length",
            return_tensors="pt",
        ).items()
    }
    with torch.inference_mode():
        embeddings = _pooled_features(model.get_text_features(**inputs))
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    if device == "mps":
        torch.mps.synchronize()
    return embeddings.cpu().numpy()


def search_frame_embeddings(
    *,
    chroma_path: Path,
    query_embedding: np.ndarray,
    collection_name: str = DEFAULT_SIGLIP_COLLECTION,
    results: int = 5,
) -> dict[str, Any]:
    """Search an existing SigLIP frame collection with a precomputed query."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(collection_name)
    result_count = min(results, collection.count())
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )


def index_frame_embeddings(
    *,
    frames: list[dict[str, Any]],
    embeddings: np.ndarray,
    frames_dir: Path,
    chroma_path: Path,
    video_id: str,
    video_path: str,
    model_name: str,
    collection_name: str = DEFAULT_SIGLIP_COLLECTION,
    replace: bool = False,
) -> dict[str, Any]:
    """Store visual embeddings in a collection separate from text vectors."""
    chroma_path = Path(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    if replace:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )
    ids = []
    documents = []
    metadatas = []
    for position, frame in enumerate(frames):
        filename = str(frame.get("file", ""))
        timestamp_sec = float(frame.get("timestamp_sec", 0.0))
        ids.append(f"{video_id}:siglip:{filename}")
        documents.append(f"Frame visual en {frame.get('timestamp', timestamp_sec)}")
        metadatas.append(
            {
                "video_id": video_id,
                "video_path": video_path,
                "type": "siglip",
                "source": model_name,
                "start": timestamp_sec,
                "end": timestamp_sec,
                "timestamp": str(frame.get("timestamp", "")),
                "timestamp_sec": timestamp_sec,
                "frame": filename,
                "frame_path": str(Path(frames_dir) / filename),
                "selection_reason": str(frame.get("selection_reason", "")),
                "position": position,
            }
        )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    return {
        "frames_indexed": len(frames),
        "dimensions": int(embeddings.shape[1]),
        "collection": collection_name,
    }


def process_siglip_frames(
    *,
    frames_json_path: Path,
    frames_dir: Path,
    chroma_path: Path,
    video_id: str,
    video_path: str = "",
    selected_count: int = 15,
    model_name: str = DEFAULT_SIGLIP_MODEL,
    collection_name: str = DEFAULT_SIGLIP_COLLECTION,
    batch_size: int = 16,
    replace: bool = False,
) -> dict[str, Any]:
    """Embed/index every frame and select a diverse subset for a VLM."""
    print("=" * 64)
    print("INDEXACIÓN VISUAL RÁPIDA CON SIGLIP")
    print("=" * 64)
    print(f"Modelo             : {model_name}")
    print(f"Colección          : {collection_name}")
    frames = load_frame_metadata(frames_json_path)
    load_start = time.perf_counter()
    model, processor, device = load_siglip_model(model_name)
    model_load_time = time.perf_counter() - load_start
    embedding_start = time.perf_counter()
    embeddings = embed_frames(
        frames, frames_dir, model, processor, device, batch_size
    )
    embedding_time = time.perf_counter() - embedding_start
    selected_indices = select_diverse_frames(embeddings, selected_count)
    selected_files = [str(frames[index]["file"]) for index in selected_indices]
    indexing_start = time.perf_counter()
    index_result = index_frame_embeddings(
        frames=frames,
        embeddings=embeddings,
        frames_dir=frames_dir,
        chroma_path=chroma_path,
        video_id=video_id,
        video_path=video_path,
        model_name=model_name,
        collection_name=collection_name,
        replace=replace,
    )
    indexing_time = time.perf_counter() - indexing_start
    result = {
        **index_result,
        "model": model_name,
        "device": device,
        "model_load_time": round(model_load_time, 3),
        "embedding_time": round(embedding_time, 3),
        "indexing_time": round(indexing_time, 3),
        "selected_count": len(selected_files),
        "selected_files": selected_files,
    }
    print(f"Dispositivo         : {device}")
    print(f"Frames indexados    : {len(frames)}")
    print(f"Frames para Qwen    : {len(selected_files)}")
    print(f"Tiempo embeddings   : {embedding_time:.3f} s")
    print(f"Tiempo indexación   : {indexing_time:.3f} s")
    print()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indexa frames con SigLIP y selecciona una muestra diversa."
    )
    parser.add_argument("--frames-json", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--chroma", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video-path", default="")
    parser.add_argument("--selected-frames", type=int, default=15)
    parser.add_argument("--model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--collection", default=DEFAULT_SIGLIP_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    process_siglip_frames(
        frames_json_path=Path(args.frames_json),
        frames_dir=Path(args.frames_dir),
        chroma_path=Path(args.chroma),
        video_id=args.video_id,
        video_path=args.video_path,
        selected_count=args.selected_frames,
        model_name=args.model,
        collection_name=args.collection,
        batch_size=args.batch_size,
        replace=args.replace,
    )


if __name__ == "__main__":
    main()
