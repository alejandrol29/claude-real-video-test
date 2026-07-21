import argparse
import json
import time
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_COLLECTION = "video_segments"


def load_segments(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict):
        possible_keys = (
            "segments",
            "visual_segments",
            "multimodal_segments",
            "items",
        )

        segments = None

        for key in possible_keys:
            value = data.get(key)

            if isinstance(value, list):
                segments = value
                break

        if segments is None:
            raise ValueError(
                "El JSON no contiene una lista de segmentos reconocible."
            )
    else:
        raise ValueError("El JSON debe contener una lista o un objeto.")

    valid_segments = []

    for position, segment in enumerate(segments):
        if not isinstance(segment, dict):
            print(f"Advertencia: segmento {position} ignorado; no es un objeto.")
            continue

        text = segment.get("text")

        if not isinstance(text, str) or not text.strip():
            print(
                f"Advertencia: segmento {position} ignorado; "
                "no contiene un campo 'text' válido."
            )
            continue

        valid_segments.append(segment)

    if not valid_segments:
        raise ValueError("No se encontraron segmentos con texto válido.")

    return valid_segments


def normalize_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        return float(value)
    except (TypeError, ValueError):
        return default


def build_document_id(
    segment: dict[str, Any],
    video_id: str,
    position: int,
) -> str:
    segment_id = segment.get("id")

    if segment_id is not None and str(segment_id).strip():
        return f"{video_id}:{segment_id}"

    segment_type = str(segment.get("type", "segment"))
    start = normalize_number(
        segment.get("start", segment.get("timestamp_sec", position))
    )

    return f"{video_id}:{segment_type}:{position:06d}:{start:.3f}"


def build_metadata(
    segment: dict[str, Any],
    video_id: str,
    video_path: str,
    segments_path: str,
    position: int,
) -> dict[str, str | int | float | bool]:
    start = normalize_number(
        segment.get("start", segment.get("timestamp_sec", 0.0))
    )

    end = normalize_number(
        segment.get("end", segment.get("timestamp_sec", start)),
        default=start,
    )

    metadata: dict[str, str | int | float | bool] = {
        "video_id": video_id,
        "video_path": video_path,
        "segments_path": segments_path,
        "position": position,
        "type": str(segment.get("type", "unknown")),
        "source": str(segment.get("source", "unknown")),
        "start": start,
        "end": end,
    }

    optional_fields = (
        "timestamp",
        "timestamp_sec",
        "frame",
        "frame_path",
        "selection_reason",
        "processing_time_seconds",
    )

    for field in optional_fields:
        value = segment.get(field)

        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            metadata[field] = value
        else:
            metadata[field] = str(value)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Genera embeddings de segmentos textuales y los indexa en ChromaDB. "
            "Admite segmentos de audio, visión o contenido multimodal."
        )
    )

    parser.add_argument(
        "--segments",
        required=True,
        help="Archivo JSON con segmentos que contengan el campo 'text'.",
    )

    parser.add_argument(
        "--chroma",
        required=True,
        help="Directorio persistente de ChromaDB.",
    )

    parser.add_argument(
        "--video-id",
        required=True,
        help="Identificador lógico del video o experimento.",
    )

    parser.add_argument(
        "--video-path",
        default="",
        help="Ruta del archivo de video original.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modelo de embeddings. Predeterminado: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Nombre de la colección. Predeterminado: {DEFAULT_COLLECTION}",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Cantidad de textos procesados por lote.",
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help="Elimina la colección existente antes de indexar.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size debe ser mayor que cero.")

    segments_path = Path(args.segments)
    chroma_path = Path(args.chroma)

    print("=" * 64)
    print("INDEXACIÓN DE SEGMENTOS")
    print("=" * 64)
    print(f"Segmentos     : {segments_path}")
    print(f"ChromaDB      : {chroma_path}")
    print(f"Video ID      : {args.video_id}")
    print(f"Video         : {args.video_path or '(no informado)'}")
    print(f"Modelo        : {args.model}")
    print(f"Colección     : {args.collection}")
    print(f"Batch size    : {args.batch_size}")
    print()

    load_start = time.perf_counter()
    segments = load_segments(segments_path)
    load_time = time.perf_counter() - load_start

    texts = [segment["text"].strip() for segment in segments]

    type_counts: dict[str, int] = {}

    for segment in segments:
        segment_type = str(segment.get("type", "unknown"))
        type_counts[segment_type] = type_counts.get(segment_type, 0) + 1

    print(f"Segmentos válidos: {len(segments)}")
    print(f"Tipos encontrados: {type_counts}")
    print(f"Lectura JSON      : {load_time:.3f} s")
    print()

    print("Cargando modelo de embeddings...")
    model_load_start = time.perf_counter()

    model = SentenceTransformer(args.model)

    model_load_time = time.perf_counter() - model_load_start

    print(f"Modelo cargado en : {model_load_time:.3f} s")
    print()

    print("Generando embeddings...")
    embedding_start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embedding_time = time.perf_counter() - embedding_start

    print(f"Embeddings         : {len(embeddings)}")
    print(f"Dimensiones        : {embeddings.shape[1]}")
    print(f"Tiempo embeddings  : {embedding_time:.3f} s")
    print()

    print("Abriendo ChromaDB...")
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))

    if args.replace:
        try:
            client.delete_collection(args.collection)
            print(f"Colección eliminada: {args.collection}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": args.model,
            "video_id": args.video_id,
        },
    )

    ids = []
    metadatas = []

    for position, segment in enumerate(segments):
        ids.append(
            build_document_id(
                segment=segment,
                video_id=args.video_id,
                position=position,
            )
        )

        metadatas.append(
            build_metadata(
                segment=segment,
                video_id=args.video_id,
                video_path=args.video_path,
                segments_path=str(segments_path),
                position=position,
            )
        )

    print("Indexando en ChromaDB...")
    indexing_start = time.perf_counter()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    indexing_time = time.perf_counter() - indexing_start
    total_time = (
        load_time
        + model_load_time
        + embedding_time
        + indexing_time
    )

    print()
    print("=" * 64)
    print("RESULTADOS DE INDEXACIÓN")
    print("=" * 64)
    print(f"Segmentos indexados : {len(segments)}")
    print(f"Tipos               : {type_counts}")
    print(f"Dimensiones         : {embeddings.shape[1]}")
    print(f"Carga del JSON      : {load_time:.3f} s")
    print(f"Carga del modelo    : {model_load_time:.3f} s")
    print(f"Generación embedding: {embedding_time:.3f} s")
    print(f"Indexación Chroma   : {indexing_time:.3f} s")
    print(f"Tiempo total        : {total_time:.3f} s")
    print(f"Directorio Chroma   : {chroma_path}")
    print(f"Colección           : {args.collection}")


if __name__ == "__main__":
    main()
