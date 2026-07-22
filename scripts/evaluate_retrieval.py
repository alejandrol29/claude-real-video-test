import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embed import load_embedding_model
from app.retrieval import hybrid_search, modality_weights
from app.siglip import embed_queries, load_siglip_model


MODES = {
    "siglip": {
        "use_text": False,
        "use_siglip": True,
        "text_where": None,
        "text_modality": "texto",
    },
    "siglip_qwen": {
        "use_text": True,
        "use_siglip": True,
        "text_where": {
            "type": {"$in": ["vision", "vision_on_demand"]}
        },
        "text_modality": "Qwen",
    },
    "qwen": {
        "use_text": True,
        "use_siglip": False,
        "text_where": {
            "type": {"$in": ["vision", "vision_on_demand"]}
        },
        "text_modality": "Qwen",
    },
    "complete": {
        "use_text": True,
        "use_siglip": True,
        "text_where": None,
        "text_modality": "texto",
    },
}


def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        queries = json.load(file)
    if not isinstance(queries, list) or not queries:
        raise ValueError("El archivo de evaluación no contiene consultas.")
    return queries


def matching_event(timestamp: float, events: list[list[float]]) -> int | None:
    for index, (start, end) in enumerate(events):
        if float(start) <= timestamp <= float(end):
            return index
    return None


def visual_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Use the matched frame timestamp when fusion attached one to text."""
    return result.get("image_metadata") or result["metadata"]


def score_results(
    results: list[dict[str, Any]],
    events: list[list[float]],
) -> dict[str, float | int]:
    matched_events = set()
    relevant_ranks = []
    for rank, result in enumerate(results, start=1):
        timestamp = float(visual_metadata(result).get("start", 0.0))
        event = matching_event(timestamp, events)
        if event is not None:
            matched_events.add(event)
            relevant_ranks.append(rank)
    return {
        "hits": len(relevant_ranks),
        "precision_at_5": len(relevant_ranks) / 5,
        "event_recall_at_5": len(matched_events) / len(events),
        "reciprocal_rank": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
    }


def evaluate(
    *,
    chroma_path: Path,
    queries_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    queries = load_queries(queries_path)
    client = chromadb.PersistentClient(path=str(chroma_path))
    text_collection = client.get_collection("video_segments")
    siglip_collection = client.get_collection("video_frames_siglip")

    model_start = time.perf_counter()
    text_model = load_embedding_model()
    siglip_model, siglip_processor, siglip_device = load_siglip_model()
    model_load_time = time.perf_counter() - model_start
    query_texts = [item["query"] for item in queries]
    text_embeddings = text_model.encode(query_texts, normalize_embeddings=True)
    siglip_embeddings = embed_queries(
        query_texts,
        siglip_model,
        siglip_processor,
        siglip_device,
    )

    report: dict[str, Any] = {
        "chroma_path": str(chroma_path),
        "queries_path": str(queries_path),
        "query_count": len(queries),
        "model_load_seconds": round(model_load_time, 3),
        "modes": {},
    }
    for mode_name, mode in MODES.items():
        evaluations = []
        search_times = []
        for index, query_data in enumerate(queries):
            text_weight, siglip_weight = modality_weights(query_data["query"])
            search_start = time.perf_counter()
            results = hybrid_search(
                text_collection=text_collection,
                text_embedding=text_embeddings[index],
                siglip_collection=siglip_collection,
                siglip_embedding=siglip_embeddings[index],
                results=5,
                candidates=20,
                text_weight=text_weight,
                siglip_weight=siglip_weight,
                **mode,
            )
            search_times.append(time.perf_counter() - search_start)
            score = score_results(results, query_data["events"])
            evaluations.append(
                {
                    "query": query_data["query"],
                    "events": query_data["events"],
                    **score,
                    "result_timestamps": [
                        round(float(visual_metadata(item).get("start", 0.0)), 3)
                        for item in results
                    ],
                }
            )
        report["modes"][mode_name] = {
            "precision_at_5": round(
                mean(item["precision_at_5"] for item in evaluations), 4
            ),
            "event_recall_at_5": round(
                mean(item["event_recall_at_5"] for item in evaluations), 4
            ),
            "mrr": round(
                mean(item["reciprocal_rank"] for item in evaluations), 4
            ),
            "average_search_seconds": round(mean(search_times), 4),
            "queries": evaluations,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara objetivamente los modos de recuperación del video."
    )
    parser.add_argument(
        "--chroma",
        type=Path,
        default=Path("data/chroma-benchmark-20min-siglip"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("evaluation/retrieval_queries.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/retrieval_report.json"),
    )
    args = parser.parse_args()
    report = evaluate(
        chroma_path=args.chroma,
        queries_path=args.queries,
        output_path=args.output,
    )
    print("=" * 72)
    print("COMPARACIÓN DE RECUPERACIÓN")
    print("=" * 72)
    for mode, result in report["modes"].items():
        print(
            f"{mode:14} "
            f"P@5={result['precision_at_5']:.3f}  "
            f"Recall eventos@5={result['event_recall_at_5']:.3f}  "
            f"MRR={result['mrr']:.3f}  "
            f"latencia={result['average_search_seconds']:.3f}s"
        )
    print(f"Reporte: {args.output}")


if __name__ == "__main__":
    main()
