import argparse
import time

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_CHROMA_PATH = "data/chroma-test-60s"
DEFAULT_COLLECTION_NAME = "video_segments"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Búsqueda semántica sobre segmentos indexados."
    )

    parser.add_argument(
        "query",
        help="Consulta en lenguaje natural.",
    )

    parser.add_argument(
        "--results",
        type=int,
        default=5,
        help="Cantidad de resultados.",
    )

    parser.add_argument(
        "--chroma",
        default=DEFAULT_CHROMA_PATH,
        help="Directorio de ChromaDB.",
    )

    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Colección.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Modelo de embeddings.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BÚSQUEDA SEMÁNTICA")
    print("=" * 70)
    print(f"Consulta     : {args.query}")
    print(f"ChromaDB     : {args.chroma}")
    print(f"Colección    : {args.collection}")
    print()

    print("Cargando modelo...")

    start = time.perf_counter()

    model = SentenceTransformer(args.model)

    model_load = time.perf_counter() - start

    print(f"Modelo cargado en {model_load:.3f} s")

    embedding_start = time.perf_counter()

    query_embedding = model.encode(
        [args.query],
        normalize_embeddings=True,
    )[0]

    embedding_time = time.perf_counter() - embedding_start

    client = chromadb.PersistentClient(path=args.chroma)

    collection = client.get_collection(args.collection)

    result_count = min(args.results, collection.count())

    search_start = time.perf_counter()

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    search_time = time.perf_counter() - search_start

    print(f"Embedding consulta : {embedding_time:.3f} s")
    print(f"Búsqueda Chroma    : {search_time:.3f} s")
    print()

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        print("No se encontraron resultados.")
        return

    for index, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        similarity = 1 - distance

        print("=" * 70)
        print(f"Resultado #{index}")
        print("=" * 70)

        print(f"Similitud : {similarity:.4f}")

        print(f"Tipo      : {metadata.get('type','-')}")
        print(f"Origen    : {metadata.get('source','-')}")

        print(
            f"Tiempo    : "
            f"{metadata.get('start')} → {metadata.get('end')}"
        )

        if "timestamp" in metadata:
            print(f"Timestamp : {metadata['timestamp']}")

        if "frame" in metadata:
            print(f"Frame     : {metadata['frame']}")

        print()
        print(document)
        print()

    print("=" * 70)
    print(f"Resultados encontrados : {len(documents)}")
    print("=" * 70)


if __name__ == "__main__":
    main()