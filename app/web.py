import chromadb
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "video_segments"


def timestamp_to_seconds(timestamp: str) -> float:
    parts = timestamp.split(":")

    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"Timestamp inválido: {timestamp}")


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_collection(chroma_path: str):
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_collection(COLLECTION_NAME)


st.set_page_config(page_title="Video Semantic Search")

st.title("🎥 Video Semantic Search")

# Buscar automáticamente todas las bases Chroma disponibles
bases = sorted(Path("data").glob("chroma*"))

if not bases:
    st.error("No se encontraron bases Chroma.")
    st.stop()

selected = st.sidebar.selectbox(
    "Experimento",
    bases,
    format_func=lambda p: p.name,
)

collection = load_collection(str(selected))
model = load_model()

query = st.text_input("Buscar")

if query:

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    st.caption(f"Base seleccionada: {selected.name}")

    for index, (doc, meta, dist) in enumerate(
        zip(docs, metas, dists),
        start=1,
    ):

        similarity = 1 - dist

        start_seconds = timestamp_to_seconds(meta["start"])
        end_seconds = timestamp_to_seconds(meta["end"])

        with st.container():

            st.subheader(f"Resultado {index}")

            st.write(f"**Video:** {meta['video_id']}")
            st.write(f"**Tiempo:** {meta['start']} → {meta['end']}")
            st.write(f"**Similitud:** {similarity:.3f}")

            st.progress(max(0.0, min(similarity, 1.0)))

            st.write(doc)

            st.video(
                meta["video_path"],
                start_time=int(start_seconds),
                end_time=int(end_seconds),
            )

            st.divider()