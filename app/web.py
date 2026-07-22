import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

from app.embed import load_embedding_model
from app.retrieval import (
    enrich_frame,
    hybrid_search,
    index_enrichment,
    load_enrichment,
    modality_weights,
)
from app.siglip import (
    DEFAULT_SIGLIP_COLLECTION,
    DEFAULT_SIGLIP_MODEL,
    embed_queries,
    load_siglip_model,
)
from app.vision import FAST_VISION_MODEL


MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "video_segments"


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def timestamp_to_seconds(timestamp: str | int | float) -> float:
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    parts = str(timestamp).split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Timestamp inválido: {timestamp}")


def format_timestamp(value: str | int | float) -> str:
    total_seconds = timestamp_to_seconds(value)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours >= 1:
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:04.1f}"
    return f"{int(minutes):02d}:{seconds:04.1f}"


@st.cache_resource(max_entries=2)
def load_text_model() -> SentenceTransformer:
    return load_embedding_model(MODEL_NAME)


@st.cache_resource(max_entries=4)
def load_collections(chroma_path: str):
    client = chromadb.PersistentClient(path=chroma_path)
    names = {collection.name for collection in client.list_collections()}
    text_collection = client.get_collection(COLLECTION_NAME)
    siglip_collection = (
        client.get_collection(DEFAULT_SIGLIP_COLLECTION)
        if DEFAULT_SIGLIP_COLLECTION in names
        else None
    )
    return text_collection, siglip_collection


@st.cache_resource(max_entries=1)
def load_visual_search_model():
    return load_siglip_model(DEFAULT_SIGLIP_MODEL)


st.set_page_config(page_title="Video Semantic Search")
st.title("Video Semantic Search")
st.caption("Búsqueda híbrida en audio, OCR, descripciones e imágenes.")

st.session_state.setdefault("search_results", [])
st.session_state.setdefault("last_query", "")
st.session_state.setdefault("last_database", "")
st.session_state.setdefault("last_mode", "Completo")

bases = sorted((PROJECT_ROOT / "data").glob("chroma*"))
if not bases:
    st.error("No se encontraron bases Chroma.")
    st.stop()

selected = st.sidebar.selectbox(
    "Experimento",
    bases,
    format_func=lambda path: path.name,
)
text_collection, siglip_collection = load_collections(str(selected))
text_model = load_text_model()

if siglip_collection is None:
    st.sidebar.caption("Índice disponible: texto")
else:
    st.sidebar.caption("Índices disponibles: texto + imagen")

with st.form("hybrid_search", border=False):
    search_modes = (
        ["Solo SigLIP", "SigLIP + Qwen", "Completo"]
        if siglip_collection is not None
        else ["Completo"]
    )
    search_mode = st.segmented_control(
        "Modo de búsqueda",
        search_modes,
        default="Completo",
        required=True,
    )
    query = st.text_input("Buscar", placeholder="Ej.: entrega de un trofeo")
    submitted = st.form_submit_button("Buscar", icon=":material/search:")

if submitted and query.strip():
    with st.spinner("Buscando en todas las modalidades..."):
        use_text = search_mode != "Solo SigLIP"
        use_siglip = search_mode != "Completo" or siglip_collection is not None
        qwen_only = search_mode == "SigLIP + Qwen"
        text_embedding = None
        if use_text:
            text_embedding = text_model.encode(
                [query],
                normalize_embeddings=True,
            )[0]
        siglip_embedding = None
        if siglip_collection is not None:
            siglip_model, siglip_processor, siglip_device = (
                load_visual_search_model()
            )
            siglip_embedding = embed_queries(
                [query],
                siglip_model,
                siglip_processor,
                siglip_device,
            )[0]
        text_weight, siglip_weight = modality_weights(query)
        st.session_state.search_results = hybrid_search(
            text_collection=text_collection,
            text_embedding=text_embedding,
            siglip_collection=siglip_collection,
            siglip_embedding=siglip_embedding,
            results=5,
            candidates=20,
            text_weight=text_weight,
            siglip_weight=siglip_weight,
            use_text=use_text,
            use_siglip=use_siglip,
            text_where=(
                {"type": {"$in": ["vision", "vision_on_demand"]}}
                if qwen_only
                else None
            ),
            text_modality="Qwen" if qwen_only else "texto",
        )
        st.session_state.last_query = query
        st.session_state.last_database = str(selected)
        st.session_state.last_mode = search_mode

if st.session_state.last_database != str(selected):
    st.session_state.search_results = []

results = st.session_state.search_results
if results:
    st.caption(
        f"Base: {selected.name} · Modo: {st.session_state.last_mode} · "
        f"Consulta: {st.session_state.last_query}"
    )

for index, result in enumerate(results, start=1):
    metadata = result["metadata"]
    image_metadata = result.get("image_metadata")
    if image_metadata and image_metadata.get("frame_path"):
        image_metadata = dict(image_metadata)
        image_metadata["frame_path"] = str(
            resolve_project_path(str(image_metadata["frame_path"]))
        )
    start_seconds = timestamp_to_seconds(metadata.get("start", 0))
    end_seconds = timestamp_to_seconds(metadata.get("end", start_seconds))
    start_label = format_timestamp(start_seconds)
    end_label = format_timestamp(end_seconds)
    modalities = " + ".join(result["modalities"])

    with st.container(border=True):
        st.subheader(f"Resultado {index}")
        st.caption(f"{start_label} → {end_label} · Encontrado por {modalities}")

        cached_enrichment = None
        if image_metadata and image_metadata.get("frame_path"):
            cached_enrichment = load_enrichment(image_metadata["frame_path"])

        if cached_enrichment:
            st.write(cached_enrichment["text"])
            st.caption("Descripción enriquecida por Qwen y guardada en el índice.")
        else:
            st.write(result["document"])

        if image_metadata and image_metadata.get("frame_path"):
            frame_path = resolve_project_path(str(image_metadata["frame_path"]))
            if frame_path.exists():
                st.image(str(frame_path), caption="Frame encontrado por SigLIP")

        video_options = {"start_time": int(start_seconds)}
        if int(end_seconds) > int(start_seconds):
            video_options["end_time"] = int(end_seconds)
        video_path = metadata.get("video_path")
        if video_path:
            st.video(str(resolve_project_path(str(video_path))), **video_options)

        if image_metadata and not cached_enrichment:
            if st.button(
                "Analizar este momento con Qwen",
                key=f"enrich_{selected.name}_{index}_{image_metadata.get('frame')}",
                icon=":material/auto_awesome:",
            ):
                with st.spinner("Qwen está describiendo el frame..."):
                    segment = enrich_frame(image_metadata, FAST_VISION_MODEL)
                    index_enrichment(
                        segment=segment,
                        collection=text_collection,
                        embedding_model=text_model,
                        video_id=str(image_metadata.get("video_id", "video")),
                        video_path=str(image_metadata.get("video_path", "")),
                    )
                st.success("Descripción guardada para futuras búsquedas.")
                st.rerun()
