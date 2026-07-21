import argparse
import json
import time
from pathlib import Path

from ollama import chat


PROMPT = """
Eres un sistema de indexación semántica de videos.

Analiza la imagen y responde EXCLUSIVAMENTE con un objeto JSON válido.

No escribas explicaciones.
No escribas texto fuera del JSON.
No uses bloques Markdown.
No uses ```json.
Todos los valores deben estar en español.

Usa exactamente esta estructura:

{
  "scene": "",
  "setting": "",
  "people": [
    {
      "role": "",
      "description": ""
    }
  ],
  "objects": [],
  "actions": [],
  "visible_text": [],
  "organizations": [],
  "topics": [],
  "visual_elements": [],
  "description": ""
}

Reglas generales:

- Describe únicamente información visible u objetivamente inferible.
- No inventes nombres, identidades, lugares ni hechos.
- No identifiques personas por su nombre.
- Si un rol es evidente, utilízalo: presentador, periodista, entrevistador, entrevistado, jugador, árbitro, médico, docente, músico, policía, funcionario, trabajador, cliente, etc.
- Si el rol no puede determinarse, utiliza "persona".
- Si una lista no tiene elementos, devuelve [].
- No devuelvas objetos dentro de las listas. Las listas deben contener solamente textos.
- No escribas valores en inglés.
- No utilices expresiones como "posiblemente", "parece ser" o "podría ser".
- No describas colores de ropa salvo que ayuden a distinguir equipos, uniformes o elementos relevantes.

Definición de los campos:

- "scene": tipo general de escena, por ejemplo transmisión televisiva, entrevista, reunión, paisaje, evento, clase, fábrica, actuación, partido, publicidad.
- "setting": lugar o entorno observable.
- "people": personas visibles, con su rol y una descripción breve.
- "objects": objetos relevantes para buscar posteriormente la escena.
- "actions": acciones observables.
- "visible_text": texto realmente visible en la imagen.
- "organizations": marcas, empresas, instituciones, canales, equipos u organizaciones visibles.
- "topics": temas generales de la escena.
- "visual_elements": elementos gráficos como logotipos, códigos QR, banners, subtítulos, marcadores, mapas, gráficos o pantalla dividida.
- "description": resumen breve, objetivo y completo de la imagen.

Ejemplo de formato válido:

{
  "scene": "Transmisión televisiva",
  "setting": "Estudio de televisión con una pantalla de fondo",
  "people": [
    {
      "role": "presentador",
      "description": "Persona hablando frente a cámara"
    }
  ],
  "objects": [
    "micrófono",
    "pantalla"
  ],
  "actions": [
    "hablar frente a cámara"
  ],
  "visible_text": [
    "Noticias"
  ],
  "organizations": [
    "Canal de televisión"
  ],
  "topics": [
    "informativo",
    "actualidad"
  ],
  "visual_elements": [
    "logotipo",
    "banner",
    "pantalla de fondo"
  ],
  "description": "Programa informativo con un presentador hablando frente a cámara."
}

El ejemplo solo muestra el formato. Adapta todos los valores a la imagen real.

Responde únicamente con el JSON.
""".strip()

def describe_frame(
    image_path: Path,
    model: str,
) -> tuple[dict[str, object], float]:
    start_time = time.perf_counter()

    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
                "images": [str(image_path)],
            }
        ],
        options={
            "temperature": 0,
        },
    )

    processing_time = time.perf_counter() - start_time
    raw_content = response.message.content.strip()

    # Algunos modelos envuelven el JSON en bloques Markdown.
    if raw_content.startswith("```"):
        lines = raw_content.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw_content = "\n".join(lines).strip()

    try:
        analysis = json.loads(raw_content)
    except json.JSONDecodeError as error:
        raise ValueError(
            "El modelo no devolvió un JSON válido. "
            f"Respuesta recibida: {raw_content}"
        ) from error

    if not isinstance(analysis, dict):
        raise ValueError(
            "La respuesta del modelo debe ser un objeto JSON."
        )

    # Campos de texto simple.
    for field_name in ("scene", "setting", "description"):
        value = analysis.get(field_name)

        if not isinstance(value, str):
            analysis[field_name] = ""
        else:
            analysis[field_name] = value.strip()

    # Campos que deben contener listas de textos.
    for field_name in (
        "objects",
        "actions",
        "visible_text",
        "organizations",
        "topics",
        "visual_elements",
    ):
        value = analysis.get(field_name)

        if not isinstance(value, list):
            analysis[field_name] = []
            continue

        analysis[field_name] = [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    # El campo people contiene objetos con role y description.
    people = analysis.get("people")

    if not isinstance(people, list):
        analysis["people"] = []
    else:
        normalized_people = []

        for person in people:
            if not isinstance(person, dict):
                continue

            role = person.get("role", "")
            description = person.get("description", "")

            normalized_people.append(
                {
                    "role": str(role).strip(),
                    "description": str(description).strip(),
                }
            )

        analysis["people"] = normalized_people

    return analysis, processing_time


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def process_frames(
    frames_json_path: Path,
    frames_dir: Path,
    output_path: Path,
    model: str = "qwen2.5vl:3b",
    limit: int | None = None,
    frame: int | None = None,
) -> dict[str, object]:
    """Analyze video frames and write the existing visual segment JSON files."""
    frames_json_path = Path(frames_json_path)
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)

    if not frames_json_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de metadatos: {frames_json_path}"
        )

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio de frames: {frames_dir}"
        )

    with frames_json_path.open("r", encoding="utf-8") as file:
        frames_data = json.load(file)

    frames = frames_data.get("frames")

    if not isinstance(frames, list):
        raise ValueError(
            "El archivo frames.json no contiene una lista válida en la clave 'frames'."
        )

    if frame is not None:
        target_filename = f"frame_{frame:03d}.jpg"
        frames = [item for item in frames if item.get("file") == target_filename]

        if not frames:
            raise ValueError(
                f"No se encontró el frame solicitado: {target_filename}"
            )
    elif limit is not None:
        if limit <= 0:
            raise ValueError("--limit debe ser mayor que cero.")

        frames = frames[:limit]

    total_frames = len(frames)

    if total_frames == 0:
        raise ValueError("No hay frames para procesar.")

    print(f"Modelo        : {model}")
    print(f"Frames        : {total_frames}")
    print(f"Directorio    : {frames_dir}")
    print(f"Salida        : {output_path}")
    print()

    visual_segments = []
    processing_times = []
    errors = []
    experiment_start = time.perf_counter()

    for position, frame_data in enumerate(frames, start=1):
        filename = frame_data.get("file")
        image_path = frames_dir / str(filename)
        timestamp = frame_data.get("timestamp")
        timestamp_sec = frame_data.get("timestamp_sec")
        selection_reason = frame_data.get("selection_reason")

        print(
            f"[{position:03d}/{total_frames:03d}] {filename} — {timestamp}",
            flush=True,
        )

        if not image_path.exists():
            error_message = f"No existe la imagen: {image_path}"
            print(f"  ERROR: {error_message}")
            errors.append({"frame": filename, "error": error_message})
            continue

        try:
            analysis, processing_time = describe_frame(image_path, model)
            processing_times.append(processing_time)
            people_text = []

            for person in analysis["people"]:
                role = person.get("role", "").strip()
                person_description = person.get("description", "").strip()

                if role and person_description:
                    people_text.append(f"{role}: {person_description}")
                elif role:
                    people_text.append(role)
                elif person_description:
                    people_text.append(person_description)

            text_parts = []
            text_fields = (
                ("scene", "Escena"),
                ("setting", "Entorno"),
            )
            for field_name, label in text_fields:
                if analysis[field_name]:
                    text_parts.append(f"{label}: {analysis[field_name]}.")

            if people_text:
                text_parts.append(f"Personas y roles: {', '.join(people_text)}.")

            list_fields = (
                ("objects", "Objetos"),
                ("actions", "Acciones"),
                ("visible_text", "Texto visible"),
                ("organizations", "Organizaciones y marcas"),
                ("topics", "Temas"),
                ("visual_elements", "Elementos visuales"),
            )
            for field_name, label in list_fields:
                if analysis[field_name]:
                    text_parts.append(f"{label}: {', '.join(analysis[field_name])}.")

            if analysis["description"]:
                text_parts.append(f"Descripción: {analysis['description']}")

            index_text = " ".join(text_parts).strip()
            if not index_text:
                raise ValueError("El modelo devolvió un análisis vacío.")

            segment = {
                "id": image_path.stem,
                "type": "vision",
                "source": model,
                "start": timestamp_sec,
                "end": timestamp_sec,
                "timestamp": timestamp,
                "timestamp_sec": timestamp_sec,
                "frame": filename,
                "frame_path": str(image_path),
                "selection_reason": selection_reason,
                "processing_time_seconds": round(processing_time, 3),
                "scene": analysis["scene"],
                "setting": analysis["setting"],
                "people": analysis["people"],
                "objects": analysis["objects"],
                "actions": analysis["actions"],
                "visible_text": analysis["visible_text"],
                "organizations": analysis["organizations"],
                "topics": analysis["topics"],
                "visual_elements": analysis["visual_elements"],
                "description": analysis["description"],
                "text": index_text,
            }
            visual_segments.append(segment)
            print(f"  Tiempo: {processing_time:.2f} s")
            print(f"  JSON  : {json.dumps(analysis, ensure_ascii=False)}")
            print(f"  Texto : {index_text}")
            print()
            save_json(output_path, visual_segments)
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            print(f"  ERROR: {error_message}")
            print()
            errors.append({"frame": filename, "error": error_message})

    total_time = time.perf_counter() - experiment_start
    successful_frames = len(visual_segments)
    average_time = (
        sum(processing_times) / len(processing_times)
        if processing_times
        else 0.0
    )
    minimum_time = min(processing_times) if processing_times else 0.0
    maximum_time = max(processing_times) if processing_times else 0.0
    frames_per_second = successful_frames / total_time if total_time > 0 else 0.0
    summary = {
        "model": model,
        "frames_requested": total_frames,
        "frames_processed": successful_frames,
        "frames_failed": len(errors),
        "total_processing_time_seconds": round(total_time, 3),
        "average_processing_time_seconds": round(average_time, 3),
        "minimum_processing_time_seconds": round(minimum_time, 3),
        "maximum_processing_time_seconds": round(maximum_time, 3),
        "frames_per_second": round(frames_per_second, 4),
        "errors": errors,
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    save_json(output_path, visual_segments)
    save_json(summary_path, summary)

    print("=" * 60)
    print("RESULTADOS DEL PROCESAMIENTO VISUAL")
    print("=" * 60)
    print(f"Frames solicitados : {total_frames}")
    print(f"Frames procesados  : {successful_frames}")
    print(f"Frames con error   : {len(errors)}")
    print(f"Tiempo total       : {total_time:.2f} s")
    print(f"Promedio por frame : {average_time:.2f} s")
    print(f"Tiempo mínimo      : {minimum_time:.2f} s")
    print(f"Tiempo máximo      : {maximum_time:.2f} s")
    print(f"Rendimiento        : {frames_per_second:.4f} frames/s")
    print(f"Segmentos visuales : {output_path}")
    print(f"Resumen            : {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza todos los frames de un video usando un modelo visual de Ollama."
    )

    parser.add_argument(
        "--frames-json",
        required=True,
        help="Ruta al archivo frames.json generado por claude-real-video.",
    )

    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directorio que contiene los archivos JPG.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Ruta del archivo visual_segments.json.",
    )

    parser.add_argument(
        "--model",
        default="qwen2.5vl:3b",
        help="Modelo visual disponible en Ollama.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesar solamente los primeros N frames.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Procesa únicamente el frame indicado (ej.: 27 -> frame_027.jpg).",
    )

    args = parser.parse_args()

    process_frames(
        frames_json_path=Path(args.frames_json),
        frames_dir=Path(args.frames_dir),
        output_path=Path(args.output),
        model=args.model,
        limit=args.limit,
        frame=args.frame,
    )


if __name__ == "__main__":
    main()
