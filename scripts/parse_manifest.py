import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(
    r'^\[(\d{2}:\d{2}\.\d)-(\d{2}:\d{2}\.\d)\] 「(.+)」$'
)


def parse_manifest(manifest_path: Path) -> list[dict]:
    segments = []

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        match = PATTERN.match(line.strip())

        if not match:
            continue

        start, end, text = match.groups()

        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    return segments


def save_segments(segments: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            segments,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)

    return parser.parse_args()


def main():
    args = parse_arguments()

    manifest = Path(args.manifest)
    output = Path(args.output)

    segments = parse_manifest(manifest)

    save_segments(segments, output)

    print(f"Segmentos encontrados : {len(segments)}")
    print(f"Archivo generado      : {output}")


if __name__ == "__main__":
    main()