#!/usr/bin/env python3
"""
Converte un dataset composto da video + annotazioni JSON in un file JSONL
compatibile con ms-swift per il finetuning di Qwen3-VL.

Struttura attesa:
    dataset_dir/
    ├── videos/          # cartella con i file .mp4
    │   ├── sample_001.mp4
    │   └── ...
    ├── annotations/     # cartella con i file .json (stesso nome dei video)
    │   ├── sample_001.json
    │   └── ...
    └── prompt.txt       # prompt fisso per tutti i campioni

Formato output (JSONL, una riga per campione):
    {"messages": [{"role": "user", "content": "<video>Il tuo prompt qui"},
                  {"role": "assistant", "content": "...risposta JSON..."}],
     "videos": ["/path/assoluto/al/video.mp4"]}

Uso:
    python build_swift_dataset.py --dataset_dir /path/to/dataset
    python build_swift_dataset.py --dataset_dir /path/to/dataset --output train.jsonl
    python build_swift_dataset.py --dataset_dir /path/to/dataset --videos_folder clips --annotations_folder labels
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_prompt(prompt_path: Path) -> str:
    """Carica il prompt dal file .txt."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        raise ValueError(f"Il file prompt è vuoto: {prompt_path}")
    return prompt


def load_annotation(annotation_path: Path) -> str:
    """
    Carica l'annotazione JSON e la restituisce come stringa formattata.
    L'intero contenuto del file JSON viene usato come risposta dell'assistant.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Restituisce il JSON formattato su più righe (come nell'originale)
    return json.dumps(data, ensure_ascii=False, indent=2)


def find_matching_pairs(videos_dir: Path, annotations_dir: Path) -> list[tuple[Path, Path]]:
    """
    Trova le coppie (video, annotazione) con lo stesso nome base.
    Es: sample_001.mp4 <-> sample_001.json
    """
    # Indicizza i video per nome (senza estensione)
    video_files = {}
    for f in videos_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            video_files[f.stem] = f

    # Indicizza le annotazioni per nome (senza estensione)
    annotation_files = {}
    for f in annotations_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".json":
            annotation_files[f.stem] = f

    # Trova le coppie
    pairs = []
    matched_names = sorted(set(video_files.keys()) & set(annotation_files.keys()))
    orphan_videos = sorted(set(video_files.keys()) - set(annotation_files.keys()))
    orphan_annotations = sorted(set(annotation_files.keys()) - set(video_files.keys()))

    for name in matched_names:
        pairs.append((video_files[name], annotation_files[name]))

    # Report
    if orphan_videos:
        print(f"⚠️  {len(orphan_videos)} video senza annotazione: {orphan_videos[:5]}{'...' if len(orphan_videos) > 5 else ''}")
    if orphan_annotations:
        print(f"⚠️  {len(orphan_annotations)} annotazioni senza video: {orphan_annotations[:5]}{'...' if len(orphan_annotations) > 5 else ''}")

    return pairs


def build_jsonl(
    pairs: list[tuple[Path, Path]],
    prompt: str,
    output_path: Path,
    use_absolute_paths: bool = True,
) -> int:
    """
    Costruisce il file JSONL compatibile con ms-swift.
    Restituisce il numero di campioni scritti.
    """
    count = 0
    errors = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for video_path, annotation_path in pairs:
            try:
                # Carica la risposta
                assistant_response = load_annotation(annotation_path)

                # Path del video (assoluto o relativo)
                video_str = str(video_path.resolve()) if use_absolute_paths else str(video_path)

                # Costruisci il record nel formato swift
                record = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"<video>{prompt}"
                        },
                        {
                            "role": "assistant",
                            "content": assistant_response
                        }
                    ],
                    "videos": [video_str]
                }

                # Scrivi come singola riga JSON
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

            except Exception as e:
                errors += 1
                print(f"❌ Errore con {video_path.name}: {e}")

    if errors:
        print(f"\n⚠️  {errors} campioni saltati per errori.")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Converte dataset video+annotazioni in JSONL per ms-swift"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Cartella principale del dataset"
    )
    parser.add_argument(
        "--videos_folder",
        type=str,
        default="videos",
        help="Nome della sottocartella video (default: 'videos')"
    )
    parser.add_argument(
        "--annotations_folder",
        type=str,
        default="annotations",
        help="Nome della sottocartella annotazioni (default: 'annotations')"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path al file .txt con il prompt (può essere ovunque nel filesystem)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path del file JSONL di output (default: dataset_dir/train.jsonl)"
    )
    parser.add_argument(
        "--relative_paths",
        action="store_true",
        help="Usa path relativi per i video invece di assoluti"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    videos_dir = dataset_dir / args.videos_folder
    annotations_dir = dataset_dir / args.annotations_folder
    prompt_path = Path(args.prompt_file)
    output_path = Path(args.output) if args.output else dataset_dir / "train.jsonl"

    # Validazione
    if not dataset_dir.is_dir():
        sys.exit(f"❌ Cartella dataset non trovata: {dataset_dir}")
    if not videos_dir.is_dir():
        sys.exit(f"❌ Cartella video non trovata: {videos_dir}")
    if not annotations_dir.is_dir():
        sys.exit(f"❌ Cartella annotazioni non trovata: {annotations_dir}")
    if not prompt_path.is_file():
        sys.exit(f"❌ File prompt non trovato: {prompt_path}")

    # Carica il prompt
    prompt = load_prompt(prompt_path)
    print(f"📝 Prompt caricato ({len(prompt)} caratteri):")
    print(f"   \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")

    # Trova le coppie video-annotazione
    pairs = find_matching_pairs(videos_dir, annotations_dir)
    print(f"\n🎬 Trovate {len(pairs)} coppie video-annotazione")

    if not pairs:
        sys.exit("❌ Nessuna coppia trovata. Verifica che i nomi dei file corrispondano.")

    # Costruisci il JSONL
    count = build_jsonl(
        pairs=pairs,
        prompt=prompt,
        output_path=output_path,
        use_absolute_paths=not args.relative_paths,
    )

    print(f"\n✅ Dataset creato: {output_path}")
    print(f"   {count} campioni scritti")
    print(f"\n💡 Per il training con swift:")
    print(f"   swift sft --model... --dataset {output_path}")


if __name__ == "__main__":
    main()

# python dataset_converter_toswift.py   --dataset_dir D:/WORK_K2K_2/QWEN3.5_FT --videos_folder DS_VID --annotations_folder DS_JSON --prompt_file D:/WORK_K2K_2/QWEN3.5_FT/prompt.txt --output D:/WORK_K2K_2/QWEN3.5_FT/train.jsonl
