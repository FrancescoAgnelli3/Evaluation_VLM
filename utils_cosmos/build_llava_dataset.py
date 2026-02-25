#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    prompt_path = Path("/home/fa/projects/Evaluation_VLM/FT_Cosmos/prompts/prompt_json.txt")
    ann_dir = Path("/opt/dataset/train_dataset_17k_json")
    media_dir = Path("/opt/dataset/train_dataset_17k")
    out_dir = Path("/opt/dataset/train_dataset_json/llava_format")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    user_value = "<video>\n" + prompt_text

    samples = []
    missing_media = []
    invalid_json = []

    for ann_path in sorted(ann_dir.glob("*.json")):
        base = ann_path.stem
        video_name = base + ".mp4"
        video_path = media_dir / video_name
        if not video_path.exists():
            missing_media.append(video_name)
            continue

        try:
            obj = json.loads(ann_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            invalid_json.append(str(ann_path))
            continue

        response_text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

        samples.append(
            {
                "id": base,
                "video": video_name,
                "images": [],
                "conversations": [
                    {"from": "human", "value": user_value},
                    {"from": "gpt", "value": response_text},
                ],
            }
        )

    out_path = out_dir / "llava_train.json"
    out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(samples)} samples to {out_path}")
    if missing_media:
        print(f"Warning: {len(missing_media)} annotations skipped due to missing videos.")
        print("First 10 missing:", missing_media[:10])
    if invalid_json:
        print(f"Warning: {len(invalid_json)} annotations skipped due to invalid JSON.")
        print("First 5 invalid:", invalid_json[:5])


if __name__ == "__main__":
    main()
