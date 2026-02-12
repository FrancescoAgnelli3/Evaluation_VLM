# Evaluation_VLM

Video VLM evaluation harness for road-safety perception. It can:
- Fine-tune Qwen3-VL and Cosmos-Reason2 on video+JSON labels.
- Run video-only inference via vLLM/OpenAI-compatible APIs.
- Score model outputs against Gemini-generated gold standards.

## Repo layout

- `Evaluation/`: evaluation and inference code.
  - `answer_questions.py`: run VLM inference on videos + questions and save JSON outputs.
  - `eval.py`: compare model outputs against gold standards and produce CSV summaries.
  - `create_gold_standard_gemini.py`: generate Gemini perception JSON (teacher runs).
  - `aggregate_gold_standard.py`: aggregate multiple Gemini runs into a single target.
  - `render_answer_images.py`: render annotated images summarizing model outputs.
  - `vllm_utils.py`: vLLM server management + model registry.
- `FT_Qwen3/`: Qwen3-VL fine-tuning scripts (LoRA/QLoRA) + merge utilities.
- `FT_Cosmos/`: Cosmos-Reason2 fine-tuning scripts (LoRA/QLoRA) + merge utilities.
- `demos/`: sample videos + `questions.json`.
- `results/`: model outputs (student).
- `results_gold/`: Gemini outputs (teacher).
- `eval_out/`: evaluation summaries.

## Requirements

- Python 3.9+.
- GPU + local vLLM install for running model inference.
- There are two distinct training environments:
  - Qwen3-VL fine-tuning env.
  - Cosmos-Reason2 fine-tuning env.
- The Cosmos training env is also used for evaluation.

Install with the provided requirement files:

```bash
python3 -m pip install -r requirements_qwen.txt
python3 -m pip install -r requirements_cosmos.txt
python3 -m pip install -r requirements_eval.txt
```

## Fine-tuning Qwen3-VL

Scripts live in `FT_Qwen3/`. The training script expects a video directory and
matching JSON labels (same basename) or a `metadata.jsonl` file.

1) Prepare data

- Videos in `VIDEO_DIR` and JSON labels in `JSON_DIR`.
- Alternatively, put a `metadata.jsonl` in `JSON_DIR` with lines like:
  `{ "video": "relative_or_abs_path.mp4", "label": "{...json string...}" }`

2) Edit defaults (optional)

`FT_Qwen3/train_qwen3vl_video_json.py` has default paths:
- `VIDEO_DIR = /mnt/ssd1/dataset_ft_VLM/dataset_train_subset_1000`
- `JSON_DIR = /mnt/ssd1/dataset_ft_VLM/dataset_train_json_subset_1000`
- `PROMPT_DIR = FT_Qwen3/prompts/prompt_json.txt`
- `OUTPUT_DIR = /mnt/ssd1/Qwen3-32B-FT/ft_both/Qwen_FT_adapter/`

Update those or pass CLI flags (see `--help`).

3) Train + merge

Single run:

```bash
python3 FT_Qwen3/train_qwen3vl_video_json.py \
  --tune both \
  --output_dir /mnt/ssd1/Qwen3-32B-FT/ft_both_1k/Qwen_FT_adapter \
  --use_qlora

python3 FT_Qwen3/merge_weights.py \
  --adapter_dir /mnt/ssd1/Qwen3-32B-FT/ft_both_1k/Qwen_FT_adapter \
  --output_dir /mnt/ssd1/Qwen3-32B-FT/ft_both_1k/Qwen_FT_merged
```

Or use the helper script (edits in-file):

```bash
bash FT_Qwen3/run_train_and_merge_all.sh
```

## Fine-tuning Cosmos-Reason2

Scripts live in `FT_Cosmos/`. This training script uses TRL SFTTrainer and has
an internal multi-process launcher when multiple GPUs are available.

1) Prepare data

- Videos in `VIDEO_DIR` and JSON labels in `JSON_DIR`.
- Same `metadata.jsonl` option as Qwen3-VL.

Default paths in `FT_Cosmos/train_qwen3vl_video_json.py`:
- `VIDEO_DIR = /opt/dataset/train_dataset`
- `JSON_DIR = /opt/dataset/train_dataset_json`
- `PROMPT_DIR = FT_Cosmos/prompts/prompt_json.txt`
- `OUTPUT_DIR = /opt/models/Cosmos-Reason2-FT/adapter/`

2) Train

```bash
bash FT_Cosmos/run.sh \
  --base_model nvidia/Cosmos-Reason2-8B \
  --output_dir /opt/models/Cosmos-Reason2-FT/adapter \
  --use_qlora
```

3) Merge LoRA adapter

```bash
python3 FT_Cosmos/merge_weights.py \
  --adapter_dir /opt/models/Cosmos-Reason2-FT/adapter \
  --output_dir /opt/models/Cosmos-Reason2-FT/LoRA/merged
```

## Inference (vLLM)

`Evaluation/answer_questions.py` auto-starts vLLM via `Evaluation/vllm_utils.py`.
Model keys are mapped to HF repos or local paths in that file.

Example:

```bash
python3 Evaluation/answer_questions.py \
  --media-dir Evaluation/demos \
  --questions Evaluation/demos/questions.json \
  --output-dir Evaluation/results \
  --model cosmos2-8B
```

Common model keys (see `Evaluation/vllm_utils.py` for the full list):
- `qwen-2B`, `qwen-8B`, `qwen-32B`
- `cosmos1`, `cosmos2-2B`, `cosmos2-8B`
- fine-tuned keys like `cosmos2-reason-LoRAFT`, `cosmos2-reason-fullFT`

Useful env vars:
- `VLLM_HOST`, `VLLM_PORT`: vLLM server bind address (default host `127.0.0.1`).
- `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`: vLLM sizing.
- `DEFAULT_MODEL`: default model key if `--model` omitted.
- `QWEN_*_REPO`, `COSMOS_*_REPO`: override model repo paths for fine-tuned weights.

## Evaluation

1) Generate gold standards with Gemini

```bash
python3 Evaluation/create_gold_standard_gemini.py \
  --videos-dir Evaluation/demos \
  --out-dir Evaluation/results_gold \
  --runs 5 \
  --model gemini-flash-latest
```

Requires a valid API key in `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

2) Aggregate teacher runs

```bash
python3 Evaluation/aggregate_gold_standard.py \
  --in-dir Evaluation/results_gold \
  --min-valid 3 \
  --write-risks
```

3) Score student outputs

```bash
python3 Evaluation/eval.py \
  --results Evaluation/results \
  --results-gold Evaluation/results_gold \
  --out Evaluation/eval_out
```

Outputs:
- `Evaluation/eval_out/per_video_scores.csv`
- `Evaluation/eval_out/model_summary.csv`
- `Evaluation/eval_out/details.json`

## Notes

- `answer_questions.py` writes per-video JSON outputs named like:
  `<question_id>_<video_stem>_<model>.json`.
- vLLM is started/stopped automatically by `vllm_utils.py`.
