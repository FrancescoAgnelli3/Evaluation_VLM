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
- The Cosmos training env can also be used for evaluation.

### uv venv setup (recommended)

Use separate local venvs per folder so dependencies don’t collide. Example with `uv`:

Qwen3-VL fine-tuning:

```bash
cd Evaluation_VLM/FT_Qwen3
uv venv .venv
source .venv/bin/activate
uv pip install -r ../requirements_qwen.txt
```

Cosmos-Reason2 fine-tuning:

```bash
cd Evaluation_VLM/FT_Cosmos
uv venv .venv
source .venv/bin/activate
uv pip install -r ../requirements_cosmos.txt
```

Evaluation:

```bash
cd Evaluation_VLM/Evaluation
uv venv .venv
source .venv/bin/activate
uv pip install -r ../requirements_eval.txt
```

## Fine-tuning Qwen3-VL

Scripts live in `FT_Qwen3/`. The training script expects a video directory and
matching JSON labels (same basename) or a `metadata.jsonl` file.

1) Prepare data

- Videos in `VIDEO_DIR` and JSON labels in `JSON_DIR`.

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

python3 FT_Qwen3/merge_weights.py \
  --adapter_dir /mnt/ssd1/Qwen3-32B-FT/ft_both_1k/Qwen_FT_adapter \
  --output_dir /mnt/ssd1/Qwen3-32B-FT/ft_both_1k/Qwen_FT_merged
```

Or use the helper script (edits in-file):

```bash
bash FT_Qwen3/run_train_and_merge_all.sh
```

For MoE LoRA training with multi-GPU torchrun defaults, use `FT_Qwen3/run.sh`:

```bash
NUM_GPUS=4 bash FT_Qwen3/run.sh \
  --model_id Qwen/Qwen3.5-397B-A17B-FP8 \
  --video_dir /opt/dataset/train_dataset_17k \
  --json_dir /opt/dataset/train_dataset_17k_json \
  --tune both \
  --output_dir /opt/models/Qwen3-MoE-FT/adapter \
  --use_qlora
```

## Fine-tuning Cosmos-Reason2

Scripts live in `FT_Cosmos/`. This training script uses TRL SFTTrainer and has
an internal multi-process launcher when multiple GPUs are available.

### Full fine-tuning Cosmos (Cosmos-Reason2 repo)

1) Clone the upstream repo

```bash
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
```

2) Copy our config into the Cosmos repo

```bash
cp Evaluation_VLM/utils_cosmos/my_sft_8gpu.toml \
  cosmos-reason2/examples/cosmos_rl/configs/
```

3) Edit the TOML as needed (model, dataset, output)

Update `cosmos-reason2/examples/cosmos_rl/configs/my_sft_8gpu.toml` to point to
your base model, dataset, and output folder.

4) Launch full FT (8 GPUs)

```bash
cd cosmos-reason2
source .venv/bin/activate
cd examples/cosmos_rl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  uv run cosmos-rl --config configs/my_sft_8gpu.toml \
  --log-dir outputs/my_sft_8gpu scripts/llava_sft.py
```

### LoRA fine-tuning Cosmos

1) Prepare data

- Videos in `VIDEO_DIR` and JSON labels in `JSON_DIR`.

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
  --output-dir Evaluation/results \
  --model all
```


## Evaluation

1) Score student outputs

```bash
python3 Evaluation/eval.py \
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
