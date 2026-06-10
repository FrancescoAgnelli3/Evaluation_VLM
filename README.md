# Evaluation_VLM

Video VLM evaluation harness for road-safety perception. It can:
- Fine-tune Qwen3-VL, Cosmos-Reason2, and Cosmos3 on video+JSON labels.
- Run video-only inference via vLLM/OpenAI-compatible APIs.
- Score model outputs against Gemini-generated gold standards.

## Repo layout

- `Evaluation/`: evaluation and inference code.
  - `answer_questions.py`: run VLM inference on videos + questions and save JSON outputs.
  - `eval.py`: compare model outputs against gold standards and produce CSV summaries.
  - `create_gold_standard_gemini.py`: generate Gemini perception JSON (teacher runs).
  - `aggregate_gold_standard.py`: aggregate multiple Gemini runs into a single target.
  - `render_answer_images.py`: render annotated images summarizing model outputs.
  - `run_answer_questions_tasks.sh`: run `answer_questions.py` across multiple tasks/models.
  - `run_answer_questions_when_gpus_free.sh`: wait for idle GPUs, then run inference.
  - `vllm_utils.py`: vLLM server management + model registry.
  - `utils/models_utils.py`: model key -> repo/path + served-name registry.
  - `utils/serve_dataset.sh`: simple HTTP server for local datasets.
  - `prompts/`: task prompts (`prompt_road.txt`, `prompt_people.txt`, `prompt_environment.txt`, `prompt_industry.txt`).
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

## Fine-tuning Cosmos3

Cosmos3 full FT uses the upstream `cosmos-framework` repo rather than the
`FT_Cosmos/` scripts in this repository.

### Download / prepare Cosmos3 weights for FT

1) Clone the upstream repo

```bash
git clone https://github.com/NVIDIA/Cosmos-Framework.git cosmos-framework
cd cosmos-framework
```

2) Create the training environment

For CUDA 12.8:

```bash
uv sync --all-extras --group=cu128-train
source .venv/bin/activate
export LD_LIBRARY_PATH=
```

For CUDA 13.0, use `--group=cu130-train` instead.

3) Convert Cosmos3-Nano into a local VLM checkpoint

This produces the local weights directory used for Cosmos3 reasoner FT:

```bash
python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors \
  --checkpoint-path Cosmos3-Nano \
  -o examples/checkpoints/Cosmos3-Nano-VLM
```

After this step, the local FT-ready weights will be at:

```bash
cosmos-framework/examples/checkpoints/Cosmos3-Nano-VLM
```

### Use those weights for local Cosmos3 FT

In our local Cosmos3 reasoner setup, this checkpoint is used through:

- `examples/toml/sft_config/local_reasoner_sft.toml`
- `examples/launch_sft_local_reasoner.sh`

The base architecture remains `Qwen/Qwen3-VL-8B-Instruct`, while
`safetensors_path` points to the converted local `Cosmos3-Nano-VLM` weights.

### FT helper files copied in this repo

The Cosmos3 FT helper files are also mirrored here:

- `utils_cosmos/cosmos3/cosmos_framework/configs/base/vlm/experiment/local_reasoner_sft.py`
- `utils_cosmos/cosmos3/examples/toml/sft_config/local_reasoner_sft.toml`
- `utils_cosmos/cosmos3/examples/launch_sft_local_reasoner.sh`

If you want to reuse them in a fresh `cosmos-framework` clone, copy them back to
the same relative paths from the `Evaluation_VLM` repo root:

```bash
cp utils_cosmos/cosmos3/cosmos_framework/configs/base/vlm/experiment/local_reasoner_sft.py \
  /path/to/cosmos-framework/cosmos_framework/configs/base/vlm/experiment/

cp utils_cosmos/cosmos3/examples/toml/sft_config/local_reasoner_sft.toml \
  /path/to/cosmos-framework/examples/toml/sft_config/

cp utils_cosmos/cosmos3/examples/launch_sft_local_reasoner.sh \
  /path/to/cosmos-framework/examples/
```

Then launch from inside `cosmos-framework`:

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=
bash examples/launch_sft_local_reasoner.sh
```

## Dataset utility scripts

The following dataset-prep helpers are mirrored in `utils_cosmos/`:

- `utils_cosmos/symlink_by_folder_name.py`
- `utils_cosmos/split_dataset.py`
- `utils_cosmos/subsample_test_dataset.py`

### `symlink_by_folder_name.py`

Create flat symlinked train/test datasets by scanning folder names and linking:

- `.json` files from folders whose name contains `json`
- `.mp4` / `.mov` files from the other folders

Examples:

```bash
python3 utils_cosmos/symlink_by_folder_name.py --all_train
python3 utils_cosmos/symlink_by_folder_name.py --all_test
```

Custom source/destination example:

```bash
python3 utils_cosmos/symlink_by_folder_name.py \
  /path/to/src_a /path/to/src_b /path/to/dst
```

Useful defaults:

- `--ds_root ds_pulito`
- `--dst_json_dir ds_pulito/train_dataset_json`
- `--dst_mp4_dir ds_pulito/train_dataset`
- `--test_dst_json_dir ds_pulito/test_dataset_json`
- `--test_dst_mp4_dir ds_pulito/test_dataset`
- `--max_files N` to stop after a limited number of linked files

### `split_dataset.py`

Split matched `*.mp4` + `*.json` pairs into train/test folders by filename stem.
By default it creates symlinks; `--copy` copies and `--move` moves.

Example:

```bash
python3 utils_cosmos/split_dataset.py \
  --mp4-dir /opt/dataset/ds_people/Dataset_training_v1_batch_1 \
  --json-dir /opt/dataset/ds_people/Dataset_training_v1_batch_1_json_final_prompt \
  --out-dir /opt/dataset/ds_people \
  --split 0.8
```

This creates:

- `train_dataset/`
- `train_dataset_json/`
- `test_dataset/`
- `test_dataset_json/`

### `subsample_test_dataset.py`

Create a camera-balanced subset from `test_dataset/` and `test_dataset_json/`.
Camera ID is inferred from the leading number in each filename.

Example:

```bash
python3 utils_cosmos/subsample_test_dataset.py \
  --mp4-dir /opt/dataset/test_dataset \
  --json-dir /opt/dataset/test_dataset_json \
  --out-mp4-dir /opt/dataset/test_dataset_subsample \
  --out-json-dir /opt/dataset/test_dataset_subsample_json \
  --size 3000
```

By default it copies files; use `--move` if you want to move them instead.

## Build Cosmos full-FT dataset (LLaVA format)

Use `utils_cosmos/build_llava_dataset.py` to convert a folder of MP4s + matching
JSON annotations into a Cosmos full-FT ready LLaVA JSON dataset.

1) Ensure matching basenames

- Every `*.json` in your JSON folder must have a matching `*.mp4` in your video folder.
  Example: `clip_001.json` + `clip_001.mp4`.

2) Use existing dataset folders (already present)

If you already have the standard folders:

- MP4s: `/opt/dataset/train_dataset_17k`
- JSON: `/opt/dataset/train_dataset_17k_json`

You can run the script as-is (no edits needed).

3) Update paths in the script (if different)

Edit the top of `utils_cosmos/build_llava_dataset.py` to point at your local folders:

- `prompt_path`: prompt used for the LLaVA "human" message (default: `FT_Cosmos/prompts/prompt_json.txt`)
- `ann_dir`: your JSON folder
- `media_dir`: your MP4 folder
- `out_dir`: output folder for the LLaVA dataset

4) Run the converter

```bash
python3 utils_cosmos/build_llava_dataset.py
```

5) Output

The script writes `llava_train.json` under `out_dir` and reports any missing videos
or invalid JSON. This file is ready to use as the Cosmos full-FT dataset in
`cosmos-reason2/examples/cosmos_rl/scripts/llava_sft.py` configs.

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
- `PROMPT_DIR = FT_Cosmos/prompts/prompt_road.txt` (update via `--prompt_path`)
- `OUTPUT_DIR = /opt/models/Cosmos-Reason2-FT/adapter/`

2) Train

```bash
bash FT_Cosmos/run.sh \
  --base_model nvidia/Cosmos-Reason2-8B \
  --prompt_path FT_Cosmos/prompts/prompt_road.txt \
  --output_dir /opt/models/Cosmos-Reason2-FT/adapter \
```

3) Merge LoRA adapter

```bash
python3 FT_Cosmos/merge_weights.py \
  --adapter_dir /opt/models/Cosmos-Reason2-FT/adapter \
  --output_dir /opt/models/Cosmos-Reason2-FT/LoRA/merged
```

Tip: For task-specific LoRA runs, point `--prompt_path` at:
`FT_Cosmos/prompts/prompt_people.txt`, `prompt_environment.txt`, or `prompt_industry.txt`.

Helper for multi-task runs:

```bash
bash FT_Cosmos/launch_ft_and_merge.sh
```

## Inference (vLLM)

`Evaluation/answer_questions.py` auto-starts vLLM via `Evaluation/vllm_utils.py`.
Model keys are mapped to HF repos or local paths in that file.

Tasks: `road`, `people`, `environment`, `industry`.
Default media dirs:
- road: `/opt/dataset/test_dataset`
- people: `/opt/dataset/ds_people/test_dataset`
- environment: `/opt/dataset/ds_environment/test_dataset`
- industry: `/opt/dataset/ds_industry/test_dataset`

Example (road task):

```bash
python3 Evaluation/answer_questions.py \
  --task road \
  --model all
```

Example (people task):

```bash
python3 Evaluation/answer_questions.py \
  --task people \
  --model all
```

By default, outputs go to `Evaluation/results_{task}` unless you override with `--output-dir`.

Batch helpers:

```bash
bash Evaluation/run_answer_questions_tasks.sh
bash Evaluation/run_answer_questions_when_gpus_free.sh
```

### Serve videos over HTTP (local dataset)

If your evaluation environment expects videos via HTTP, use the helper script to
serve a local dataset folder and configure the URL env vars:

```bash
bash Evaluation_VLM/Evaluation/utils/serve_dataset.sh 8000 /opt/dataset

export VIDEO_USE_DATA_URL=0
export VIDEO_URL_ROOT=/opt/dataset
export VIDEO_URL_PREFIX=http://127.0.0.1:8000
```

### Add a new model choice (new repo/path)

To add a selectable model key (for `--model`), update `Evaluation/utils/models_utils.py`:

1) Add a repo/path variable (with env override)

Add a new `*_REPO` constant near the top, for example:
`MY_MODEL_REPO = os.environ.get("MY_MODEL_REPO", "/opt/models/MyModel")`

2) Register the new model key

Append your key to `MODEL_CHOICES`, e.g. `"my-model"`.

3) Map key -> served model name

Add a case in `served_name_for()` so vLLM knows the served name.

4) Map key -> repo/path

Add a case in `resolve_model_repo()` to return your `MY_MODEL_REPO`.

Then run inference with:

```bash
python3 Evaluation/answer_questions.py \
  --task road \
  --model my-model
```

Tip: You can avoid code edits by reusing an existing key and overriding its repo via env var,
e.g. `COSMOS3_REPO=/path/to/new/model`.

Current Cosmos3 wiring uses `cosmos3` as the model key and defaults to
`nvidia-cosmos-ea/Cosmos3-Super-Reasoner` via `COSMOS3_REPO`.

The reasoning variant `cosmos3-reason` uses the same repo but appends a
think-style instruction to the user prompt and strips any `<think>...`
section before saving the JSON response.

### Fine-tuning Cosmos3 on a local dataset

We also have a local Cosmos3 nano full-finetune export that can be used for
evaluation or further SFT work. The local evaluation key is `cosmos3-nano-fullFT`,
which maps to the HF-style export directory under:

```bash
/opt/models/dataset_clean/Cosmos3-FT/nano/local_reasoner_sft/hf_exports/iter_000000500/
```

That export contains the `config.json` and safetensors shards that vLLM needs.
The raw training checkpoint under `.../checkpoints/iter_000000500/` is *not*
servable by vLLM on its own.

To reproduce the local full-FT flow:

1) Train or resume the Cosmos3 local reasoner recipe in `cosmos-framework`
   using [`examples/toml/sft_config/local_reasoner_sft.toml`](../cosmos-framework/examples/toml/sft_config/local_reasoner_sft.toml) and
   [`examples/launch_sft_local_reasoner.sh`](../cosmos-framework/examples/launch_sft_local_reasoner.sh).

2) Point `VLM_SAFETENSORS_PATH` at the exported HF directory if you want to
   override the default at launch time:

```bash
export VLM_SAFETENSORS_PATH=/opt/models/dataset_clean/Cosmos3-FT/nano/local_reasoner_sft/hf_exports/iter_000000500
bash examples/launch_sft_local_reasoner.sh
```

3) Use the evaluation harness with:

```bash
python3 Evaluation/answer_questions.py \
  --task road \
  --model cosmos3-nano-fullFT
```


## Evaluation

1) Score student outputs (road task)

```bash
python3 Evaluation/eval.py \
  --task road
```

Outputs:
- `Evaluation/eval_out_road/per_video_scores.csv`
- `Evaluation/eval_out_road/model_summary.csv`
- `Evaluation/eval_out_road/details.json`

Other tasks:

```bash
python3 Evaluation/eval.py --task people
python3 Evaluation/eval.py --task environment
python3 Evaluation/eval.py --task industry
```

Defaults:
- road: `results_road` -> `eval_out_road`
- people: `results_people` -> `eval_out_people`
- environment: set with `--results` / `--out` (see `Evaluation/utils/eval_enviroment.py`)
- industry: `results_industry` -> `eval_out_industry`

## Notes

- `answer_questions.py` writes per-video JSON outputs named like:
  `<question_id>_<video_stem>_<model>.json`.
- vLLM is started/stopped automatically by `vllm_utils.py`.
