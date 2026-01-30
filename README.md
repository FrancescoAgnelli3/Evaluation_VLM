# Evaluation_VLM

Video VLM evaluation harness for road-safety perception. It runs video-only inference via
vLLM/OpenAI-compatible APIs, writes per-video JSON outputs, and scores them against
Gemini-generated gold standards.

## What is in here

- `answer_questions.py`: Run VLM inference on videos + questions and save JSON outputs.
- `eval.py`: Compare model outputs against gold standards and produce CSV summaries.
- `create_gold_standard_gemini.py`: Generate Gemini perception JSON (teacher runs).
- `aggregate_gold_standard.py`: Aggregate multiple Gemini runs into a single target.
- `render_answer_images.py`: Render annotated images summarizing model outputs.
- `demos/`: Sample videos + `questions.json`.
- `results/`: Model outputs (student).
- `results_gold/`: Gemini outputs (teacher).
- `eval_out/`: Evaluation summaries.
- `prompts/`: Prompt templates.

## Requirements

- Python 3.9+.
- Dependencies used by scripts:
  - `openai`, `requests`, `vllm` (for `answer_questions.py` + `vllm_utils.py`)
  - `numpy`, `pandas` (for `eval.py`)
  - `google-genai` (for `create_gold_standard_gemini.py`)
  - `Pillow` (for `render_answer_images.py`)
- GPU + local vLLM install for running model inference.

## Quick start

Run inference with vLLM-served models:

```bash
python answer_questions.py \
  --media-dir demos \
  --questions demos/questions.json \
  --output-dir results \
  --model cosmos2-2B
```

Model choices: `qwen-2B`, `qwen-8B`, `qwen-32B`, `cosmos1`, `cosmos2-2B`, `cosmos2-8B`, `all`.

Generate gold standards with Gemini:

```bash
python create_gold_standard_gemini.py \
  --videos-dir demos \
  --out-dir results_gold \
  --runs 5 \
  --model gemini-flash-latest
```

Aggregate teacher runs:

```bash
python aggregate_gold_standard.py --in-dir results_gold --min-valid 3 --write-risks
```

Evaluate model outputs:

```bash
python eval.py --results results --results-gold results_gold --out eval_out
```

Outputs:
- `eval_out/per_video_scores.csv`
- `eval_out/model_summary.csv`
- `eval_out/details.json`

## Notes

- vLLM is started/stopped automatically by `vllm_utils.py`. Configure host/port with
  `VLLM_HOST` and `VLLM_PORT` if needed.
- `answer_questions.py` expects video inputs and writes per-video JSON files to `results/`
  named like `<question_id>_<video_stem>_<model>.json`.
- `create_gold_standard_gemini.py` expects a valid Gemini API key
  (`GEMINI_API_KEY` or `GOOGLE_API_KEY`) available in the environment.
