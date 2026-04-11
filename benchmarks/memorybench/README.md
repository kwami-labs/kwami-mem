# MemoryBench Benchmark for kwami-mem

Evaluates kwami-mem's retrieval and answer accuracy on the [MemoryBench](https://huggingface.co/datasets/THUIR/MemoryBench) dataset, evaluating capabilities like user-feedback simulation and continual learning.

## Setup

1. Install development dependencies which include `datasets`:
```bash
uv pip install -e ".[dev]"
```
*(We use the Hugging Face `datasets` library to securely download the required configuration).*

2. Set your API key for Google Gemini:
```bash
export GEMINI_API_KEY=your-key-here
```

## Running the Benchmark

MemoryBench contains several dataset configurations (e.g., `Locomo-0`, `DialSim-friends`). You can run a benchmark using the `--config` flag.

### Quick smoke test
```bash
uv run python benchmarks/memorybench/run_benchmark.py \
  --config Locomo-0 \
  --max-questions 5 \
  --output benchmarks/memorybench/results/locomo_0_results.jsonl
```

### Full run
```bash
uv run python benchmarks/memorybench/run_benchmark.py \
  --config Locomo-0 \
  --output benchmarks/memorybench/results/locomo_0_results.jsonl
```

### Options

| Flag | Default | Description |
|:---|:---|:---|
| `--config` | `Locomo-0` | MemoryBench config variant to run (e.g. `DialSim-friends`) |
| `--output` | `results/memorybench_results.jsonl` | Output JSONL path |
| `--top-k` | `10` | Number of retrieved memories per question |
| `--max-questions` | all | Cap the number of questions |
| `--embedding-dims` | `768` | Embedding dimensionality |
| `--embedding-model` | `gemini-embedding-2-preview` | Embedding model |
| `--generation-model` | `gemini-2.5-flash` | Answer generation model |
| `--disable-rate-limit` | `False` | Disable 62s pauses between embedding batches (for paid accounts) |


## Evaluation

Once complete, evaluate the EM (Exact Match) and F1 token intersection against the ground truth labels:

```bash
uv run python benchmarks/memorybench/evaluate.py \
  --results benchmarks/memorybench/results/locomo_0_results.jsonl
```
