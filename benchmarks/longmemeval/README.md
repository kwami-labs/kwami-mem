# LongMemEval Benchmark for kwami-mem

Evaluates kwami-mem's retrieval and answer accuracy on the [LongMemEval](https://github.com/xiaowu0162/longmemeval) benchmark (ICLR 2025).

## Setup

### 1. Download the dataset

```bash
cd benchmarks/longmemeval/data
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

### 2. Set your API key

```bash
export GEMINI_API_KEY=your-key-here
```

## Running the Benchmark

### Quick smoke test (5 questions)

```bash
uv run python benchmarks/longmemeval/run_benchmark.py \
  --max-questions 5 \
  --output benchmarks/longmemeval/results/smoke_test.jsonl
```

### Full run (500 questions)

```bash
uv run python benchmarks/longmemeval/run_benchmark.py \
  --output benchmarks/longmemeval/results/full_run.jsonl
```

### Options

| Flag | Default | Description |
|:---|:---|:---|
| `--dataset` | `data/longmemeval_s_cleaned.json` | Dataset path |
| `--output` | `results/kwami_mem_results.jsonl` | Output JSONL path |
| `--top-k` | `10` | Number of retrieved memories per question |
| `--max-questions` | all | Cap the number of questions |
| `--embedding-dims` | `768` | Embedding dimensionality |
| `--embedding-model` | `gemini-embedding-2-preview` | Embedding model |
| `--generation-model` | `gemini-2.0-flash` | Answer generation model |
| `--disable-rate-limit` | `False` | Disable artificial 62s pauses between embedding batches (for paid-tier accounts) |

### Rate Limits

By default, the benchmark script adds a 62-second pause after every 90 embeddings to comply with the Google Gemini free tier rate limit of 100 requests per minute. If you are using a paid Google AI Studio API key, you should use the `--disable-rate-limit` flag to bypass these pauses and run the benchmark at full speed.

## Evaluation

### Retrieval metrics only (no API calls)

```bash
uv run python benchmarks/longmemeval/evaluate.py \
  --results benchmarks/longmemeval/results/full_run.jsonl \
  --mode retrieval
```

Reports:
- **Session Recall@K** — % of evidence sessions found in top-K
- **Turn Recall@K** — % of evidence turns found
- **MRR** — Mean Reciprocal Rank
- All metrics broken down by question type

### Full evaluation (with LLM judge)

```bash
uv run python benchmarks/longmemeval/evaluate.py \
  --results benchmarks/longmemeval/results/full_run.jsonl \
  --mode full
```

Additionally reports answer accuracy using Gemini as judge, following LongMemEval's task-specific evaluation protocol.

## Benchmark Details

LongMemEval tests 5 core long-term memory abilities:

| Ability | Types | Count |
|:---|:---|:---|
| Information Extraction | `single-session-user`, `single-session-assistant`, `single-session-preference` | 156 |
| Multi-Session Reasoning | `multi-session` | 133 |
| Temporal Reasoning | `temporal-reasoning` | 133 |
| Knowledge Updates | `knowledge-update` | 78 |
| Abstention | `*_abs` variants across types | 30 |

Each question includes ~40 conversation sessions (~115k tokens) as a "haystack" in which the evidence is hidden.
