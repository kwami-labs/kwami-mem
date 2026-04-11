"""LongMemEval benchmark runner for kwami-mem.

Ingests LongMemEval conversation histories into kwami-mem,
retrieves relevant memories for each question, generates answers
using Gemini, and outputs results compatible with LongMemEval's
evaluation format.

Uses batch embedding and rate limiting to stay within API quotas.

Usage:
    uv run python benchmarks/longmemeval/run_benchmark.py \
        --dataset benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
        --output benchmarks/longmemeval/results/kwami_mem_results.jsonl \
        --top-k 10 \
        --max-questions 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from google import genai
from google.genai import types

from kwami_mem.embedding.gemini import GeminiEmbeddingProvider
from kwami_mem.models import MemoryEntry, MemoryType, Modality, Role
from kwami_mem.storage.qdrant import QdrantStorage
from kwami_mem.utils.hashing import content_hash


# ---------------------------------------------------------------------------
# Rate-limited batch embedding
# ---------------------------------------------------------------------------

BATCH_SIZE = 90  # Stay under 100 req/min free tier
RATE_LIMIT_PAUSE = 62.0  # Seconds to wait between batches


async def embed_texts_batched(
    embedder: GeminiEmbeddingProvider,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    pause_seconds: float = RATE_LIMIT_PAUSE,
) -> list[list[float]]:
    """Embed texts in batches with rate limiting.

    The Gemini free tier allows 100 embed_content requests per minute.
    We batch texts and pause between batches to avoid 429 errors.
    """
    all_vectors: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(texts), batch_size):
        batch = texts[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        if batch_idx > 0:
            print(
                f"    [rate limit] waiting {pause_seconds:.0f}s before batch "
                f"{batch_num}/{total_batches}...",
                flush=True,
            )
            await asyncio.sleep(pause_seconds)

        # Embed the batch — each call embeds one text (Gemini embed_content
        # accepts a list of contents, so we send the whole batch in one call)
        retries = 0
        while retries < 3:
            try:
                vectors = await embedder.embed_texts(
                    batch, task_type="RETRIEVAL_DOCUMENT"
                )
                all_vectors.extend(vectors)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    retries += 1
                    wait = pause_seconds * retries
                    print(
                        f"    [429] rate limited, retrying in {wait:.0f}s "
                        f"(attempt {retries}/3)...",
                        flush=True,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

    return all_vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_answer_prompt(question: str, context: str, question_date: str) -> str:
    """Build the prompt for generating an answer from retrieved context."""
    return (
        "You are a helpful assistant with access to a user's conversation history. "
        "Use the following retrieved conversation memories to answer the user's question. "
        "If the information is not available in the memories, say that you don't have "
        "enough information to answer.\n\n"
        f"Current date: {question_date}\n\n"
        f"## Retrieved Memories\n{context}\n\n"
        f"## Question\n{question}\n\n"
        "Answer concisely and directly."
    )


# ---------------------------------------------------------------------------
# Main benchmark pipeline
# ---------------------------------------------------------------------------

async def run_single_question(
    question_data: dict,
    gemini_api_key: str,
    top_k: int = 10,
    embedding_dims: int = 768,
    embedding_model: str = "gemini-embedding-2-preview",
    generation_model: str = "gemini-2.5-flash",
) -> dict:
    """Run benchmark on a single LongMemEval question.

    Uses batch embedding to minimize API calls:
    1. Collects all turn texts
    2. Batch-embeds them (with rate limiting)
    3. Bulk-upserts into Qdrant
    4. Searches for relevant memories
    5. Generates answer with Gemini
    """
    qid = question_data["question_id"]
    question = question_data["question"]
    question_date = question_data["question_date"]
    sessions = question_data["haystack_sessions"]
    session_ids = question_data["haystack_session_ids"]
    session_dates = question_data["haystack_dates"]
    answer_session_ids = set(question_data["answer_session_ids"])

    # --- 1. Create storage + embedder directly (bypass KwamiMemory overhead) ---
    storage = QdrantStorage(
        url=None,  # In-memory
        collection_name=f"bench_{qid}",
        vector_size=embedding_dims,
    )
    await storage.initialize()

    embedder = GeminiEmbeddingProvider(
        api_key=gemini_api_key,
        model=embedding_model,
        dimensions=embedding_dims,
    )

    # --- 2. Collect all turns ---
    turn_texts: list[str] = []
    turn_entries: list[MemoryEntry] = []
    turn_session_ids: list[str] = []
    turn_has_answer: list[bool] = []

    global_turn_idx = 0
    for session_idx, (session, sid, sdate) in enumerate(
        zip(sessions, session_ids, session_dates)
    ):
        conv_id = f"session_{sid}"
        for turn_idx, turn in enumerate(session):
            content = turn["content"]
            role_str = turn["role"]
            has_answer = turn.get("has_answer", False)

            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content=content,
                role=Role(role_str),
                conversation_id=conv_id,
                user_id="benchmark",
                memory_type=MemoryType.EPISODIC,
                modality=Modality.TEXT,
                turn_index=global_turn_idx,
                content_hash=content_hash(content, conv_id, global_turn_idx),
            )

            turn_texts.append(content)
            turn_entries.append(entry)
            turn_session_ids.append(sid)
            turn_has_answer.append(has_answer)
            global_turn_idx += 1

    total_turns = len(turn_texts)

    # --- 3. Batch embed all turns ---
    vectors = await embed_texts_batched(embedder, turn_texts)

    # --- 4. Bulk upsert into Qdrant ---
    await storage.upsert(turn_entries, vectors)

    # --- 5. Embed query and search ---
    query_vector = await embedder.embed_text(question, task_type="RETRIEVAL_QUERY")
    search_results = await storage.search(query_vector, limit=top_k)

    # --- 6. Compute retrieval metrics ---
    # Map entry IDs to session IDs and has_answer flags
    entry_id_to_idx = {e.id: i for i, e in enumerate(turn_entries)}

    retrieved_session_ids_set = set()
    retrieved_has_answer_count = 0
    total_evidence_turns = sum(1 for h in turn_has_answer if h)

    mrr = 0.0
    for rank, r in enumerate(search_results, 1):
        idx = entry_id_to_idx.get(r.entry.id)
        if idx is not None:
            sid = turn_session_ids[idx]
            retrieved_session_ids_set.add(sid)
            if turn_has_answer[idx]:
                retrieved_has_answer_count += 1
                if mrr == 0.0 and (sid in answer_session_ids):
                    mrr = 1.0 / rank
        # Also check by session membership for MRR
        if mrr == 0.0 and idx is not None:
            sid = turn_session_ids[idx]
            if sid in answer_session_ids:
                mrr = 1.0 / rank

    session_recall = (
        len(answer_session_ids & retrieved_session_ids_set) / len(answer_session_ids)
        if answer_session_ids
        else 0.0
    )
    turn_recall = (
        retrieved_has_answer_count / total_evidence_turns
        if total_evidence_turns > 0
        else 0.0
    )

    # --- 7. Generate answer ---
    context_parts = []
    for r in search_results:
        context_parts.append(
            f"[{r.entry.role.value}] (score: {r.score:.3f}): {r.entry.content}"
        )
    context_str = "\n".join(context_parts) if context_parts else "(no relevant memories found)"

    prompt = build_answer_prompt(question, context_str, question_date)

    # Generate with Gemini (retry on rate limit)
    client = genai.Client(api_key=gemini_api_key)
    hypothesis = ""
    for attempt in range(3):
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=generation_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=512,
                    ),
                ),
            )
            hypothesis = response.text.strip() if response.text else ""
            break
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                await asyncio.sleep(15 * (attempt + 1))
            else:
                hypothesis = f"[Generation error: {e}]"

    return {
        "question_id": qid,
        "hypothesis": hypothesis,
        "retrieval_metrics": {
            "session_recall": session_recall,
            "turn_recall": turn_recall,
            "mrr": mrr,
            "retrieved_session_ids": list(retrieved_session_ids_set),
            "total_turns": total_turns,
            "total_evidence_turns": total_evidence_turns,
            "top_k": top_k,
        },
    }


async def run_benchmark(
    dataset_path: str,
    output_path: str,
    top_k: int = 10,
    max_questions: int | None = None,
    embedding_dims: int = 768,
    embedding_model: str = "gemini-embedding-2-preview",
    generation_model: str = "gemini-2.5-flash",
) -> None:
    """Run the full LongMemEval benchmark."""
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("ERROR: GEMINI_API_KEY environment variable is required.")
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        dataset = json.load(f)

    if max_questions and max_questions < len(dataset):
        dataset = dataset[:max_questions]

    # Count total turns
    total_turns = sum(
        len(turn)
        for q in dataset
        for turn in q["haystack_sessions"]
    )

    print(f"Running benchmark on {len(dataset)} questions (top_k={top_k}, dims={embedding_dims})")
    print(f"Total turns to embed: {total_turns:,}")
    print(f"Embedding model: {embedding_model}")
    print(f"Generation model: {generation_model}")
    print(f"Rate limit: {BATCH_SIZE} embeddings/batch, {RATE_LIMIT_PAUSE:.0f}s pause")
    print("-" * 60)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    results = []
    total_session_recall = 0.0
    total_turn_recall = 0.0
    total_mrr = 0.0
    start_time = time.time()

    with open(output_path, "w") as out_f:
        for i, question_data in enumerate(dataset):
            qid = question_data["question_id"]
            qtype = question_data["question_type"]
            n_sessions = len(question_data["haystack_sessions"])
            n_turns = sum(len(s) for s in question_data["haystack_sessions"])

            print(
                f"\n[{i+1}/{len(dataset)}] {qid} ({qtype}, "
                f"{n_sessions} sessions, {n_turns} turns)",
                flush=True,
            )

            try:
                result = await run_single_question(
                    question_data,
                    gemini_api_key=gemini_api_key,
                    top_k=top_k,
                    embedding_dims=embedding_dims,
                    embedding_model=embedding_model,
                    generation_model=generation_model,
                )
                results.append(result)

                metrics = result["retrieval_metrics"]
                total_session_recall += metrics["session_recall"]
                total_turn_recall += metrics["turn_recall"]
                total_mrr += metrics["mrr"]

                print(
                    f"  → SR={metrics['session_recall']:.2f} "
                    f"TR={metrics['turn_recall']:.2f} "
                    f"MRR={metrics['mrr']:.2f} "
                    f"| Answer: {result['hypothesis'][:80]}..."
                )

                # Write to output file (JSONL)
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

            except Exception as e:
                print(f"  → ERROR: {e}")
                placeholder = {
                    "question_id": qid,
                    "hypothesis": f"Error: {e}",
                    "retrieval_metrics": {
                        "session_recall": 0.0,
                        "turn_recall": 0.0,
                        "mrr": 0.0,
                        "retrieved_session_ids": [],
                        "total_turns": 0,
                        "total_evidence_turns": 0,
                        "top_k": top_k,
                    },
                }
                out_f.write(json.dumps(placeholder) + "\n")
                out_f.flush()
                results.append(placeholder)

    elapsed = time.time() - start_time
    n = len(results)
    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETE — {n} questions in {elapsed:.1f}s ({elapsed/n:.1f}s/question)")
    print(f"  Avg Session Recall@{top_k}: {total_session_recall/n:.4f}")
    print(f"  Avg Turn Recall@{top_k}:    {total_turn_recall/n:.4f}")
    print(f"  Avg MRR:                     {total_mrr/n:.4f}")
    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark on kwami-mem"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmarks/longmemeval/data/longmemeval_s_cleaned.json",
        help="Path to LongMemEval dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/longmemeval/results/kwami_mem_results.jsonl",
        help="Output path for results JSONL",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieved memories per question (default: 10)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to run (default: all)",
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        default=768,
        help="Embedding dimensionality (default: 768)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="gemini-embedding-2-preview",
        help="Gemini embedding model (default: gemini-embedding-2-preview)",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini generation model for answers (default: gemini-2.5-flash)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            dataset_path=args.dataset,
            output_path=args.output,
            top_k=args.top_k,
            max_questions=args.max_questions,
            embedding_dims=args.embedding_dims,
            embedding_model=args.embedding_model,
            generation_model=args.generation_model,
        )
    )


if __name__ == "__main__":
    main()
