"""MemoryBench benchmark runner for kwami-mem.

Ingests THUIR/MemoryBench conversation histories into kwami-mem,
retrieves relevant memories for each question, generates answers
using Gemini, and outputs results.

Usage:
    uv run python benchmarks/memorybench/run_benchmark.py \
        --config Locomo-0 \
        --output benchmarks/memorybench/results/locomo_0_results.jsonl \
        --top-k 10 \
        --max-questions 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

from datasets import load_dataset
from google import genai
from google.genai import types

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kwami_mem.embedding.gemini import GeminiEmbeddingProvider
from kwami_mem.models import MemoryEntry, MemoryType, Modality, Role
from kwami_mem.storage.qdrant import QdrantStorage
from kwami_mem.utils.hashing import content_hash


BATCH_SIZE = 90
RATE_LIMIT_PAUSE = 62.0


async def embed_texts_batched(
    embedder: GeminiEmbeddingProvider,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    pause_seconds: float = RATE_LIMIT_PAUSE,
) -> list[list[float]]:
    """Embed texts in batches with rate limiting."""
    all_vectors: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(texts), batch_size):
        batch = texts[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        if batch_idx > 0 and pause_seconds > 0:
            print(f"    [rate limit] waiting {pause_seconds:.0f}s before batch {batch_num}/{total_batches}...", flush=True)
            await asyncio.sleep(pause_seconds)

        retries = 0
        while retries < 3:
            try:
                vectors = await embedder.embed_texts(batch, task_type="RETRIEVAL_DOCUMENT")
                all_vectors.extend(vectors)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    retries += 1
                    wait = pause_seconds * retries if pause_seconds > 0 else 60.0
                    print(f"    [429] rate limited, retrying in {wait:.0f}s (attempt {retries}/3)...", flush=True)
                    await asyncio.sleep(wait)
                else:
                    raise

    return all_vectors


def parse_dialog(dialog_str: str) -> list[dict]:
    """Parse the dialog string into a list of session dicts."""
    try:
        messages = json.loads(dialog_str)
        text = "\n".join([m["content"] for m in messages if m["role"] == "user"])
    except Exception:
        text = dialog_str

    match = re.search(r"Context:\n(.*?)\n\nUser:", text, re.DOTALL)
    context = match.group(1) if match else text

    sessions = []
    current_session = []
    session_id = "1"
    date_str = "Unknown"

    for line in context.split('\n'):
        line = line.strip()
        if not line: continue
        
        if line.startswith("Coversation ["):
            if current_session:
                sessions.append({"id": session_id, "date": date_str, "turns": current_session})
                current_session = []
            date_str = line[len("Coversation ["):-2]
            session_id = str(len(sessions) + 1)
        elif line.startswith("Speaker "):
            idx = line.find("says : ")
            if idx != -1:
                name = line[8:idx].strip()
                content = line[idx+7:].strip()
                current_session.append({"role": name, "content": content})

    if current_session:
        sessions.append({"id": session_id, "date": date_str, "turns": current_session})

    return sessions


def build_answer_prompt(question: str, context: str) -> str:
    """Build the prompt for generating an answer."""
    return (
        "You are a helpful assistant evaluating a user's memory state. "
        "Use the following retrieved conversation memories to answer the user's question. "
        "Your goal is to answer the question concisely and correctly based on the history.\n\n"
        f"## Retrieved Memories\n{context}\n\n"
        f"## Question\n{question}\n\n"
        "Answer concisely and directly."
    )


async def run_single_question(
    qid: str,
    question: str,
    dialog_str: str,
    gemini_api_key: str,
    top_k: int = 10,
    embedding_dims: int = 768,
    embedding_model: str = "gemini-embedding-2-preview",
    generation_model: str = "gemini-2.5-flash",
    disable_rate_limit: bool = False,
) -> dict:
    """Run benchmark on a single MemoryBench question."""
    
    # Parse dialog
    sessions = parse_dialog(dialog_str)
    
    storage = QdrantStorage(url=None, collection_name=f"bench_{qid}", vector_size=embedding_dims)
    await storage.initialize()

    embedder = GeminiEmbeddingProvider(api_key=gemini_api_key, model=embedding_model, dimensions=embedding_dims)

    turn_texts: list[str] = []
    turn_entries: list[MemoryEntry] = []

    global_turn_idx = 0
    for session in sessions:
        conv_id = f"session_{session['id']}"
        date_str = session["date"]
        for turn in session["turns"]:
            content = turn["content"]
            role_str = "user" if turn["role"].lower() in ["user", "human"] else "assistant"
            
            # Prefix with the date context to help chronological understanding
            full_content = f"[{date_str}] {turn['role']}: {content}"
            
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content=full_content,
                role=Role(role_str) if role_str in ["user", "assistant", "system"] else Role.USER,
                conversation_id=conv_id,
                user_id="benchmark",
                memory_type=MemoryType.EPISODIC,
                modality=Modality.TEXT,
                turn_index=global_turn_idx,
                content_hash=content_hash(full_content, conv_id, global_turn_idx),
            )
            turn_texts.append(full_content)
            turn_entries.append(entry)
            global_turn_idx += 1

    total_turns = len(turn_texts)
    
    pause_sec = 0.0 if disable_rate_limit else RATE_LIMIT_PAUSE
    if total_turns > 0:
        vectors = await embed_texts_batched(embedder, turn_texts, pause_seconds=pause_sec)
        await storage.upsert(turn_entries, vectors)

    query_vector = await embedder.embed_text(question, task_type="RETRIEVAL_QUERY")
    search_results = await storage.search(query_vector, limit=top_k)

    context_parts = [f"[{r.entry.role.value}] (score: {r.score:.3f}): {r.entry.content}" for r in search_results]
    context_str = "\n".join(context_parts) if context_parts else "(no relevant memories found)"

    prompt = build_answer_prompt(question, context_str)
    
    client = genai.Client(api_key=gemini_api_key)
    hypothesis = ""
    for attempt in range(3):
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=generation_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=512),
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
        "question": question,
        "hypothesis": hypothesis,
        "retrieval_metrics": {
            "total_turns": total_turns,
            "top_k": top_k,
            "retrieved_contexts": context_parts,
        },
    }


async def run_benchmark(
    config_name: str,
    output_path: str,
    top_k: int = 10,
    max_questions: int | None = None,
    embedding_dims: int = 768,
    embedding_model: str = "gemini-embedding-2-preview",
    generation_model: str = "gemini-2.5-flash",
    disable_rate_limit: bool = False,
) -> None:
    """Run MemoryBench evaluation."""
    
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("ERROR: GEMINI_API_KEY environment variable is required.")
        sys.exit(1)

    print(f"Downloading dataset THUIR/MemoryBench config={config_name}...")
    dataset = load_dataset("THUIR/MemoryBench", config_name, split="test")

    if max_questions and max_questions < len(dataset):
        dataset = dataset.select(range(max_questions))

    print(f"Running benchmark on {len(dataset)} questions from {config_name}")
    print("-" * 60)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = time.time()

    with open(output_path, "w") as out_f:
        for i, sample in enumerate(dataset):
            qid = str(sample.get("test_idx", i))
            question = sample.get("origin_question") or sample.get("question", "")
            dialog_str = sample.get("dialog_bm25_dialog", "")
            info = sample.get("info", "{}")

            print(f"\n[{i+1}/{len(dataset)}] ID: {qid}", flush=True)

            try:
                result = await run_single_question(
                    qid=qid,
                    question=question,
                    dialog_str=dialog_str,
                    gemini_api_key=gemini_api_key,
                    top_k=top_k,
                    embedding_dims=embedding_dims,
                    embedding_model=embedding_model,
                    generation_model=generation_model,
                    disable_rate_limit=disable_rate_limit,
                )
                
                # Attach ground truth info for evaluation script
                try:
                    result["info"] = json.loads(info)
                except Exception:
                    result["info"] = {"raw": info}

                results.append(result)

                print(f"  → Truth: {result['info'].get('golden_answer', '')}")
                print(f"  → Answer: {result['hypothesis'][:80]}...")

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

            except Exception as e:
                print(f"  → ERROR: {e}")
                err_result = {"question_id": qid, "error": str(e)}
                out_f.write(json.dumps(err_result) + "\n")
                out_f.flush()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETE — {len(results)} questions in {elapsed:.1f}s")
    print(f"Results saved to: {output_path}")


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run THUIR/MemoryBench on kwami-mem")
    parser.add_argument("--config", type=str, default="Locomo-0", help="MemoryBench config to run (e.g. Locomo-0, DialSim-friends)")
    parser.add_argument("--output", type=str, default="benchmarks/memorybench/results/memorybench_results.jsonl", help="Output path for results JSONL")
    parser.add_argument("--top-k", type=int, default=10, help="Number of retrieved memories per question (default: 10)")
    parser.add_argument("--max-questions", type=int, default=None, help="Maximum number of questions to run (default: all)")
    parser.add_argument("--embedding-dims", type=int, default=768, help="Embedding dimensionality (default: 768)")
    parser.add_argument("--embedding-model", type=str, default="gemini-embedding-2-preview", help="Gemini embedding model (default: gemini-embedding-2-preview)")
    parser.add_argument("--generation-model", type=str, default="gemini-2.5-flash", help="Gemini generation model for answers (default: gemini-2.5-flash)")
    parser.add_argument("--disable-rate-limit", action="store_true", help="Disable artificial pauses between batches (for paid tier accounts)")

    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            config_name=args.config,
            output_path=args.output,
            top_k=args.top_k,
            max_questions=args.max_questions,
            embedding_dims=args.embedding_dims,
            embedding_model=args.embedding_model,
            generation_model=args.generation_model,
            disable_rate_limit=args.disable_rate_limit,
        )
    )

if __name__ == "__main__":
    main()
