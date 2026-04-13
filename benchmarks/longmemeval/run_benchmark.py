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

from kwami_mem.embedding.local import LocalEmbeddingProvider
from kwami_mem.embedding.gemini import GeminiEmbeddingProvider
from kwami_mem.embedding.sparse import SparseEmbeddingProvider
from kwami_mem.embedding.rerank import CrossEncoderReranker
from qdrant_client.models import Filter, FieldCondition, MatchValue
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
        "You are an expert AI assistant evaluating long-term conversational memory. "
        "Use ONLY the following retrieved memories to answer the user's question.\n\n"
        "RULES:\n"
        "1. STRICT ABSTENTION: If the answer is not explicitly stated in the retrieved memories, you MUST output exactly: \"I don't have enough information to answer.\"\n"
        "2. TEMPORAL REASONING: Use the provided memory dates (in brackets) to determine chronological order and understand what is most recent or happened first. The current date is provided below.\n"
        "3. PREFERENCES: When determining preferences, distinguish between passing mentions and explicit statements of preference.\n\n"
        f"Current date: {question_date}\n\n"
        f"## Retrieved Memories (Format: [Date] [Role] (score): Content)\n{context}\n\n"
        f"## Question\n{question}\n\n"
        "Answer concisely and directly. If the information is not present, remember Rule 1."
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
    disable_rate_limit: bool = False,
    use_hybrid: bool = True,
    shared_embedder: EmbeddingProvider | None = None,
    shared_sparse_embedder: SparseEmbeddingProvider | None = None,
    shared_reranker: CrossEncoderReranker | None = None,
    qdrant_url: str | None = None,
    openai_base_url: str | None = None,
) -> dict:
    """Run benchmark on a single LongMemEval question."""
    qid = question_data["question_id"]
    question = question_data["question"]
    question_date = question_data["question_date"]
    sessions = question_data["haystack_sessions"]
    session_ids = question_data["haystack_session_ids"]
    session_dates = question_data["haystack_dates"]
    answer_session_ids = set(question_data["answer_session_ids"])

    # --- 1. Create storage + embedder ---
    col_name = f"longmemeval_{embedding_dims}_hybrid" if use_hybrid else f"longmemeval_{embedding_dims}_dense"
    storage = QdrantStorage(
        url=qdrant_url,
        collection_name=col_name,
        vector_size=embedding_dims,
        use_hybrid=use_hybrid,
    )
    await storage.initialize()

    embedder = shared_embedder or GeminiEmbeddingProvider(
        api_key=gemini_api_key,
        model=embedding_model,
        dimensions=embedding_dims,
    )
    
    sparse_embedder = shared_sparse_embedder
    reranker = shared_reranker
    if use_hybrid and not sparse_embedder:
        sparse_embedder = SparseEmbeddingProvider(model="Qdrant/bm25")
    if use_hybrid and not reranker:
        reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

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
                user_id=qid,
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

    # --- 3. Check if already cached ---
    skip_embedding = False
    if qdrant_url:
        existing_recs = await storage._run_sync(
            storage._client.count,
            collection_name=col_name,
            count_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=qid))])
        )
        if existing_recs.count >= total_turns:
            skip_embedding = True

    if not skip_embedding:
        pause_sec = 0.0 if disable_rate_limit else RATE_LIMIT_PAUSE
        vectors = await embed_texts_batched(embedder, turn_texts, pause_seconds=pause_sec)
        sparse_vectors = None
        if use_hybrid and turn_texts:
            sparse_vectors = await sparse_embedder.embed_texts(turn_texts)
        await storage.upsert(turn_entries, vectors, sparse_vectors=sparse_vectors)

    # --- 4. Search ---
    query_vector = await embedder.embed_text(question, task_type="RETRIEVAL_QUERY")
    sparse_query_vec = None
    if use_hybrid:
        sparse_query_vec = await sparse_embedder.embed_text(question)
        search_limit = top_k * 5
    else:
        search_limit = top_k

    search_results = await storage.search(
        query_vector,
        sparse_query_vector=sparse_query_vec,
        limit=search_limit,
        filters={"user_id": qid},
    )

    if use_hybrid and search_results:
        search_results = reranker.rerank(question, search_results, top_k=top_k)

    # --- 5. Metrics ---
    retrieved_session_ids_set = set()
    retrieved_has_answer_count = 0
    total_evidence_turns = sum(1 for h in turn_has_answer if h)
    mrr = 0.0
    for rank, r in enumerate(search_results, 1):
        idx = r.entry.turn_index
        sid = turn_session_ids[idx]
        retrieved_session_ids_set.add(sid)
        if turn_has_answer[idx]:
            retrieved_has_answer_count += 1
            if mrr == 0.0 and (sid in answer_session_ids):
                mrr = 1.0 / rank
        if mrr == 0.0 and turn_session_ids[idx] in answer_session_ids:
            mrr = 1.0 / rank

    session_recall = (len(answer_session_ids & retrieved_session_ids_set) / len(answer_session_ids)) if answer_session_ids else 0.0
    turn_recall = (retrieved_has_answer_count / total_evidence_turns) if total_evidence_turns > 0 else 0.0

    # --- 6. Generate ---
    sid_to_date = dict(zip(session_ids, session_dates))
    context_parts = [f"[{sid_to_date.get(turn_session_ids[r.entry.turn_index], 'Unknown')}] [{r.entry.role.value}] (score: {r.score:.3f}): {r.entry.content}" for r in search_results]
    context_str = "\n".join(context_parts) if context_parts else "(no relevant memories found)"
    prompt = build_answer_prompt(question, context_str, question_date)

    hypothesis = ""
    if openai_base_url:
        import openai
        o_client = openai.Client(api_key="local", base_url=openai_base_url)
        for attempt in range(3):
            try:
                response = o_client.chat.completions.create(model=generation_model, messages=[{"role": "user", "content": prompt}], max_tokens=64, temperature=0.0)
                hypothesis = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt == 2: hypothesis = f"ERROR: {e}"
                await asyncio.sleep(2)
    else:
        client = genai.Client(api_key=gemini_api_key)
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=generation_model, contents=prompt, config={"temperature": 0.0, "max_output_tokens": 64})
                hypothesis = response.text.strip() if response.text else "(no output)"
                break
            except Exception as e:
                if attempt == 2: hypothesis = f"ERROR: {e}"
                await asyncio.sleep(5)

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
    embedding_dims: int = 768,
    embedding_model: str = "gemini-embedding-2-preview",
    generation_model: str = "gemini-2.5-flash",
    disable_rate_limit: bool = False,
    no_hybrid: bool = False,
    use_local_embeddings: bool = False,
    qdrant_url: str | None = None,
    openai_base_url: str | None = None,
    max_questions: int | None = None,
) -> None:
    """Run the full LongMemEval benchmark."""
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("ERROR: GEMINI_API_KEY environment variable is required.")
        sys.exit(1)

    with open(dataset_path) as f:
        dataset = json.load(f)
    if max_questions: dataset = dataset[:max_questions]

    if use_local_embeddings:
        shared_embedder = LocalEmbeddingProvider(model="BAAI/bge-small-en-v1.5", dimensions=384)
        embedding_dims = 384  # Force correct dimensions for BGE
    else:
        shared_embedder = GeminiEmbeddingProvider(api_key=gemini_api_key, model=embedding_model, dimensions=embedding_dims)
        
    shared_sparse_embedder = SparseEmbeddingProvider(model="Qdrant/bm25") if not no_hybrid else None
    shared_reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2") if not no_hybrid else None

    results = []
    with open(output_path, "w") as out_f:
        for i, question_data in enumerate(dataset):
            result = await run_single_question(
                question_data, gemini_api_key, top_k, embedding_dims, embedding_model, generation_model, 
                disable_rate_limit, not no_hybrid, shared_embedder, shared_sparse_embedder, shared_reranker, 
                qdrant_url, openai_base_url
            )
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmarks/longmemeval/data/longmemeval_s_cleaned.json")
    parser.add_argument("--output", default="benchmarks/longmemeval/results/kwami_mem_results.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--dims", type=int, default=768)
    parser.add_argument("--embedding-model", default="gemini-embedding-2-preview")
    parser.add_argument("--generation-model", default="gemini-2.5-flash")
    parser.add_argument("--disable-rate-limit", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--use-local-embeddings", action="store_true", help="Use local BAAI/bge-small instead of Gemini")
    parser.add_argument("--qdrant-url", type=str)
    parser.add_argument("--openai-base-url", type=str)
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        args.dataset, args.output, args.top_k, 384 if args.use_local_embeddings else args.dims, args.embedding_model, 
        args.generation_model, args.disable_rate_limit, args.no_hybrid, args.use_local_embeddings,
        args.qdrant_url, args.openai_base_url, args.max_questions
    ))


if __name__ == "__main__":
    main()
